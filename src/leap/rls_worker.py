"""Source-precision RLS feedback on the unchanged FeedbackWorker thread/queue contract.

The prior mean is the deployed source head, including its export rounding; the prior
precision is the source-feature Gram matrix plus the feature penalty, with an
unpenalized intercept. Every eight applied completions produce a Woodbury update
in square-root target coordinates. P uses FP32 (about 67 MB at D=4096), while small
solves and coefficients use FP64. Completion history is not replayed. The direct
batch reference is anchored to this deployed prior, rather than a new source-label fit.
"""
import hashlib,math,threading,time
from collections import Counter
import queue
import numpy as np
from ._feedback import FeedbackCore,Snapshot,FeedbackWorker,inverse

class RLSFeedbackCore(FeedbackCore):
    def __init__(self,w,b,mean,scale,capacity,prior,mode='full',sparse_relative=0.05):
        super().__init__(w,b,mean,scale,capacity);self.method='ttt_rls'
        coef=np.asarray(prior['coef'],np.float64);P=np.asarray(prior['P'])
        if P.dtype!=np.float32 or P.shape!=(len(self.w)+1,)*2 or coef.shape!=(len(self.w)+1,):raise ValueError('RLS prior contract')
        if np.abs(np.asarray(prior['mean'])-self.mean).max()>0 or np.abs(np.asarray(prior['scale'])-self.scale).max()>0:raise ValueError('RLS prior uses other standardization')
        # Prior mean must be exactly the deployed head (same z0 for every request).
        if np.abs(coef[:-1]-self.w*self.scale).max()>1e-12*max(1.,np.abs(coef).max()) or abs(coef[-1]-(self.b+self.mean@self.w))>1e-9:raise ValueError('RLS prior mean differs from deployed head')
        self.P=np.array(P,copy=True);self.coef=coef.copy();self.raw={};self.pending=[];self.updates=None  # None: synchronous (unit tests)
        if mode not in ('full','shadow','sparse'):raise ValueError('Unknown RLS publication mode')
        # shadow: identical updates, no learned score is ever published (cost isolation); sparse: republish only moved scores
        self.mode=mode;self.sparse_relative=sparse_relative;self.published_total={}
    def u(self,x):return np.append((np.asarray(x,np.float64)-self.mean)/self.scale,1.)
    def features(self,keys,x,initial):
        for key,row in zip(keys,x):
            if tuple(key) in self.raw:raise RuntimeError('Duplicate cached feature')
            self.raw[tuple(key)]=self.u(row)
        return super().features(keys,x,initial)
    def drain(self):
        while self.feedback and self.feedback[0][0] in self.entries:
            key,length,ordinal,observed=self.feedback.popleft();self.entries.pop(key);self.published_total.pop(key,None);self.completed+=1
            if ordinal!=self.completed:raise RuntimeError('Completion sequence gap')
            self.pending.append((self.raw.pop(key),math.sqrt(length),ordinal))
            if self.completed%8==0:self.fit(observed)
    def fit(self,last_observed_ns):
        # Hand the batch to the updater thread; initial readouts never queue behind the O(d^2) update.
        U=np.stack([r[0] for r in self.pending]);t=np.array([r[1] for r in self.pending]);first=self.pending[0][2];self.pending=[]
        job=(U,t,first,self.completed,last_observed_ns,time.perf_counter_ns())
        if self.updates is None:delta,timing=self.update(U,t);self.publish(delta,*job[2:],timing)
        else:self.updates.put(job)
    def update(self,U,t):
        """Updater thread only: exact rank-k Woodbury step on (P, coef)."""
        start=time.perf_counter_ns();cpu=time.thread_time_ns()
        PU=(self.P@U.astype(np.float32).T).astype(np.float64);S=np.eye(len(U))+U@PU
        K=np.linalg.solve(S,PU.T).T;innovation=t-U@self.coef
        coef=self.coef+K@innovation;self.P-=(K@PU.T).astype(np.float32)
        if not np.isfinite(coef).all() or not np.isfinite(S).all():raise RuntimeError('RLS numeric gate failed')
        self.coef=coef;delta=coef.copy();delta.setflags(write=False)
        return delta,dict(fit_start_ns=start,fit_end_ns=time.perf_counter_ns(),fit_ns=time.perf_counter_ns()-start,updater_thread_cpu_ns=time.thread_time_ns()-cpu,
            innovation_abs_mean=float(np.abs(innovation).mean()))
    def publish(self,delta,first,completed,last_observed_ns,handoff_ns,timing):
        """Worker thread: publish the finished version and rescore cached live features."""
        begin=time.perf_counter_ns();published=time.perf_counter_ns()
        self.snapshot=Snapshot(self.snapshot.version+1,completed,0.,delta,published)
        self.event('ttt_version_published',version=self.snapshot.version,completed=completed,window_start=1,window_end=completed,window_count=completed,
            batch_start=first,batch_count=completed-first+1,last_feedback_observed_ns=last_observed_ns,handoff_ns=handoff_ns,published_ns=published,
            snapshot_copy_ns=published-begin,kkt_relative=None,state_bytes=int(self.P.nbytes+self.coef.nbytes),
            state_sha256=hashlib.sha256(np.asarray(delta,dtype='<f8').tobytes()).hexdigest(),**timing)
        scores=self.scores(list(self.entries),rescore=True)
        if scores is not None:self.events.append(scores)
    def scores(self,keys,rescore=False):
        if self.mode=='shadow':return None
        start=time.perf_counter_ns();s=self.snapshot;rows=[]
        for key in keys:
            r=self.entries.get(key)
            if r is None:continue
            z=r['z0'] if s.delta is None else float(self.raw[key]@s.delta);total=float(inverse(z))
            last=self.published_total.get(key)
            if rescore and self.mode=='sparse' and last is not None and abs(total-last)<=self.sparse_relative*last:continue
            self.published_total[key]=total
            rows.append(dict(key=list(key),initial_total=r['initial'],latest_total=total))
        if not rows:return None
        return dict(type='ttt_scores_published',version=s.version,completed=s.completed,version_published_ns=s.published_ns,
            score_start_ns=start,scores_published_ns=time.perf_counter_ns(),rescore_ns=time.perf_counter_ns()-start,rows=rows)
    def cancel(self,key):
        super().cancel(key);self.raw.pop(tuple(key),None);self.published_total.pop(tuple(key),None)

READY=object()

class _Inputs(queue.Queue):
    """Worker input queue that applies finished RLS versions on the worker thread between jobs.
    Shutdown (None) first drains every in-flight update, so no applied label is lost."""
    def bind(self,worker):self.worker=worker
    def get(self,block=True,timeout=None):
        while True:
            item=super().get(block,timeout)
            if item is READY:self.task_done();self.worker._publish_ready();continue
            if item is None:self.worker._finish_updates()
            return item

class RLSWorker(FeedbackWorker):
    """FeedbackWorker with the RLS core. Original queue/thread/DMA/publication loop (_run) is inherited unchanged;
    a second CPU thread performs the Woodbury updates and wakes the worker to publish them in order."""
    def __init__(self,head,w,b,mean,scale,prior,capacity=2048,mode='full',sparse_relative=0.05):
        self.head=head;self.core=RLSFeedbackCore(w,b,mean,scale,capacity,prior,mode,sparse_relative);self.core.updates=queue.Queue()
        self.inputs=_Inputs(capacity*2+16);self.inputs.bind(self);self.results=queue.Queue(capacity*4+32);self.ready=queue.Queue()
        self.closed=False;self.fatal=None;self.counters=Counter(submitted=0,processed=0,rls_updates=0)
        self.updater=threading.Thread(target=self._update_loop,name='sqrt-ttt-rls-update',daemon=True);self.updater.start()
        self.thread=threading.Thread(target=self._run,name='sqrt-ttt-cpu',daemon=True);self.thread.start()
    def _update_loop(self):
        try:
            while True:
                job=self.core.updates.get()
                if job is None:break
                U,t,first,completed,observed,handoff=job
                delta,timing=self.core.update(U,t);self.ready.put((delta,first,completed,observed,handoff,timing))
                self.inputs.put(READY)
        except BaseException as exc:self.fatal=self.fatal or 'RLS updater: '+repr(exc);self.ready.put(exc);self.inputs.put(READY)
    def _publish_ready(self):
        while True:
            try:item=self.ready.get_nowait()
            except queue.Empty:return
            if isinstance(item,BaseException):raise item
            self.core.publish(*item);self.counters['rls_updates']+=1
            for row in self.core.events:self.emit(dict(row,kind='adaptation'))
            self.core.events.clear()
    def _finish_updates(self):
        self.core.updates.put(None);self.updater.join(120)
        if self.updater.is_alive():raise TimeoutError('RLS updater did not finish')
        self._publish_ready()
