"""Bounded prediction-worker lifecycle used by completion-supervised RLS."""
from collections import deque
from dataclasses import dataclass
import queue, time, math, hashlib
import numpy as np


def inverse(z):
    return np.clip(np.clip(z, 1., math.sqrt(2048.)) ** 2, 1., 2048.)


@dataclass(frozen=True)
class Snapshot:
    version: int
    completed: int
    offset: float
    delta: object
    published_ns: int


class FeedbackCore:
    def __init__(self, w, b, mean, scale, capacity):
        self.w = np.array(w, dtype=np.float64, copy=True)
        self.b = float(b)
        self.mean = np.array(mean, copy=True)
        self.scale = np.maximum(scale, 1e-8)
        self.capacity = capacity
        self.entries = {}
        self.feedback = deque()
        self.completed = self.received = 0
        self.snapshot = Snapshot(0, 0, 0., None, time.perf_counter_ns())
        self.events = []

    def event(self, kind, **kw):
        self.events.append(dict(type=kind, **kw))

    def features(self, keys, x, initial):
        if x.ndim != 2 or x.shape[1] != len(self.w) or len(keys) != len(x) or not np.isfinite(x).all():
            raise ValueError('Feature contract')
        prepared = time.perf_counter_ns()
        for key, row, value in zip(keys, x, initial):
            key = tuple(key)
            if key in self.entries or len(self.entries) >= self.capacity:
                raise RuntimeError('Duplicate/overflow cached feature')
            z0 = float(row.astype(np.float64) @ self.w + self.b)
            if abs(float(inverse(z0)) - float(value)) > 1e-7:
                raise ValueError('Native source vs raw sqrt dot mismatch')
            self.entries[key] = dict(z0=z0, initial=float(value))
        self.event('ttt_feature_cached', keys=[list(k) for k in keys],
                   feature_prepare_ns=time.perf_counter_ns()-prepared,
                   cached=len(self.entries), cached_array_bytes=0)
        self.drain()
        return self.scores([tuple(k) for k in keys]) if self.snapshot.version else None

    def completion(self,key,length,ordinal,observed_ns):
        key=tuple(key)
        if type(length)!=int or not 1<=length<=2048 or ordinal!=self.received+1:raise ValueError('Invalid actual completion/order')
        if len(self.feedback)>=self.capacity:raise RuntimeError('Pending completion capacity exceeded')
        self.received=ordinal;self.feedback.append((key,length,ordinal,observed_ns))
        self.event('ttt_completion_received',key=list(key),length=length,ordinal=ordinal,observed_ns=observed_ns,worker_received_ns=time.perf_counter_ns())
        self.drain()
    def cancel(self,key):
        key=tuple(key)
        if any(r[0]==key for r in self.feedback):raise RuntimeError('Cannot cancel an actual completion awaiting feature')
        self.entries.pop(key,None)


class FeedbackWorker:
    """Queue and lifecycle methods; RLSWorker supplies the core and threads."""
    def put(self,kind,**kw):
        if self.closed or self.fatal:raise RuntimeError('TTT worker closed/failed')
        try:self.inputs.put_nowait(dict(kind=kind,enqueued_ns=time.perf_counter_ns(),**kw))
        except queue.Full:raise RuntimeError('Bounded TTT queue full; no silent feedback loss')
        self.counters['submitted']+=1
    def submit(self,ticket,wait_ready,array,context,keys):
        self.put('feature',ticket=ticket,wait_ready=wait_ready,array=array,context=context,keys=keys);return True
    def completion(self,key,length,ordinal,observed_ns):self.put('completion',key=key,length=length,ordinal=ordinal,observed_ns=observed_ns)
    def cancel(self,key):self.put('cancel',key=key)
    def emit(self,row):self.results.put_nowait(row)
    def _run(self):
        try:
            while True:
                item=self.inputs.get()
                if item is None:self.inputs.task_done();break
                start=time.perf_counter_ns();cpu=time.thread_time_ns();kind=item['kind']
                if kind=='feature':
                    result=dict(ticket=item['ticket'],status='ok',copy_ready=False,kind='initial',timing=dict(worker_start_ns=start))
                    try:
                        item['wait_ready']();ready=time.perf_counter_ns();result['copy_ready']=True
                        x=item['array']();values=self.head.run(x).copy();predicted=time.perf_counter_ns()
                        result['feature_sha256']=[hashlib.sha256(np.ascontiguousarray(r,dtype='<f4').tobytes()).hexdigest() for r in x]
                        context_start=time.perf_counter_ns();context=item['context']()
                        result.update(values=values,context=context,publication_context_ns=time.perf_counter_ns()-context_start,
                            timing=dict(worker_start_ns=start,copy_ready_ns=ready,wait_ns=ready-start,predict_ns=predicted-ready,prediction_end_ns=predicted,
                                publish_ns=time.perf_counter_ns(),thread_cpu_ns=time.thread_time_ns()-cpu))
                        # Cache all data before ack permits model owner to reuse pinned buffer.
                        scores=self.core.features(item['keys'],x,values)
                        result['timing']['feature_prepare_and_feedback_ns']=time.perf_counter_ns()-predicted
                        result['timing']['publish_ns']=time.perf_counter_ns()
                        result['timing']['thread_cpu_ns']=time.thread_time_ns()-cpu
                        self.emit(result)
                        if scores:self.core.events.append(scores)
                    except BaseException as exc:
                        result.update(status='error',error=repr(exc));result['timing']['publish_ns']=time.perf_counter_ns();self.emit(result);raise
                elif kind=='completion':self.core.completion(item['key'],item['length'],item['ordinal'],item['observed_ns'])
                elif kind=='cancel':self.core.cancel(item['key'])
                else:raise ValueError('Unknown worker message')
                for row in self.core.events:self.emit(dict(row,kind='adaptation'))
                self.core.events.clear()
                self.emit(dict(kind='adaptation',type='ttt_worker_job',job=kind,enqueued_ns=item['enqueued_ns'],worker_start_ns=start,
                    worker_end_ns=time.perf_counter_ns(),queue_wait_ns=start-item['enqueued_ns'],thread_cpu_ns=time.thread_time_ns()-cpu))
                self.counters['processed']+=1;self.inputs.task_done()
            if self.core.feedback:raise RuntimeError('Shutdown has actual completions without feature; never drop labels')
        except BaseException as exc:self.fatal=repr(exc)
        finally:self.head.close()
    def poll(self):
        try:return self.results.get_nowait()
        except queue.Empty:
            if self.fatal:raise RuntimeError(self.fatal)
            return None
    def snapshot(self):
        return dict(self.counters,queue_size=self.inputs.qsize(),result_size=self.results.qsize(),updates=self.core.snapshot.version,
            feedback_received=self.core.received,feedback_applied=self.core.completed,pending_feedback=len(self.core.feedback),cached_features=len(self.core.entries),fatal=self.fatal)
    def close(self):
        if self.closed:return
        self.closed=True
        self.inputs.put(None,timeout=120);self.thread.join(120)
        if self.thread.is_alive():raise TimeoutError('TTT worker retains its feature/head; unsafe to free')
        if self.fatal:raise RuntimeError(self.fatal)
