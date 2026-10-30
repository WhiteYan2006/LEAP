"""RLS core equals a direct refit on prior + all applied completions; contract gates."""
import math,unittest
import numpy as np
from leap.rls_worker import RLSFeedbackCore
from leap._feedback import inverse
def problem(d=12,n0=40,alpha=.5,seed=0):
    g=np.random.default_rng(seed);Xs=g.normal(size=(n0,d))*3+1;mean=Xs.mean(0);scale=Xs.std(0,ddof=1)+1e-6
    U=np.c_[(Xs-mean)/scale,np.ones(n0)];A=U.T@U;A[np.arange(d),np.arange(d)]+=alpha*d
    coef=np.linalg.solve(A,U.T@(g.uniform(3,30,n0)));w=coef[:-1]/scale;b=coef[-1]-mean@w;coef=np.r_[w*scale,b+mean@w]
    return g,d,mean,scale,w,b,A,dict(P=np.linalg.inv(A).astype(np.float32),coef=coef,mean=mean,scale=scale)
class Tests(unittest.TestCase):
    def test_equals_refit_and_completion_order(self):
        g,d,mean,scale,w,b,A,prior=problem();core=RLSFeedbackCore(w,b,mean,scale,256,prior)
        X=g.normal(size=(40,d))*3+1.5;L=g.integers(1,2049,40);keys=[(i,0) for i in range(40)]
        core.features(keys,X,inverse(X@w+b))
        for i in range(40):core.completion(keys[i],int(L[i]),i+1,i)
        self.assertEqual(core.snapshot.version,5)
        U=np.c_[(X-mean)/scale,np.ones(40)];ref=np.linalg.solve(A+U.T@U,A@prior['coef']+U.T@np.sqrt(L))
        self.assertLess(np.abs(core.coef-ref).max()/np.abs(ref).max(),1e-4)
        v=[e for e in core.events if e['type']=='ttt_version_published']
        self.assertEqual([(e['window_count'],e['batch_start'],e['batch_count']) for e in v],[(8*i,8*i-7,8) for i in range(1,6)])
    def test_initial_equals_deployed_head_and_rescore(self):
        g,d,mean,scale,w,b,A,prior=problem(seed=1);core=RLSFeedbackCore(w,b,mean,scale,256,prior)
        X=g.normal(size=(12,d));keys=[(i,1) for i in range(12)];core.features(keys,X,inverse(X@w+b))
        for i in range(8):core.completion(keys[i],100,i+1,i)
        s=[e for e in core.events if e['type']=='ttt_scores_published'][-1]
        self.assertEqual(sorted(tuple(r['key']) for r in s['rows']),keys[8:])
        u=np.c_[(X[8:]-mean)/scale,np.ones(4)]
        self.assertTrue(np.allclose([r['latest_total'] for r in s['rows']],inverse(u@core.coef)))
    def test_rejects_other_head_or_dtype(self):
        g,d,mean,scale,w,b,A,prior=problem()
        with self.assertRaises(ValueError):RLSFeedbackCore(w*1.01,b,mean,scale,8,prior)
        with self.assertRaises(ValueError):RLSFeedbackCore(w,b,mean,scale,8,dict(prior,P=prior['P'].astype(np.float64)))
        with self.assertRaises(ValueError):RLSFeedbackCore(w,b,mean,scale*2,8,prior)
    def test_threaded_worker_publishes_in_order_and_drains_on_close(self):
        import time
        from leap.rls_worker import RLSWorker
        g,d,mean,scale,w,b,A,prior=problem(seed=2)
        class Head:
            def run(self,x):return inverse(x@w+b)
            def close(self):pass
        wk=RLSWorker(Head(),w,b,mean,scale,prior,256);X=g.normal(size=(40,d));L=g.integers(1,2049,40)
        for i in range(5):wk.submit(i,lambda:None,lambda i=i:X[8*i:8*i+8],lambda:None,[(j,0) for j in range(8*i,8*i+8)])
        for j in range(40):wk.completion((j,0),int(L[j]),j+1,j)
        wk.close();rows=[]
        while True:
            r=wk.poll()
            if r is None:break
            rows.append(r)
        v=[r for r in rows if r.get('type')=='ttt_version_published']
        self.assertEqual([(r['version'],r['completed']) for r in v],[(i,8*i) for i in range(1,6)])
        self.assertTrue(all(r['last_feedback_observed_ns']<=r['fit_start_ns']<=r['fit_end_ns']<=r['published_ns'] for r in v))
        U=np.c_[(X-mean)/scale,np.ones(40)];ref=np.linalg.solve(A+U.T@U,A@prior['coef']+U.T@np.sqrt(L))
        self.assertLess(np.abs(wk.core.coef-ref).max()/np.abs(ref).max(),1e-4);self.assertIsNone(wk.fatal)
    def test_shadow_publishes_nothing_and_sparse_only_moved(self):
        g,d,mean,scale,w,b,A,prior=problem(seed=3);X=g.normal(size=(24,d));keys=[(i,0) for i in range(24)]
        sh=RLSFeedbackCore(w,b,mean,scale,256,prior,'shadow');sh.features(keys,X,inverse(X@w+b))
        for i in range(16):sh.completion(keys[i],100+i,i+1,i)
        self.assertEqual([e for e in sh.events if e['type']=='ttt_scores_published'],[])
        self.assertEqual(sh.snapshot.version,2)
        full=RLSFeedbackCore(w,b,mean,scale,256,prior,'full');sp=RLSFeedbackCore(w,b,mean,scale,256,prior,'sparse',0.05)
        for c in (full,sp):
            c.features(keys,X,inverse(X@w+b))
            for i in range(16):c.completion(keys[i],100+i,i+1,i)
        self.assertTrue(np.allclose(full.coef,sp.coef) and np.allclose(full.coef,sh.coef))
        f=[e for e in full.events if e['type']=='ttt_scores_published'];s=[e for e in sp.events if e['type']=='ttt_scores_published']
        self.assertLessEqual(sum(len(e['rows']) for e in s),sum(len(e['rows']) for e in f))
if __name__=='__main__':unittest.main()
