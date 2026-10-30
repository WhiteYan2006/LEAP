"""CPU-only mathematical demonstration; these are not paper measurements."""
import json, time
import numpy as np
from leap import fit_readout, make_rls_prior, RLSFeedbackCore
from leap._feedback import inverse

def main():
    g=np.random.default_rng(0);d=16;x=g.normal(size=(200,d));w=g.normal(size=d)*.2
    y=np.clip(x@w+15,1,40)**2
    head,_,_=fit_readout(x[:120],y[:120],x[120:160],y[120:160],feature_id='synthetic')
    prior=make_rls_prior(head,x[:120]);core=RLSFeedbackCore(head.w,head.b,head.mean,head.scale,128,prior)
    feedback=g.normal(size=(32,d));lengths=np.rint(np.clip(feedback@w+18,1,40)**2).astype(int)
    keys=[(i,0) for i in range(32)];core.features(keys,feedback,head.predict(feedback))
    for i in range(32):core.completion(keys[i],int(lengths[i]),i+1,time.perf_counter_ns())
    u=np.c_[(feedback-head.mean)/head.scale,np.ones(32)]
    source_u=np.c_[(x[:120]-head.mean)/head.scale,np.ones(120)]
    a=source_u.T@source_u;a[np.arange(d),np.arange(d)]+=head.metadata['regularization_lambda']
    batch=np.linalg.solve(a+u.T@u,a@prior['coef']+u.T@np.sqrt(lengths))
    np.testing.assert_allclose(core.coef,batch,rtol=1e-4,atol=1e-5)
    print(json.dumps(dict(synthetic=True,paper_result=False,source_training_examples=120,
                         actual_completion_labels=32,RLS_versions=core.snapshot.version,
                         max_coefficient_error_vs_direct_solve=float(np.max(np.abs(core.coef-batch))))))
if __name__=='__main__':main()

