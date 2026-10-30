import json, tempfile, unittest
from pathlib import Path
import numpy as np
from leap import LinearReadout, fit_readout, make_rls_prior
from leap._ridge_reference import fit_heads
from leap.cli import main, check_ids


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        g=np.random.default_rng(7);self.x=g.normal(size=(60,8));self.y=np.clip(12+self.x@np.arange(8)/10,1,40)**2

    def fit(self, **kw):
        return fit_readout(self.x[:40],self.y[:40],self.x[40:50],self.y[40:50],**kw)[0]

    def test_original_fit_parity_and_source_only_statistics(self):
        head=self.fit();ref,_,_,mean,scale=fit_heads(self.x,np.arange(40),np.arange(40,50),self.y[:40],self.y[40:50])
        ref=next(x for x in ref if x['mapping']=='sqrt')
        np.testing.assert_array_equal(head.w,ref['w'].astype(np.float32).astype(np.float64))
        self.assertEqual(head.b,float(np.float32(ref['b'])))
        np.testing.assert_array_equal(head.mean,mean);np.testing.assert_array_equal(head.scale,scale)
        other=fit_readout(self.x[:40],self.y[:40],self.x[40:50]+1e5,self.y[40:50])[0]
        np.testing.assert_array_equal(other.mean,head.mean);np.testing.assert_array_equal(other.scale,head.scale)

    def test_clip_before_inverse(self):
        h=LinearReadout(np.array([1.]),0.,np.zeros(1),np.ones(1))
        np.testing.assert_array_equal(h.predict([[-100],[0],[2],[100]]),[1,1,4,2048])

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'head.npz';head=self.fit();head.save(p);loaded=LinearReadout.load(p)
            np.testing.assert_array_equal(loaded.predict(self.x),head.predict(self.x))
            self.assertEqual(loaded.metadata,head.metadata)
            with self.assertRaises(FileExistsError):head.save(p)

    def test_rls_prior_has_unpenalized_intercept_and_deployed_mean(self):
        h=self.fit();prior=make_rls_prior(h,self.x[:40]);u=np.c_[(self.x[:40]-h.mean)/h.scale,np.ones(40)]
        a=u.T@u;a[np.arange(8),np.arange(8)]+=h.metadata['regularization_lambda']
        np.testing.assert_allclose(prior['P'],np.linalg.inv(a),rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(u@prior['coef'],h.coordinates(self.x[:40]),atol=1e-12)
        self.assertEqual(prior['P'].dtype,np.float32)
        with self.assertRaises(ValueError):make_rls_prior(h,self.x[:40]+1)

    def test_rejects_nonfinite_and_invalid_inputs(self):
        with self.assertRaises(ValueError):self.fit(alphas=[0])
        with self.assertRaises(ValueError):self.fit(mapping='unknown')
        with self.assertRaises(ValueError):self.fit().predict(np.zeros((2,9)))
        with self.assertRaises(ValueError):self.fit().predict(np.full((2,8),np.nan))
        with self.assertRaises(ValueError):fit_readout(self.x[:40],self.y[:40]*0,self.x[40:50],self.y[40:50])

    def test_split_id_leakage_rejected(self):
        with self.assertRaises(ValueError):check_ids(dict(train_ids=['a'],val_ids=['a'],train_x=np.zeros((1,2)),val_x=np.zeros((1,2))))

    def test_documented_cli_cached_feature_workflow(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);p=t/'data.npz';out=t/'trained';result=t/'metrics.json'
            np.savez(p,train_x=self.x[:40],train_y=self.y[:40],val_x=self.x[40:50],val_y=self.y[40:50],test_x=self.x[50:],test_y=self.y[50:],train_ids=np.arange(40),val_ids=np.arange(40,50),test_ids=np.arange(50,60))
            main(['fit','--data',str(p),'--output',str(out),'--feature-id','synthetic','--model-revision','synthetic','--rls-prior'])
            main(['evaluate','--head',str(out/'head.npz'),'--data',str(p),'--output',str(result)])
            self.assertEqual(json.loads(result.read_text())['metrics']['n'],10)
            self.assertFalse(json.loads(result.read_text())['refit_or_selection'])

if __name__=='__main__':unittest.main()

