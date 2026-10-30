from pathlib import Path
import json,subprocess,sys,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class DatasetPreparationTests(unittest.TestCase):
    def run_case(self, overlap=False):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);features=p/'features.npz';labels=p/'labels.jsonl';split=p/'split.json';out=p/'cached.npz'
            x=np.arange(24,dtype=np.float32).reshape(6,4);ids=list('abcdef');np.savez(features,x=x,ids=ids)
            rows=[dict(id=rid,length=i+10,parent_id='p'+str(i)) for i,rid in enumerate(ids)]
            if overlap:rows[2]['parent_id']=rows[0]['parent_id']
            labels.write_text(''.join(json.dumps(r)+'\n' for r in reversed(rows)))
            split.write_text(json.dumps(dict(train=['b','a'],val=['d','c'],test=['f','e'])))
            r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_cached_dataset.py'),
                '--features',str(features),'--labels',str(labels),'--split',str(split),
                '--label-key','length','--output',str(out),'--require-disjoint-parents'],capture_output=True)
            if overlap:
                self.assertNotEqual(r.returncode,0);self.assertIn('Parent-conversation overlap',r.stderr.decode());self.assertFalse(out.exists())
            else:
                self.assertEqual(r.returncode,0,r.stderr.decode())
                with np.load(out,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['train_x'],x[[1,0]])
                    np.testing.assert_array_equal(z['train_y'],[11,10]);self.assertEqual(z['test_ids'].tolist(),['f','e'])

    def test_identity_join_preserves_frozen_split_order(self):self.run_case()
    def test_parent_leakage_rejected(self):self.run_case(overlap=True)

if __name__=='__main__':unittest.main()
