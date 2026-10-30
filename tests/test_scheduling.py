import unittest
import numpy as np
from leap.ledger import Ledger
from leap.scheduling import RequestState, select_published
from leap.features import last_prompt_features

class SchedulingTests(unittest.TestCase):
    def row(self,rid,slot,total,accepted=10,version=0):
        return dict(request_id=rid,request_slot=slot,generation=0,phase='initial' if version==0 else 'online',source_age=0,value=total,published_ns=accepted-1,accepted_ns=accepted,model_version=version)

    def test_no_future_consumption_and_fifo_fallback(self):
        ledger=Ledger();ledger.publish(self.row('a',0,100));ledger.publish(self.row('b',1,20,accepted=30))
        states=[RequestState('a',0,0,5,2048,1),RequestState('b',1,0,2,2048,2)]
        result,proof=select_published(states,ledger,now_ns=20,decode_capacity=1)
        self.assertTrue(proof['fifo_fallback']);self.assertEqual(result[0].request_id,'a')
        result,proof=select_published(states,ledger,now_ns=40,decode_capacity=1)
        self.assertFalse(proof['fifo_fallback']);self.assertEqual(result[0].request_id,'b')

    def test_countdown_once_and_new_version(self):
        ledger=Ledger();ledger.publish(self.row('a',0,100));s=RequestState('a',0,0,20,2048,1)
        self.assertEqual(ledger.snapshot(s,'ttt_rls',15)['score'],80)
        ledger.publish(self.row('a',0,60,accepted=20,version=1))
        self.assertEqual(ledger.snapshot(s,'ttt_rls',30)['score'],40)
        self.assertEqual(ledger.snapshot(s,'ttt_rls',15)['score'],80)

    def test_stale_or_regressing_publication_rejected(self):
        ledger=Ledger();ledger.publish(self.row('a',0,100))
        with self.assertRaises(ValueError):ledger.publish(self.row('b',0,20,accepted=20,version=1))
        with self.assertRaises(ValueError):ledger.publish(self.row('a',0,20,accepted=20))

    def test_last_prompt_features_both_padding_directions(self):
        x=np.arange(2*4*3).reshape(2,4,3)
        out=last_prompt_features(x,[[0,0,1,1],[1,1,0,0]])
        np.testing.assert_array_equal(out,np.stack([x[0,3],x[1,1]]))
        with self.assertRaises(ValueError):last_prompt_features(x,[[0,0,0,0],[1,1,0,0]])

if __name__=='__main__':unittest.main()

