"""Optional real Transformers interface check on a tiny random local model."""
import importlib.util,json,tempfile,unittest
from pathlib import Path

AVAILABLE=all(importlib.util.find_spec(n) for n in ('torch','transformers','tokenizers'))

@unittest.skipUnless(AVAILABLE,'Optional torch/transformers feature dependencies are not installed')
class FeatureCollectionTests(unittest.TestCase):
    def test_local_prefill_only_extraction(self):
        import torch
        from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
        from tokenizers import Tokenizer,models,pre_tokenizers
        from leap.cli import main
        import numpy as np
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);model_dir=root/'tiny';model_dir.mkdir()
            tok=Tokenizer(models.WordLevel({'[PAD]':0,'[UNK]':1,'hello':2,'world':3},unk_token='[UNK]'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',pad_token='[PAD]')
            tokenizer.save_pretrained(model_dir)
            config=Qwen3Config(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=2,head_dim=8,pad_token_id=0)
            Qwen3ForCausalLM(config).save_pretrained(model_dir)
            prompts=root/'prompts.jsonl';prompts.write_text(json.dumps(dict(id='a',prompt='hello world'))+'\n'+json.dumps(dict(id='b',prompt='hello'))+'\n')
            out=root/'features.npz'
            main(['collect-features','--model',str(model_dir),'--revision','local-fixture','--prompts',str(prompts),'--output',str(out),'--layer','2','--batch','2','--dtype','float32'])
            with np.load(out,allow_pickle=False) as z:
                self.assertEqual(z['x'].shape,(2,16));self.assertTrue(np.isfinite(z['x']).all())
                self.assertFalse(json.loads(str(z['metadata_json']))['generation'])

if __name__=='__main__':unittest.main()
