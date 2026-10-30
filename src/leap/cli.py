"""Portable cached-feature fitting/evaluation; model inference is an explicit extra."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from .readout import LinearReadout, fit_readout, make_rls_prior, labels, matrix
from ._ridge_reference import metric


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_ids(z):
    sets = []
    for split in ('train', 'val', 'test'):
        key = split+'_ids'
        if key in z:
            ids = [str(x) for x in z[key]]
            # ID disjointness is checked across all splits, but training must not
            # load the test feature/label arrays even for a shape inspection.
            if split != 'test' and (split+'_x' not in z or len(ids) != len(z[split+'_x'])):
                raise ValueError('ID/feature count differs in '+split)
            if len(ids) != len(set(ids)): raise ValueError('Duplicate IDs in '+split)
            sets.append((split, set(ids)))
    for i, (left, a) in enumerate(sets):
        for right, b in sets[i+1:]:
            if a & b: raise ValueError('Split ID overlap: '+left+'/'+right)
    return len(sets) >= 2


def train(args):
    out = Path(args.output)
    if out.exists(): raise FileExistsError('Output exists; choose a fresh training directory')
    with np.load(args.data, allow_pickle=False) as z:
        ids_checked = check_ids(z)
        x, y, vx, vy = [z[k] for k in ('train_x', 'train_y', 'val_x', 'val_y')]
    head, candidates, costs = fit_readout(x, y, vx, vy, mapping=args.mapping,
        feature_id=args.feature_id, metadata=dict(input_sha256=digest(args.data),
        model_revision=args.model_revision, split_ids_checked=ids_checked))
    out.mkdir(parents=True)
    head.save(out/'head.npz')
    (out/'selection.json').write_text(json.dumps(dict(metadata=head.metadata,
        candidates=candidates, costs=costs, head_sha256=digest(out/'head.npz')), indent=2, allow_nan=False)+'\n')
    if args.rls_prior:
        prior = make_rls_prior(head, x)
        np.savez(out/'rls_prior.npz', **prior)
    print(json.dumps(dict(head=str(out/'head.npz'), selected_alpha=head.metadata['alpha'],
                         validation_mae=head.metadata['validation_mae'], test_used=False)))


def evaluate(args):
    head = LinearReadout.load(args.head)
    with np.load(args.data, allow_pickle=False) as z:
        x = matrix(z[args.split+'_x'], len(head.w))
        y = labels(z[args.split+'_y'], len(x), head.budget)
        if args.split+'_ids' in z and len(z[args.split+'_ids']) != len(x):
            raise ValueError('Evaluation ID/feature count differs')
    p = head.predict(x)
    result = dict(split=args.split, metrics=metric(y, p), data_sha256=digest(args.data),
                  head_sha256=digest(args.head), refit_or_selection=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


def collect(args):
    import torch, transformers
    from .features import select_decoder_layer, last_prompt_features
    rows = [json.loads(s) for s in Path(args.prompts).read_text(encoding='utf8').splitlines() if s.strip()]
    if not rows or len({str(r['id']) for r in rows}) != len(rows): raise ValueError('Nonempty, unique prompt IDs required')
    target = Path(args.output)
    if target.exists(): raise FileExistsError('Feature output exists')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                    local_files_only=not args.allow_download, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise ValueError('Tokenizer has no padding or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
               local_files_only=not args.allow_download, trust_remote_code=False, torch_dtype=dtype).to(args.device).eval()
    block = select_decoder_layer(model, args.layer)
    captured = []
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(h) or h.ndim != 3: raise ValueError('Unsupported decoder-output contract')
        captured.append(h.detach())
    handle = block.register_forward_hook(hook)
    features = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch):
                chunk = rows[start:start+args.batch]
                tokens = tokenizer([r['prompt'] for r in chunk], padding=True, truncation=False, return_tensors='pt')
                if tokens['input_ids'].shape[1] > args.max_prompt_tokens:
                    raise ValueError('Prompt exceeds explicit limit; no silent truncation')
                captured.clear()
                model(**{k:v.to(args.device) for k,v in tokens.items()}, use_cache=False)
                if len(captured) != 1: raise ValueError('Selected layer did not run exactly once')
                # Match the research feature export's FP16 storage, then FP32 readout.
                h = captured[0].to(torch.float16).float().cpu().numpy()
                features.append(last_prompt_features(h, tokens['attention_mask'].numpy()))
    finally: handle.remove()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez(f, x=np.concatenate(features), ids=np.asarray([str(r['id']) for r in rows]),
                 metadata_json=json.dumps(dict(model=args.model, revision=args.revision, one_based_layer=args.layer,
                    feature_storage='float16 then float32', generation=False,
                    torch=torch.__version__, transformers=transformers.__version__, prompts_sha256=digest(args.prompts))))
    print(json.dumps(dict(output=str(target), rows=len(rows), generation=False)))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='leap',
        description='Fit and evaluate linear output-length readouts from cached features.',
    )
    sub = p.add_subparsers(dest='command', required=True)

    t = sub.add_parser('fit', help='Fit on source train; select regularization on validation.')
    t.add_argument('--data', required=True, help='Cached-feature NPZ with train and validation splits.')
    t.add_argument('--output', required=True, help='New directory for the head and selection report.')
    t.add_argument('--mapping', choices=['sqrt', 'log', 'identity'], default='sqrt')
    t.add_argument('--feature-id', required=True, help='Identity of the feature extraction contract.')
    t.add_argument('--model-revision', required=True, help='Backbone revision used to collect features.')
    t.add_argument('--rls-prior', action='store_true', help='Also export the source-anchored RLS prior.')
    t.set_defaults(run=train)

    e = sub.add_parser('evaluate', help='Evaluate a saved head without refitting.')
    e.add_argument('--head', required=True, help='Saved head.npz checkpoint.')
    e.add_argument('--data', required=True, help='Cached-feature NPZ containing the evaluation split.')
    e.add_argument('--split', choices=['val', 'test'], default='test')
    e.add_argument('--output', help='New JSON file for metrics; also printed to stdout.')
    e.set_defaults(run=evaluate)

    c = sub.add_parser('collect-features', help='Collect last-prompt-token features with prefill only.')
    c.add_argument('--model', required=True, help='Local model directory, or model ID with --allow-download.')
    c.add_argument('--revision', required=True, help='Backbone revision identifier.')
    c.add_argument('--prompts', required=True, help='JSONL containing id and prompt fields.')
    c.add_argument('--output', required=True, help='New feature NPZ file.')
    c.add_argument('--layer', type=int, default=30, help='One-based decoder block (default: 30).')
    c.add_argument('--batch', type=int, default=1)
    c.add_argument('--max-prompt-tokens', type=int, default=4096)
    c.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    c.add_argument('--device', default='cpu')
    c.add_argument('--allow-download', action='store_true', help='Allow model/tokenizer network access.')
    c.set_defaults(run=collect)

    args = p.parse_args(argv)
    if args.command == 'collect-features' and (args.batch < 1 or args.max_prompt_tokens < 1):
        p.error('Positive batch/token limits required')
    args.run(args)

if __name__ == '__main__':
    main()

