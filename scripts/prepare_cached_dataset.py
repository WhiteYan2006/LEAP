"""Join independently collected features/labels using an explicit frozen split."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--split',required=True);p.add_argument('--label-key',required=True);p.add_argument('--output',required=True)
    p.add_argument('--require-disjoint-parents',action='store_true');a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('Dataset output exists')
    with np.load(a.features,allow_pickle=False) as z:x=z['x'];ids=[str(v) for v in z['ids']]
    if x.ndim!=2 or len(x)!=len(ids) or len(set(ids))!=len(ids) or not np.isfinite(x).all():raise ValueError('Invalid feature/ID matrix')
    index={v:i for i,v in enumerate(ids)};rows=[json.loads(s) for s in Path(a.labels).read_text(encoding='utf8').splitlines() if s.strip()]
    labels={str(r['id']):r for r in rows}
    if len(labels)!=len(rows):raise ValueError('Duplicate label IDs')
    split=json.loads(Path(a.split).read_bytes());seen=set();parents={};arrays={}
    for name in ('train','val','test'):
        selected=[str(v) for v in split[name]]
        if not selected or len(set(selected))!=len(selected) or seen.intersection(selected):raise ValueError('Empty, duplicate or overlapping split')
        seen.update(selected)
        if any(i not in index or i not in labels for i in selected):raise ValueError('Split ID lacks features or labels')
        target=np.asarray([labels[i][a.label_key] for i in selected],dtype=np.float64)
        if not np.isfinite(target).all() or (target<1).any() or (target>2048).any():raise ValueError('Invalid length labels')
        groups={str(labels[i]['parent_id']) for i in selected if 'parent_id' in labels[i]}
        if a.require_disjoint_parents and any('parent_id' not in labels[i] for i in selected):raise ValueError('Parent IDs required for leakage check')
        if any(groups.intersection(old) for old in parents.values()):raise ValueError('Parent-conversation overlap between splits')
        parents[name]=groups;arrays.update({name+'_x':x[[index[i] for i in selected]],name+'_y':target,name+'_ids':np.asarray(selected)})
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    arrays['metadata_json']=json.dumps(dict(features_sha256=sha(a.features),labels_sha256=sha(a.labels),split_sha256=sha(a.split),label_key=a.label_key,parent_ids_required=a.require_disjoint_parents))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:np.savez(f,**arrays)
    print(json.dumps(dict(output=str(output),split_counts={k:len(split[k]) for k in ('train','val','test')},no_new_generation=True)))
if __name__=='__main__':main()

