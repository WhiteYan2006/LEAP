"""Compile the original native head with the user's existing C++ compiler."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--compiler',default='g++');p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[1]/'native/readout4096.cpp';out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Native output exists')
    compiler=shutil.which(a.compiler)
    if not compiler:raise FileNotFoundError('C++ compiler not found: '+a.compiler)
    out.parent.mkdir(parents=True,exist_ok=True)
    flags=['-O2','-std=c++17','-shared']
    flags+=['-static','-static-libstdc++','-static-libgcc'] if os.name=='nt' else ['-fPIC']
    subprocess.run([compiler,*flags,str(source),'-o',str(out)],check=True)
    info=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),compiler=compiler,flags=flags,platform=sys.platform)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
if __name__=='__main__':main()

