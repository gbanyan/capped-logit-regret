"""Audit a separate origin-fetched checkout,manifests,LFS bytes and serialization."""
import argparse
import hashlib
import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
import torch

REL=Path('results/capped_logits_completion_2026-10-22')


def git(root,*args,**kwargs):
    prefix=['rtk','proxy'] if shutil.which('rtk') else []
    return subprocess.check_output([*prefix,'git','-C',str(root),*args],**kwargs)


def main(args):
    assert not args.output.exists(),'Never overwrite a restoration attempt'
    root=args.root.resolve();revision=git(root,'rev-parse','HEAD',text=True).strip()
    assert revision==args.expected_commit
    storage=Path(git(root,'config','--local','--get','lfs.storage',text=True).strip()).resolve()
    assert storage.is_relative_to(root/'.git'),'Require isolated local LFS storage,not the original object store'
    paths=[Path(p) for p in git(root,'ls-tree','-r','--name-only','HEAD','--',str(REL),text=True).splitlines()]
    assert paths;hashes={}
    def sha(path):
        if path not in hashes:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            hashes[path]=h.hexdigest()
        return hashes[path]
    binaries=[p for p in paths if p.suffix in ('.pt','.npz')]
    raw=git(root,'cat-file','--batch',input=''.join('HEAD:'+str(p)+'\n' for p in binaries).encode())
    position=0;checked=[];payloads={};torch.set_num_threads(2)
    for rel in binaries:
        end=raw.index(b'\n',position);header=raw[position:end].split();assert header[1]==b'blob'
        size=int(header[2]);pointer=raw[end+1:end+1+size];position=end+size+2
        assert pointer.startswith(b'version https://git-lfs.github.com/spec/v1\n')
        fields=dict(line.decode().split(' ',1) for line in pointer.splitlines()[1:])
        worktree=root/rel;oid=fields['oid'].removeprefix('sha256:')
        local=storage/'objects'/oid[:2]/oid[2:4]/oid
        assert local.resolve().is_relative_to(storage)
        assert local.is_file() and sha(local)==oid and local.stat().st_size==int(fields['size']),'Missing/corrupt independent cold-fetched object'
        materialized=worktree.stat().st_size==int(fields['size']) and sha(worktree)==oid
        if not materialized:assert worktree.read_bytes()==pointer,'Worktree must contain exact Git pointer or exact payload'
        # Fetching and parsing isolated LFS payloads avoids a second full binary
        # copy on disk. A skipped smudge is recorded,never called model inference.
        path=local;payloads[rel]=path
        row=dict(path=str(rel),sha256=oid,bytes=path.stat().st_size,worktree_materialized=materialized)
        if rel.suffix=='.pt':
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert isinstance(state,dict)
            tensors=[v for v in state.values() if isinstance(v,torch.Tensor)]
            assert tensors
            row.update(checkpoint_safely_loaded=True,tensor_count=len(tensors),
                all_tensors_finite=all(bool(torch.isfinite(v).all()) for v in tensors))
            assert row['all_tensors_finite'],'Nonfinite checkpoint tensors:'+str(rel)
            del state,tensors
        else:
            with np.load(path,allow_pickle=False) as arrays:
                row['arrays']=[dict(name=k,shape=list(arrays[k].shape),dtype=str(arrays[k].dtype)) for k in arrays.files]
            row['arrays_safely_loaded']=True
        checked.append(row)
    manifests=[]
    for rel in paths:
        if rel.name!='manifest.json':continue
        folder=(root/rel).parent;m=json.loads((root/rel).read_text());assert 'files' in m
        for name,pin in m['files'].items():
            logical=(folder/name).resolve();assert logical.is_relative_to(folder.resolve())
            path=payloads.get(logical.relative_to(root),logical)
            assert path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],str(logical)
        manifests.append(dict(path=str(rel),files=len(m['files']),manifest_sha256=sha(root/rel)))
    git(root,'diff','--exit-code','HEAD','--',str(REL))
    result=dict(status='PASS',commit=revision,root=str(root),isolated_lfs_storage=str(storage),
        manifests=manifests,checked_binaries=checked,binary_paths=len(checked),
        unique_LFS_objects=len({r['sha256'] for r in checked}),bytes=sum(r['bytes'] for r in checked),
        materialized_binary_paths=sum(r['worktree_materialized'] for r in checked),
        torch=torch.__version__,numpy=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual isolated remote retrieval/manifest/OID/serialization audit;not new inference or full R1-R7 completion')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',commit=revision,manifests=len(manifests),binaries=len(checked),
        unique_objects=result['unique_LFS_objects'],bytes=result['bytes'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
