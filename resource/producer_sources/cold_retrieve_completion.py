"""Start or finalize an origin-only LFS retrieval in one initially empty store."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROJECT='/mnt/SAM1TB/dev/projects/oscillation-trans-affine-reference-20261022'
DATA='/mnt/SAM1TB/dev/data/oscillation-trans-affine-reference-20261022'
ORIGIN='PRIVATE_ORIGIN_REDACTED'
BRANCH='research/capped-logit-experiments-20261022'


def main(args):
    assert len(args.commit)==40 and set(args.commit)<=set('0123456789abcdef')
    source=Path(__file__).read_bytes();revision=subprocess.check_output(['rtk','proxy','git','rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['rtk','proxy','git','show',revision+':'+str(Path(__file__).relative_to(ROOT))])==source
    spec=dict(mode=args.mode,commit=args.commit,project=PROJECT,data=DATA,origin=ORIGIN,branch=BRANCH,
        source_sha256=hashlib.sha256(source).hexdigest(),prepared_commit=revision)
    program='''import os,subprocess,json,time,hashlib,shutil
from pathlib import Path
spec=json.loads(SPEC)
base=Path(spec['project'])/'final_origin_retrieval_v1';root=base/'checkout'
record=Path(spec['data'])/('cold_origin_'+spec['mode']+'_attempt001');record.mkdir(exist_ok=False)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));os.nice(10)
env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1',GIT_TERMINAL_PROMPT='0')
prefix=['rtk','proxy'] if shutil.which('rtk') else []
def run(command):
 index=len(list(record.glob('step*.json')))
 r=subprocess.run(prefix+command,env=env,text=True,capture_output=True)
 (record/('step'+str(index)+'.json')).write_text(json.dumps(dict(command=prefix+command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+chr(10))
 print(json.dumps(dict(step=index,exit_code=r.returncode,command=command[:3])),flush=True)
 assert r.returncode==0,'Preserve failed retrieval;no fallback/cache reuse'
 return r.stdout.strip()
started=time.time()
if spec['mode']=='start':
 assert not base.exists();base.mkdir()
 (base/'initial_empty_store_receipt.json').write_text(json.dumps(dict(started=started,origin=spec['origin'],source_sha256=spec['source_sha256'],no_existing_checkout_or_LFS_store=True),indent=2)+chr(10))
 run(['git','clone','--no-checkout','--single-branch','--branch',spec['branch'],spec['origin'],str(root)])
 assert not (root/'.git/cold-lfs').exists()
 run(['git','-C',str(root),'config','--local','lfs.storage',str(root/'.git/cold-lfs')])
 run(['git','-C',str(root),'lfs','install','--local','--skip-smudge'])
else:
 initial=json.loads((base/'initial_empty_store_receipt.json').read_text());assert initial['no_existing_checkout_or_LFS_store']
 assert run(['git','-C',str(root),'config','--local','--get','lfs.storage'])==str(root/'.git/cold-lfs')
 assert run(['git','-C',str(root),'remote','get-url','origin'])==spec['origin']
 run(['git','-C',str(root),'fetch','origin',spec['branch']])
run(['git','-C',str(root),'checkout','--detach',spec['commit']])
assert run(['git','-C',str(root),'rev-parse','HEAD'])==spec['commit']
run(['git','-C',str(root),'lfs','fetch','--include=results/capped_logits_completion_2026-10-22/**','--exclude=','origin',spec['commit']])
receipt=dict(status='FETCH_COMPLETE_NOT_YET_AUDITED',**spec,root=str(root),isolated_store=str(root/'.git/cold-lfs'),started=started,completed=time.time(),affinity=sorted(os.sched_getaffinity(0)),all_payload_sources='origin only;no original worktree/LFS store copied',final_assets_covered=spec['mode']=='finalize')
(record/'summary.json').write_text(json.dumps(receipt,indent=2)+chr(10));print(json.dumps(receipt),flush=True)
'''.replace('SPEC',repr(json.dumps(spec)))
    subprocess.run(['rtk','proxy','ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gbminipc','python3 -'],input=program,text=True,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('start','finalize'),required=True)
    p.add_argument('--commit',required=True);main(p.parse_args())
