"""Strict adverse sequences, actual score PGD and explicit ReLU witness controls."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import oracles as O


def main(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    rng=np.random.default_rng(222501);sequences=[];boundaries=[];arrays={};witnesses=[]
    for n in (4,7,10,25,40,100,1000):
        for ratio in (1.01,3.,30.,300.):
            beta=(n-2)/(2*ratio+n-2);cap=np.log(ratio)/2
            slope=1+beta+2/(ratio-1)
            for fraction in (.5,.9,.999):
                gamma=fraction*beta;tail=(beta-gamma)/2
                p=np.full((1,n),tail/(n-2));perm=rng.permutation(n)
                b,c=perm[:2];p[0,b]=(1-tail+gamma)/2;p[0,c]=(1-tail-gamma)/2
                oracle=O.box(p,ratio)
                assert abs(oracle['z'][0,b]-cap)<1e-10 and abs(oracle['z'][0,c]-cap)<1e-10
                oo=O.ball(p,np.sqrt(2)*cap)
                assert oo['q'][0].argmax()==b
                domain_gap=oo['q'][0,b]-np.delete(oo['q'][0],b).max()
                assert domain_gap>0
                for k in range(8):
                    delta=min(.1,cap/10)*10.**(-k)
                    z=oracle['z'].copy();z[0,b]-=delta
                    D=float((O.loss(p,z)-oracle['CE'])[0]);regret=float(p[0,b]-p[0,z.argmax(1)[0]])
                    assert z.argmax(1)[0]==c and D>0 and abs(regret-gamma)<1e-12
                    assert regret<=beta+slope*np.sqrt(2*(D+oracle['gap'][0]+1e-12))
                    sequences.append(dict(n=n,R=ratio,beta=beta,gamma=gamma,tau=tail,
                        delta=delta,D=D,regret=regret,ball_optimal_gap=float(domain_gap)))
            for name,p in [('uniform',np.ones((1,n))/n),('deterministic',np.eye(n)[:1])]:
                box=O.box(p,ratio);ball=O.ball(p,np.sqrt(2)*cap)
                assert name=='uniform' or (box['q'][0].argmax()==0 and ball['q'][0].argmax()==0)
                boundaries.append(dict(n=n,R=ratio,posterior=name,box_CE=float(box['CE'][0]),
                    ball_CE=float(ball['CE'][0]),box_FW=float(box['gap'][0]),ball_FW=float(ball['gap'][0])))
    for ratio in (1.01,3.,30.,300.):
        p=np.array([[.7,.3]]);cap=np.log(ratio)/2
        for domain,oracle in [('box',O.box(p,ratio)),('ball',O.ball(p,np.sqrt(2)*cap))]:
            assert oracle['q'][0].argmax()==0
            boundaries.append(dict(n=2,R=ratio,posterior='binary',domain=domain,beta=0.,CE=float(oracle['CE'][0])))
    n=10;beta=.3;ratio=(n-2)*(1-beta)/(2*beta);cap=np.log(ratio)/2;radius=np.sqrt(2)*cap
    p=np.full((1,n),.06/(n-2));p[0,0]=.45;p[0,1]=.49
    raw=np.full((1,n),-4*cap);raw[0,0]=4*cap
    hard=raw.copy();soft=raw.copy();pgd=np.clip(raw,-cap,cap);ball=O.project_ball(raw,radius)
    initial_hard=np.clip(hard,-cap,cap);stress=[]
    for step in range(20001):
        hz=np.clip(hard,-cap,cap);sz=cap*np.tanh(soft/cap)
        if step in (0,1,10,100,1000,20000):
            for name,z,domain,budget in [('hard_jacobian',hz,'box',cap),('tanh_jacobian',sz,'box',cap),('score_PGD',pgd,'box',cap),('ball_PGD',ball,'ball',radius)]:
                stress.append(dict(step=step,arm=name,CE=float(O.loss(p,z)[0]),
                    FW=float(O.fw_gap(p,z,domain,budget)[0]),pred=int(z.argmax(1)[0])))
        if step==20000:break
        hard-=2*(np.exp(O.logs(hz))-p)*(abs(hard)<cap)
        soft-=2*(np.exp(O.logs(sz))-p)*(1-np.tanh(soft/cap)**2)
        pgd=np.clip(pgd-2*(np.exp(O.logs(pgd))-p),-cap,cap)
        ball=O.project_ball(ball-2*(np.exp(O.logs(ball))-p),radius)
    assert np.array_equal(np.clip(hard,-cap,cap),initial_hard)
    assert O.fw_gap(p,pgd,'box',cap)[0]<=1e-9
    assert O.fw_gap(p,ball,'ball',radius)[0]<=1e-9
    for n in (4,10,40,100):
        r=np.random.default_rng(222501+n);c=r.integers(n,size=10000);u=r.uniform(size=10000)
        v=np.eye(n)[c]+r.uniform(-.05,.05,(len(c),n))
        h=10*(np.maximum(0,v-.5)-np.maximum(0,v-.6))
        assert np.max(abs(h-np.eye(n)[c]))<1e-12
        for ratio in (3.,30.):
            cap=np.log(ratio)/2;beta=(n-2)/(2*ratio+n-2)
            z=-cap+2*cap*(h+np.roll(h,1,axis=1));z=np.clip(z,-cap,cap)
            low=1/(2*ratio+n-2);high=ratio*low;s=.95*beta;gamma=.9*beta*(2*u-1)
            p=np.full((len(c),n),low-s/(n-2));p[np.arange(len(c)),c]=high+(s+gamma)/2
            p[np.arange(len(c)),(c+1)%n]=high+(s-gamma)/2
            oracle=O.box(p,ratio);difference=O.loss(p,z)-oracle['CE']
            assert p.min()>0 and abs(difference).max()<1e-10
            assert O.fw_gap(p,z,'box',cap).max()<1e-10
            witnesses.append(dict(n=n,R=ratio,points=len(c),hidden_units=2*n,
                max_oracle_CE_difference=float(abs(difference).max()),FW=float(O.fw_gap(p,z,'box',cap).max())))
    result=dict(status='PASS',strict_sequences=sequences,boundaries=boundaries,gradient_stress=stress,
        witnesses=witnesses,seconds=time.monotonic()-start,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),Path(O.__file__))},numpy=np.__version__)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',sequences=len(sequences),boundaries=len(boundaries),
        witness_inputs=sum(w['points'] for w in witnesses),seconds=result['seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:main(args.output)
    except Exception:
        if args.output.is_dir():(args.output/'failure.txt').write_text(traceback.format_exc())
        raise
