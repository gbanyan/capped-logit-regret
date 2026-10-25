"""Reaudit identical attained points with strictly positive duals;zero optimizer steps."""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import affine_raw_reference as R


def positive_shrink(T,trial):
    residual=trial-T;residual-=residual.mean(1,keepdims=True)
    candidate=T+residual
    if np.all(candidate>0):return candidate,1.
    negative=residual<0;boundary=float(np.min(T[negative]/(-residual[negative])))
    factor=min(1.,.5*boundary);candidate=T+factor*residual
    assert np.isfinite(candidate).all() and candidate.min()>0
    return candidate,factor


def candidate_points(X,W,T,coefficients):
    z=X@coefficients;q=np.exp(z-z.max(1,keepdims=True));q*=(T.sum(1)/q.sum(1))[:,None]
    yield 'weighted_softmax',q.copy(),None
    Z=X@W;G=Z.T@Z/len(T)
    for iteration in (1,2):
        correction=Z@np.linalg.solve(G,Z.T@(q-T)/len(T))
        q,factor=positive_shrink(T,q-correction)
        yield 'positive_gram_projection_'+str(iteration),q.copy(),factor


def verify_chart(X,W):
    N,d=X.shape;Z=X@W;G=Z.T@Z/N;inverse=np.linalg.inv(W)
    for name,value in [('stored_X',X),('stored_W',W),('stored_Z',Z),('stored_G',G),('inverse',inverse)]:R.normal_finite(value,np.float64,name)
    left=R.norm_up(inverse@W-np.eye(d))+4*R.gamma(d,R.U64)*R.norm_up(np.abs(inverse)@np.abs(W))
    right=R.norm_up(W@inverse-np.eye(d))+4*R.gamma(d,R.U64)*R.norm_up(np.abs(W)@np.abs(inverse))
    eZ=4*R.gamma(d,R.U64)*R.norm_up(np.abs(X)@np.abs(W));zn=R.norm_up(Z)
    error=4*R.gamma(N+4,R.U64)*R.norm_up(np.abs(Z).T@np.abs(Z)/N)+(2*zn*eZ+eZ*eZ)/N
    error+=4*R.gamma(8*d*d+100,R.U64)*(1+R.norm_up(G))
    lam=R.L(1)-R.norm_up(G-np.eye(d))-error
    h=(np.sqrt(np.sum(Z.astype(R.L)**2,axis=1,dtype=R.L)).max()+eZ)*(1+4*R.gamma(N*d+100,R.U64))
    return dict(rank_and_coordinate_change_verified=bool(min(left,right)<1 and lam>R.L('.5')),
        gram_lambda_lower=R.lower(lam),feature_norm_upper=R.upper(h),
        left_inverse_defect_upper=R.upper(left),right_inverse_defect_upper=R.upper(right),
        gram_error_upper=R.upper(error),full_dimension=d)


def certify(X,T,coefficients,W):
    N,n=T.shape;d=X.shape[1];geometry=verify_chart(X,W)
    primal=R.primal_interval(X,T,coefficients);entropy,_,_=R.entropy_interval(T)
    zero=R.primal_interval(X,T,np.zeros_like(coefficients));radius=None
    if geometry['rank_and_coordinate_change_verified']:
        radius=R.upper(R.L(geometry['feature_norm_upper'])*R.L(min(zero['upper'],primal['upper']))/
            (R.L(T.min())*np.sqrt(R.L(n)/R.L(n-1))*R.L(geometry['gram_lambda_lower']))*(1+R.gamma(100)))
    XL=X.astype(R.L);WL=W.astype(R.L);TL=T.astype(R.L)
    Z=XL@WL;eZ=4*R.gamma(d+4)*(np.abs(XL)@np.abs(WL))
    low=entropy['lower'];rows=[];arrays=dict(coefficients=coefficients.copy(),preconditioner=W.copy())
    for name,Q,factor in candidate_points(X,W,T,coefficients):
        arrays[name]=Q;ent,q,e_q=R.entropy_interval(T,Q);residual=q-TL;eR=e_q+4*R.UL*(np.abs(q)+np.abs(TL))
        moment=(Z.T@residual)/N
        eM=4*R.gamma(N+4)*(np.abs(Z).T@np.abs(residual))/N
        eM+=4*(np.abs(Z).T@eR+eZ.T@(np.abs(residual)+eR))/N
        moment_upper=R.upper(R.norm_up(moment)+R.norm_up(eM))
        lower=R.lower(R.L(ent['lower'])-R.L(radius)*R.L(moment_upper)) if radius is not None else None
        if lower is not None:low=max(low,lower)
        rows.append(dict(name=name,shrink_factor=factor,entropy=ent,moment_estimated=float(R.norm_up(moment)),
            moment_error_upper=R.upper(R.norm_up(eM)),moment_upper=moment_upper,lower_bound=lower))
    assert low<=primal['upper'];width=R.upper(R.L(primal['upper'])-R.L(low))
    record=dict(status='REAUDITED_SAME_POINT',objective_scope='trainingB_unbounded_fixed_affine',geometry=geometry,
        primal=primal,target_weighted_entropy=entropy,optimum_radius_upper=radius,dual_candidates=rows,
        lower_bound=low,upper_bound=primal['upper'],interval_width=width,
        tight_interval=bool(geometry['rank_and_coordinate_change_verified'] and width<=1e-6),
        training_D_floor_lower=max(0.,R.lower(R.L(low)-R.L(entropy['upper']))),
        training_D_floor_upper=R.upper(R.L(primal['upper'])-R.L(entropy['lower'])),
        same_point_no_optimizer_updates=True,optimizer_steps=0,parameter_class_narrowed=False,
        reference_coefficient_max_abs=float(abs(coefficients).max()),
        reference_coefficient_exceeds_FP32_max=bool(abs(coefficients).max()>np.finfo(np.float32).max),
        numerical_validity='Conditional IEEE/libm model from original reference,not formal universal machine proof',
        official_test_decoded=False,longdouble_nmant=np.finfo(R.L).nmant)
    return record,arrays


def main(args):
    assert np.finfo(R.L).nmant>52;jobs=json.loads(args.queue.read_text())['jobs'];assert len(jobs)==3;failed=[]
    for job in jobs:
        architecture=job['architecture'];out=args.output/architecture;out.mkdir(exist_ok=False);started=time.monotonic()
        try:
            source=args.point_root/architecture;parent=json.loads((source/'complete.json').read_text())
            assert parent['source_sha256']==R.sha(Path(R.__file__)) and parent['job']==job
            assert R.sha(source/'reference.npz')==parent['arrays_sha256']
            data=args.input_root/architecture;assert R.sha(data/'training_inputs.npz')==job['input_sha256']
            assert R.sha(data/'complete.json')==job['complete_sha256']
            with np.load(data/'training_inputs.npz',allow_pickle=False) as a:
                X=np.column_stack([a['features'][:,a['kept_feature_columns']].astype(np.float64),np.ones(len(a['targets']))]);T=a['targets']
            with np.load(source/'reference.npz',allow_pickle=False) as a:coefficients=a['coefficients'].copy();W=a['preconditioner'].copy()
            record,arrays=certify(X,T,coefficients,W)
            assert np.array_equal(arrays['coefficients'],coefficients) and np.array_equal(arrays['preconditioner'],W)
            np.savez_compressed(out/'reference.npz',**arrays)
            record.update(job=job,parent_complete_sha256=R.sha(source/'complete.json'),parent_arrays_sha256=parent['arrays_sha256'],
                parent_solver=parent['solver'],arrays_sha256=R.sha(out/'reference.npz'),input_receipt=parent['input_receipt'],
                source_sha256=R.sha(Path(__file__)),reference_source_sha256=R.sha(Path(R.__file__)),seconds=time.monotonic()-started)
            R.write(out/'complete.json',record);print(json.dumps(dict(architecture=architecture,width=record['interval_width'],tight=record['tight_interval'])),flush=True)
        except Exception:failed.append(architecture);R.write(out/'failure.json',dict(job=job,traceback=traceback.format_exc(),optimality='UNKNOWN'))
    R.write(args.output/'queue_complete.json',dict(jobs=3,failed=failed,all_successful=not failed,official_test_decoded=False))
    if failed:raise RuntimeError('Preserved same-point dual-audit failures:'+str(failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--point-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
