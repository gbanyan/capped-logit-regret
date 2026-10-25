"""Unbounded TRAIN-B raw affine reference;conditional arithmetic intervals,not formal proof."""
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

L=np.longdouble
UL=L(np.finfo(L).eps)/2
U64=L(2.)**-53
LIBM=8


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def gamma(k,u=UL):
    value=L(k)*u;assert 0<=value<L('.01')
    return value/(1-value)


def norm_up(value):
    value=np.asarray(value,dtype=L)
    return np.sqrt(np.sum(value*value,dtype=L))/(1-gamma(4*value.size+16))


def normal_finite(value,dtype,name):
    value=np.asarray(value,dtype=dtype);absolute=np.abs(value)
    assert np.isfinite(value).all(),name+':nonfinite'
    assert not np.any((absolute>0)&(absolute<np.finfo(dtype).tiny)),name+':subnormal'


def upper(value):return float(np.nextafter(float(value),np.inf))


def lower(value):return float(np.nextafter(float(value),-np.inf))


def mp_exact(value):
    mantissa,exponent=np.frexp(L(value));bits=np.finfo(L).nmant+1
    return mp.mpf(int(np.ldexp(mantissa,bits)))*mp.power(2,int(exponent)-bits)


def log_spots(values):
    flat=np.asarray(values,dtype=L).ravel();assert np.all(flat>0)
    ids=np.unique(np.r_[0,len(flat)-1,np.linspace(0,len(flat)-1,17,dtype=np.int64),flat.argmin(),flat.argmax()])
    largest=0.
    with mp.workdps(90):
        for index in ids:
            exact=mp.log(mp_exact(flat[index]));observed=mp_exact(np.log(flat[index]))
            allowed=mp.mpf(LIBM)*mp_exact(UL)*(1+abs(exact))
            ratio=float(abs(observed-exact)/allowed);largest=max(largest,ratio)
            assert ratio<=1,'Long-double log spot exceeds declared model'
    return dict(points=len(ids),largest_fraction_of_assumed_allowance=largest,scope='High-precision spot check,not proof of the libm bound at all arguments')


def exp_spots(values):
    flat=np.asarray(values,dtype=L).ravel();assert np.isfinite(flat).all() and np.all(flat<=0)
    ids=np.unique(np.r_[0,len(flat)-1,np.linspace(0,len(flat)-1,17,dtype=np.int64),flat.argmin(),flat.argmax()])
    largest=0.
    with mp.workdps(90):
        for index in ids:
            exact=mp.exp(mp_exact(flat[index]));observed=mp_exact(np.exp(flat[index]))
            ratio=float(abs(observed-exact)/(mp.mpf(LIBM)*mp_exact(UL)*exact));largest=max(largest,ratio)
            assert ratio<=1,'Long-double exp spot exceeds declared model'
    return dict(points=len(ids),largest_fraction_of_assumed_allowance=largest,scope='High-precision spot check,not proof of the libm bound at all arguments')


def primal_interval(X,T,coefficients):
    X=np.asarray(X,dtype=L);T=np.asarray(T,dtype=L);b=np.asarray(coefficients,dtype=L)
    N,n=T.shape;d=X.shape[1];s=T.sum(1,dtype=L);z=X@b
    assert np.isfinite(z).all()
    shifted=z-z.max(1,keepdims=True)
    assert shifted.min()>np.log(L(np.finfo(L).tiny))+100,'Exponential underflow model unresolved'
    exponential=np.exp(shifted);sums=exponential.sum(1,dtype=L)
    normal_finite(exponential,L,'stable_exp');normal_finite(sums,L,'exp_sums')
    logs=np.log(sums)+z.max(1)
    row=s*logs-(T*z).sum(1,dtype=L);value=row.mean(dtype=L)
    dot_error=4*gamma(d)*(np.abs(X)@np.abs(b))
    forward_error=(2*s*dot_error.max(1)).mean(dtype=L)
    # Stable LSE has exp arguments<=0 and positive exponential sum>=1.
    # This coarse operation-count envelope includes subtraction,positive
    # reductions,products,mass sums and the conditional libm error model.
    operations=32*N*n+32*N+100
    scale=(s*(1+4*np.abs(z).max(1)+2*L(math.log(n)))+(T*np.abs(z)).sum(1,dtype=L)).mean(dtype=L)
    arithmetic=4*(gamma(operations)+LIBM*UL)*scale
    allowance=forward_error+arithmetic
    return dict(value=float(value),lower=lower(value-allowance),upper=upper(value+allowance),
        forward_allowance=upper(forward_error),arithmetic_allowance=upper(arithmetic),operations=operations,
        maximum_score=float(np.abs(z).max()),spot_checks=dict(exp=exp_spots(shifted),log_sums=log_spots(sums),log_mass=log_spots(s)),
        exp_and_log_model='IEEE round-to-nearest;log absolute error<=8u*(1+abs(log(x))),exp relative error<=8u;normal finite intermediates')


def entropy_interval(T,Q=None):
    T=np.asarray(T,dtype=L);N,n=T.shape;s=T.sum(1,dtype=L)
    assert np.isfinite(T).all() and T.min()>0
    if Q is None:
        q=T;e_q=np.zeros_like(T)
    else:
        raw=np.asarray(Q,dtype=L);assert raw.shape==T.shape and np.isfinite(raw).all() and raw.min()>0
        c=raw.sum(1,dtype=L);q=raw*(s/c)[:,None]
        # Q* is the EXACT real row rescaling s_i Q_i/sum Q_i.
        # All sums are positive;16gamma(n+4) covers both sums,division/product.
        e_q=16*gamma(n+4)*q
    e_s=4*gamma(n+1)*s
    assert np.all(q-e_q>0) and np.all(s-e_s>0)
    qlog=np.log(q);slog=np.log(s)
    value=(-np.sum(q*qlog,dtype=L)+np.sum(s*slog,dtype=L))/N
    lipschitz_q=1+np.maximum(np.abs(np.log(q-e_q)),np.abs(np.log(q+e_q)))
    lipschitz_s=1+np.maximum(np.abs(np.log(s-e_s)),np.abs(np.log(s+e_s)))
    mass_error=(np.sum(e_q*lipschitz_q,dtype=L)+np.sum(e_s*lipschitz_s,dtype=L))/N
    operations=32*N*n+32*N+100
    magnitude=1+s.mean(dtype=L)+(np.sum(np.abs(q*qlog),dtype=L)+np.sum(np.abs(s*slog),dtype=L))/N
    arithmetic=4*(gamma(operations)+LIBM*UL)*magnitude
    error=mass_error+arithmetic
    return dict(value=float(value),lower=lower(value-error),upper=upper(value+error),
        mass_allowance=upper(mass_error),arithmetic_allowance=upper(arithmetic),operations=operations,
        spot_checks=dict(log_q=log_spots(q),log_mass=log_spots(s))),q,e_q


def precondition(X):
    N,d=X.shape;assert np.all(X[:,-1]==1);normal_finite(X,np.float64,'X')
    C=np.eye(d)
    if d>1:
        mean=X[:,:-1].mean(0);std=X[:,:-1].std(0)
        # Numerical reference coordinates have no training-specific std floor.
        # Preserve even tiny nonconstant columns;do not replace them by zero.
        std=np.where(std>0,std,1.)
        C[np.arange(d-1),np.arange(d-1)]=1/std;C[-1,:-1]=-mean/std
    scaled=X@C;_,singular,Vh=np.linalg.svd(scaled,full_matrices=False)
    used_svd=(len(singular)==d and np.all(singular>0))
    W=C@(Vh.T*(np.sqrt(N)/singular)[None,:]) if used_svd else C
    assert W.shape==(d,d) and np.isfinite(W).all()
    normal_finite(W,np.float64,'W')
    Z=X@W;G=Z.T@Z/N
    normal_finite(Z,np.float64,'Z');normal_finite(G,np.float64,'Gram')
    # W is treated as stored exact FP64 data. Verify invertibility by a
    # residual bound;the SVD's numerical rank is never used to drop directions.
    try:
        inverse=np.linalg.inv(W);normal_finite(inverse,np.float64,'inverse_candidate')
        right=norm_up(W@inverse-np.eye(d))+4*gamma(d,U64)*norm_up(np.abs(W)@np.abs(inverse))
        left=norm_up(inverse@W-np.eye(d))+4*gamma(d,U64)*norm_up(np.abs(inverse)@np.abs(W))
        inverse_defect=min(left,right)
    except np.linalg.LinAlgError:inverse_defect=left=right=L(np.inf)
    abs_product=np.abs(X)@np.abs(W)
    normal_finite(abs_product,np.float64,'absolute_dot_magnitudes')
    eZ=4*gamma(d,U64)*norm_up(abs_product)
    znorm=norm_up(Z)
    gram_error=4*gamma(N+4,U64)*norm_up(np.abs(Z).T@np.abs(Z)/N)+(2*znorm*eZ+eZ*eZ)/N
    gram_error+=4*gamma(8*d*d+100,U64)*(1+norm_up(G))
    identity_defect=norm_up(G-np.eye(d))
    lambda_low=L(1)-identity_defect-gram_error
    rownorm=np.sqrt(np.sum(Z.astype(L)**2,axis=1,dtype=L)).max()
    h_up=(rownorm+eZ)*(1+4*gamma(N*d+100,U64))
    verified=bool(inverse_defect<1 and lambda_low>L('.5'))
    report=dict(dimension=d,N=N,used_all_singular_directions=True,used_svd_preconditioner=bool(used_svd),
        smallest_reported_singular_value=float(singular[-1]),largest_reported_singular_value=float(singular[0]),
        numerical_singular_values_are_not_rank_proof=True,
        invertibility_defect_upper=upper(inverse_defect) if np.isfinite(inverse_defect) else None,
        left_inverse_defect_upper=upper(left) if np.isfinite(left) else None,
        right_inverse_defect_upper=upper(right) if np.isfinite(right) else None,
        gram_identity_defect_upper=upper(identity_defect),gram_error_upper=upper(gram_error),
        gram_lambda_lower=lower(lambda_low),feature_norm_upper=upper(h_up),rank_and_coordinate_change_verified=verified)
    return W,Z,G,report


def moment_upper(X,W,T,q,e_q):
    X=np.asarray(X,dtype=L);W=np.asarray(W,dtype=L);T=np.asarray(T,dtype=L)
    N,n=T.shape;d=X.shape[1];residual=q-T
    e_residual=e_q+4*UL*(np.abs(q)+np.abs(T))
    # Form the whitened design in extended precision before the moment;
    # propagate its errors instead of amplifying an uncentered moment by |W|.
    Z=X@W;eZ=4*gamma(d+4)*(np.abs(X)@np.abs(W))
    value=(Z.T@residual)/N
    evalue=4*gamma(N+4)*(np.abs(Z).T@np.abs(residual))/N
    evalue+=4*(np.abs(Z).T@e_residual+eZ.T@(np.abs(residual)+e_residual))/N
    return dict(estimated_norm=float(norm_up(value)),arithmetic_norm_allowance=upper(norm_up(evalue)),
        upper=upper(norm_up(value)+norm_up(evalue)))


def dual_points(X,W,Z,G,T,coefficients):
    scores=X@coefficients;s=T.sum(1)
    shifted=scores-scores.max(1,keepdims=True);q=np.exp(shifted);q*= (s/q.sum(1))[:,None]
    result=[('weighted_softmax',q.copy())]
    for iteration in (1,2):
        residual=q-T;moment=Z.T@residual/len(T)
        try:correction=Z@np.linalg.solve(G,moment)
        except np.linalg.LinAlgError:
            break
        residual=q-correction-T;residual-=residual.mean(1,keepdims=True)
        negative=residual<0
        # Prospective positivity margin is half the ROW MINIMUM,not T_ij/2.
        floor=.5*T.min(1,keepdims=True)
        factor=min(1.,float(np.min(np.broadcast_to(T-floor,T.shape)[negative]/(-residual[negative])))) if negative.any() else 1.
        q=T+factor*residual;assert np.isfinite(q).all() and q.min()>0
        result.append(('gram_projection_'+str(iteration),q.copy()))
    return result


def fit_arrays(X,T,progress=None,attained=None):
    started=time.monotonic();X=np.asarray(X,dtype=np.float64);T=np.asarray(T,dtype=np.float64)
    assert X.ndim==T.ndim==2 and len(X)==len(T) and T.shape[1]>=2
    assert np.isfinite(X).all() and np.isfinite(T).all() and T.min()>=np.finfo(np.float64).tiny
    normal_finite(T,np.float64,'stored_targets')
    N,n=T.shape;d=X.shape[1];W,Z,G,geometry=precondition(X)
    s=T.sum(1);calls=0
    def objective(flat):
        nonlocal calls
        theta=flat.reshape(d,n);theta=theta-theta.mean(1,keepdims=True)
        b=W@theta;score=X@b;assert np.isfinite(score).all()
        maximum=score.max(1,keepdims=True);exp=np.exp(score-maximum);q=exp/exp.sum(1,keepdims=True)
        loss=float((s*(np.log(exp.sum(1))+maximum[:,0])-(T*score).sum(1)).mean())
        gradient=W.T@(X.T@(q*s[:,None]-T)/N);gradient-=gradient.mean(1,keepdims=True)
        assert np.isfinite(loss) and np.isfinite(gradient).all();calls+=1
        if progress is not None and (calls==1 or calls%25==0):progress(dict(function_calls=calls,objective=loss,gradient_linf=float(abs(gradient).max()),seconds=time.monotonic()-started))
        return loss,gradient.ravel()
    solution=minimize(objective,np.zeros(d*n),method='L-BFGS-B',jac=True,
        options=dict(maxiter=2000,maxfun=10000,maxcor=25,maxls=40,gtol=1e-12,ftol=2e-15))
    theta=solution.x.reshape(d,n);theta-=theta.mean(1,keepdims=True);coefficients=W@theta
    assert np.isfinite(coefficients).all()
    solver=dict(success=bool(solution.success),status=int(solution.status),message=str(solution.message),
        iterations=int(solution.nit),function_evaluations=int(solution.nfev),gradient_linf=float(abs(solution.jac).max()))
    if attained is not None:attained(dict(coefficients=coefficients,preconditioner=W),dict(solver=solver,geometry=geometry))
    upper_report=primal_interval(X,T,coefficients);entropy,_,_=entropy_interval(T)
    low=entropy['lower'];rows=[];arrays=dict(coefficients=coefficients,preconditioner=W)
    zero_upper=primal_interval(X,T,np.zeros_like(coefficients))['upper']
    radius=None
    if geometry['rank_and_coordinate_change_verified']:
        radius=upper(L(geometry['feature_norm_upper'])*L(min(zero_upper,upper_report['upper']))/
            (L(T.min())*np.sqrt(L(n)/L(n-1))*L(geometry['gram_lambda_lower']))*(1+gamma(100)))
    for name,Q in dual_points(X,W,Z,G,T,coefficients):
        arrays[name]=Q;ent,q,e_q=entropy_interval(T,Q)
        moment=moment_upper(X,W,T,q,e_q)
        bound=lower(L(ent['lower'])-L(radius)*L(moment['upper'])) if radius is not None else None
        if bound is not None:low=max(low,bound)
        rows.append(dict(name=name,entropy=ent,moment=moment,lower_bound=bound))
    assert low<=upper_report['upper'],'Inconsistent numerical interval'
    width=upper(L(upper_report['upper'])-L(low))
    record=dict(status='ATTAINED_REFERENCE',objective_scope='trainingB_unbounded_fixed_affine',N=N,n=n,dimension=d,
        solver=solver,
        geometry=geometry,primal=upper_report,target_weighted_entropy=entropy,optimum_radius_upper=radius,
        dual_candidates=rows,lower_bound=low,upper_bound=upper_report['upper'],interval_width=width,
        tight_interval=bool(geometry['rank_and_coordinate_change_verified'] and width<=1e-6),
        training_D_floor_lower=max(0.,lower(L(low)-L(entropy['upper']))),
        training_D_floor_upper=upper(L(upper_report['upper'])-L(entropy['lower'])),
        numerical_validity='Conditional IEEE/libm error-model interval,not a formal machine-arithmetic proof',
        longdouble_nmant=np.finfo(L).nmant,float64_nmant=np.finfo(np.float64).nmant,
        log_spot_check=log_spots(T),seconds=time.monotonic()-started,
        targets_renormalized=False,parameter_box_imposed=False,official_test_decoded=False,
        scope='Original training features and stored row masses;no population/validation/whole-native-class optimum')
    return record,arrays


def main(args):
    assert np.finfo(L).nmant>np.finfo(np.float64).nmant,'Production audit requires verified extended precision'
    queue=json.loads(args.queue.read_text());jobs=queue['jobs'];assert len(jobs)==3
    failed=[]
    for job in jobs:
        out=args.output/job['architecture'];out.mkdir(exist_ok=False)
        try:
            source=args.input_root/job['architecture'];receipt=json.loads((source/'complete.json').read_text())
            assert sha(source/'training_inputs.npz')==job['input_sha256']==receipt['input_sha256']
            assert sha(source/'complete.json')==job['complete_sha256']
            with np.load(source/'training_inputs.npz',allow_pickle=False) as a:
                phi=a['features'];T=a['targets'];kept=a['kept_feature_columns'];removed=a['constant_feature_columns'];constant=a['constant_feature_values']
                assert np.array_equal(phi[:,removed],np.broadcast_to(constant,(len(phi),len(removed))))
                assert np.array_equal(np.sort(np.r_[kept,removed]),np.arange(phi.shape[1]))
                X=np.column_stack([phi[:,kept].astype(np.float64),np.ones(len(phi))])
            def retain_attained(arrays,details):
                np.savez_compressed(out/'attained.npz',**arrays)
                write(out/'attained.json',dict(status='ATTAINED_OPTIMALITY_UNKNOWN',job=job,details=details,
                    arrays_sha256=sha(out/'attained.npz'),source_sha256=sha(Path(__file__)),official_test_decoded=False))
            report,arrays=fit_arrays(X,T,lambda p:print(json.dumps(dict(architecture=job['architecture'],**p)),flush=True),retain_attained)
            np.savez_compressed(out/'reference.npz',**arrays)
            report.update(job=job,input_receipt=receipt,arrays_sha256=sha(out/'reference.npz'),
                source_sha256=sha(Path(__file__)),numpy=np.__version__,scipy=__import__('scipy').__version__)
            write(out/'complete.json',report)
            print(json.dumps(dict(architecture=job['architecture'],width=report['interval_width'],tight=report['tight_interval'])),flush=True)
        except Exception:
            failed.append(job['architecture']);write(out/'failure.json',dict(job=job,traceback=traceback.format_exc(),
                attained_point_preserved=(out/'attained.npz').exists(),optimality='UNKNOWN'))
    write(args.output/'queue_complete.json',dict(jobs=3,failed=failed,all_successful=not failed,official_test_decoded=False))
    if failed:raise RuntimeError('Preserved numerical reference failures:'+str(failed))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
