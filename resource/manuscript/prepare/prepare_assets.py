"""Build manuscript assets from frozen summaries; never train or infer a model."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results/capped_logits_completion_2026-10-22'
INPUTS = [
    RESULTS / 'official_report_attempt001/all648.csv',
    RESULTS / 'official_summary_attempt001/summary.json',
    RESULTS / 'official_tie_attempt001.json',
    RESULTS / 'synthetic_confirmation_summary_attempt001/summary.json',
    RESULTS / 'cpu_reference_phases/raw_affine_adequacy_v1/output/summary.json',
    RESULTS / 'cold_replay_verification_attempt001/audit.json',
    RESULTS / 'official_bound_attempt001.json',
    RESULTS / 'classifier_feasibility_identity_attempt001/summary.json',
]
EVIDENCE = []
ARCH = [('fashion_cnn', 'FashionCNN'), ('fashion_mlp', 'FashionMLP'), ('resnet18', 'ResNet18')]
HEADS = ['raw', 'box', 'tanh', 'ball']
COLORS = ['#475569', '#a6273e', '#167d8d', '#6e57a5']

def load(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fmt(v, digits=3):
    v = float(v)
    if v == 0:
        return '$0$'
    if .001 <= abs(v) < 1000:
        return f'${v:.{digits}g}$'
    a, b = f'{v:.{digits-1}e}'.split('e')
    return f'${a}\\!\\times\\!10^{{{int(b)}}}$'

def fmt_bound(v, digits, upper):
    """Conservative significant-digit display; do not nearest-round an endpoint."""
    exact = Decimal(str(v))
    if exact == 0:
        return '$0$'
    quantum = Decimal(1).scaleb(exact.adjusted() - digits + 1)
    rounded = exact.quantize(quantum, rounding=ROUND_CEILING if upper else ROUND_FLOOR)
    assert rounded >= exact if upper else rounded <= exact
    return fmt(float(rounded), digits)

def table(name, columns, rows):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += '\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (HERE / 'tables' / name).write_text(text)

def longtable(name, columns, header, rows):
    line = ' & '.join(header) + r' \\'
    text = '\\begin{longtable}{' + columns + '}\n\\toprule\n' + line
    text += '\n\\midrule\n\\endhead\n'
    text += '\n'.join(' & '.join(map(str, r)) + r' \\' for r in rows)
    text += '\n\\bottomrule\n\\end{longtable}\n'
    (HERE / 'tables' / name).write_text(text)

def stat(xs, key):
    return float(np.mean([float(x[key]) for x in xs]))

def savefig(fig, name):
    fig.savefig(HERE / 'figures' / (name + '.pdf'), bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(HERE / 'figures' / (name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    for d in ['tables', 'figures', 'data']:
        (HERE / d).mkdir(exist_ok=True)
    rows = list(csv.DictReader(INPUTS[0].open()))
    official, ties, synthetic, q3, replay, bound, feasibility = map(load, INPUTS[1:])
    assert len(rows) == 648 == official['model_slots'] == len(ties['rows'])
    assert sum(x['derived'] == 'True' for x in rows) == 180
    assert len({(r['phase'], r['job']) for r in rows}) == 648
    assert len(synthetic['rows']) == 980 and synthetic['G1_passing_configurations'] == 10
    assert synthetic['G1_total_configurations'] == 11
    assert replay['status'] == 'PASS' and replay['all_metrics_and_saved_array_values_bitwise']
    assert replay['serialized_test_NPZ_bytes_equal'] == 648
    assert replay['complete_JSON_bytes_equal'] == 648
    assert bound['status'] == 'PASS' and bound['checked_models'] == 339
    assert feasibility['status'] == 'PASS' and len(feasibility['rows']) == 1176
    assert feasibility['all_deployed_first_argmax_and_top_ties_preserved']
    assert len(q3['canonical_problems']) == 3 and len(q3['raw_learners']) == 30
    assert len(q3['transport_checks']) == 15 and all(t['passed'] for t in q3['transport_checks'])
    # Keep the exact source CSV; condensed companion files expose every seed.
    (HERE / 'data/all648.csv').write_bytes(INPUTS[0].read_bytes())
    cohort = []
    cohort_labels={
        'bounded_confirmation_v1':'Natural, bounded heads',
        'disjoint_student_confirmation_v1':'Disjoint frozen-logit readouts',
        'native_hard_head_transfer_v1':'Native tanh-to-box transfers',
        'pixel_affine_primary_v1':'Fixed affine, primary',
        'pixel_affine_raw_head_transfer_v1':'Fixed affine, raw-to-box',
        'pixel_affine_secondary_v1':'Fixed affine, rate / 10',
        'pixel_affine_tanh_head_transfer_v1':'Fixed affine, tanh-to-box',
        'pixel_affine_teacher_reference_v1':'Fixed affine, teacher references',
        'pixel_capacity_replay_v1':'Full-pixel capacity copies',
        'pixel_kkt_confirmation_v1':'Random pixel, native KKT',
        'pixel_kkt_hard_v1':'Random pixel, sampled labels',
        'pixel_kkt_warm_v1':'Native pixel, warm initialization',
        'pixel_mixed_confirmation_v1':'Random pixel, mixed law',
        'raw_confirmation_v1':'Natural, raw head',
    }
    for phase, count in sorted(Counter(r['phase'] for r in rows).items()):
        group = [r for r in rows if r['phase'] == phase]
        vg = sum(r['validation_gate'] == 'True' for r in group)
        tg = sum(r['official_gate'] == 'True' for r in group)
        natural = all(r['kind'] == 'natural' for r in group)
        cohort.append([cohort_labels[phase], count,
                       'derived' if group[0]['derived'] == 'True' else ('natural' if natural else 'trained'),
                       '--' if natural else vg, '--' if natural else tg])
    table('cohorts.tex', 'lrrrr', [['Cohort', 'Slots', 'Role', 'Val.', 'Official'], r'\midrule'.split('|')] + cohort)
    # Special row separators are written explicitly, not as cells.
    p = HERE / 'tables/cohorts.tex'
    p.write_text(p.read_text().replace(r'\midrule \\', r'\midrule'))
    primary = []
    seedrows = []
    tiedrows = []
    for arch, label in ARCH:
        xs = [r for r in rows if r['phase'] == 'pixel_affine_primary_v1' and r['architecture'] == arch and r['head'] == 'box']
        ts = [r['metrics'] for r in ties['rows'] if r['job'].startswith('pixel_affine_primary_v1__') and r['config']['architecture'] == arch and r['config']['head'] == 'box']
        assert len(xs) == len(ts) == 5 and all(r['official_gate'] == 'True' for r in xs)
        delta = [float(r['Q2_signed_head_minus_raw']) for r in xs]
        assert min(delta) > 0 and all(t['best_model_tie_regret'] == 0 for t in ts)
        primary.append([label, '5/5', fmt(stat(xs, 'official_mean_D')), f"{stat(xs, 'official_near'):.5f}", fmt(np.mean(delta), 4), fmt(np.std(delta, ddof=1), 3)])
        tiedrows.append([label] + [fmt(stat(ts,k),5) for k in ['first_regret','last_regret','exact_uniform_model_tie_regret','best_model_tie_regret','worst_model_tie_regret']] + [f"{stat(ts,'exact_tie_fraction'):.5f}"])
        for r in xs:
            seedrows.append([label, r['seed'], fmt(r['official_mean_D'],5), r['Q2_near_examples'], fmt(r['Q2_signed_head_minus_raw'],6)])
    table('primary.tex', 'lrrrrr', [['Representation','Q1','Mean $D$','Coverage',r'$\Delta_h$','Seed SD'], [r'\midrule']] + primary)
    table('ties.tex','lrrrrrr', [['Representation','First','Last','Uniform','Best','Worst','Tie fraction'], [r'\midrule']] + tiedrows)
    table('primary_seeds.tex','lrrrr', [['Representation','Seed','Mean $D$','$|N_h|$',r'$\Delta_h$'],[r'\midrule']] + seedrows)
    for name in ['primary.tex','ties.tex','primary_seeds.tex']:
        p = HERE / 'tables' / name
        p.write_text(p.read_text().replace(r'\midrule \\', r'\midrule'))
    naturals=[]
    affine=[]
    for arch,label in ARCH:
        for head in HEADS:
            ns=[r for r in rows if r['kind']=='natural' and r['architecture']==arch and r['head']==head]
            assert len(ns)==5
            naturals.append([label,head,fmt(stat(ns,'official_CE'),6),f"{100*stat(ns,'official_error'):.3f}"])
        for arm in ['primary','secondary']:
            for head in HEADS:
                xs=[r for r in rows if r['phase']==f'pixel_affine_{arm}_v1' and r['architecture']==arch and r['head']==head]
                assert len(xs)==5
                affine.append([label,arm,head,fmt(stat(xs,'official_mean_D'),4),f"{stat(xs,'official_near'):.5f}",fmt(stat(xs,'official_regret'),4),f"{sum(r['official_gate']=='True' for r in xs)}/5"])
    longtable('natural.tex','llrr',['Representation','Head','CE (nats)',r'Error (\%)'],naturals)
    longtable('affine.tex','lllrrrr',['Representation','Arm','Head','Mean $D$','Coverage','Regret','Q1'],affine)
    qrows=[]
    for arch,label in ARCH:
        ref=next(x for x in q3['canonical_problems'] if x['architecture']==arch)
        x=[r for r in q3['raw_learners'] if r['architecture']==arch and r['arm']=='primary']
        y=[r for r in q3['raw_learners'] if r['architecture']==arch and r['arm']=='secondary']
        qrows.append([label,fmt_bound(ref['interval_width'],4,True),fmt_bound(ref['training_D_floor_lower'],5,False),fmt_bound(min(r['empirical_class_gap_lower'] for r in x),4,False),fmt_bound(max(r['empirical_class_gap_upper'] for r in x),4,True),f"{sum(r['raw_adequacy_confirmed'] for r in x)}/5",fmt_bound(max(r['empirical_class_gap_upper'] for r in y),4,True)])
    table('q3.tex','lrrrrrr',[['Representation','Ref. width','$D$ floor','Primary min.','Primary max.','Q3','Second. max.'],[r'\midrule']]+qrows)
    p=HERE/'tables/q3.tex';p.write_text(p.read_text().replace(r'\midrule \\',r'\midrule'))
    # All 196 original synthetic groups, including failures and precision/stress arms.
    sg=[]
    for g in sorted(synthetic['groups'],key=lambda g:(g['config']['group'],g['config']['n'],g['config']['beta'],g['config']['mode'],g['head'],g['objective'],g['config_id'])):
        c=g['config']; m=g['metrics']
        sg.append([c['group'].replace('_',r'\_'),str(c['n']),str(c['beta']),f"{c['ratio']:.3g}",c['mode'].replace('realizable','real.').replace('gaussian','Gauss.'),f"{c['depth']}x{c['width']}",g['head'],g['objective'].replace('distill','dist.'),fmt(m['mean_D']['mean'],3),f"{m['near_coverage']['mean']:.3f}",f"{g['combined_gate_seeds']}/5"])
    longtable('synthetic_groups.tex','lrrrllllrrr',['Group','$n$',r'$\beta$','$R$','Law','MLP','Head','Obj.','Mean $D$','Cov.','Gate'],sg)
    def dump_csv(name, vals):
        keys=list(vals[0]);f=io.StringIO(newline='');w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(vals);(HERE/'data'/name).write_text(f.getvalue())
    dump_csv('synthetic980.csv',[{**{k:r.get(k) for k in ['job','config_id','head','objective','seed','mean_D','near_coverage','regret','combined_gate']},**r['config']} for r in synthetic['rows']])
    dump_csv('raw_affine30.csv',[{k:r.get(k) for k in ['architecture','arm','seed','empirical_class_gap_lower','empirical_class_gap_upper','raw_adequacy_confirmed']} for r in q3['raw_learners']])
    plt.rcParams.update({'font.size':12.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    ratios=np.geomspace(1.001,1000,400)
    for n, style in zip([2,4,10,100], ['-', '--', '-.', ':']):
        axs[0].plot(ratios,(n-2)/(2*ratios+n-2),ls=style,label=f'n={n}')
    axs[0].set(xscale='log',xlabel='Probability-ratio cap R',ylabel='Sharp limiting residual beta',ylim=(-.02,1));axs[0].legend(frameon=False)
    n=10;ratio=28/3;beta=(n-2)/(2*ratio+n-2);slope=1+beta+2/(ratio-1)
    eta=np.geomspace(1e-8,.5,200);v=1/(2*ratio+n-2);t=.999*beta
    ds=np.log1p(ratio*v*np.expm1(-eta))+(ratio*v+t)*eta
    assert np.all(ds>0)
    axs[1].plot(ds,np.full_like(ds,t),color=COLORS[1],ls='--',label='Strict wrong sequence')
    axs[1].plot(ds,np.minimum(1,beta+slope*np.sqrt(2*ds)),color=COLORS[0],label='Finite upper bound')
    axs[1].axhline(beta,color='black',ls=':',label='Sharp residual')
    axs[1].set(xscale='log',xlabel='Conditional CE excess D (nats)',ylabel='Posterior regret',ylim=(0,1.02));axs[1].legend(frameon=False,fontsize=12)
    savefig(fig,'theory')
    fig,axs=plt.subplots(2,3,figsize=(10,5.2),layout='constrained')
    for col,(arch,label) in enumerate(ARCH):
        for h,(head,color) in enumerate(zip(HEADS,COLORS)):
            for arm,marker,offset in [('primary','o',-.07),('secondary','x',.07)]:
                xs=sorted([r for r in rows if r['phase']==f'pixel_affine_{arm}_v1' and r['architecture']==arch and r['head']==head],key=lambda r:r['seed'])
                xx=h+offset+np.linspace(-.025,.025,5)
                axs[0,col].scatter(xx,[float(r['official_mean_D']) for r in xs],marker=marker,color=color,s=22)
                axs[1,col].scatter(xx,[float(r['official_near']) for r in xs],marker=marker,color=color,s=22)
        axs[0,col].set(yscale='log',title=label,xticks=range(4),xticklabels=HEADS)
        axs[0,col].axhline(.001,color='black',ls=':',lw=1)
        axs[1,col].set(xticks=range(4),xticklabels=HEADS,ylim=(-.03,1.03));axs[1,col].axhline(.5,color='black',ls=':',lw=1)
    axs[0,0].set_ylabel('Official mean D (nats)');axs[1,0].set_ylabel('Near-set coverage')
    fig.suptitle('Fixed affine: circles primary, crosses rate / 10; every seed shown',fontsize=12.5)
    savefig(fig,'affine_learning')
    fig,axs=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,(arch,label) in zip(axs,ARCH):
        xs=sorted([r for r in rows if r['phase']=='pixel_affine_primary_v1' and r['architecture']==arch and r['head']=='box'],key=lambda r:r['seed'])
        vals=[float(r['Q2_signed_head_minus_raw']) for r in xs]
        ax.scatter(range(1,6),vals,color=COLORS[1]);ax.axhline(0,color='black',ls=':',lw=1)
        ax.set(title=label,xlabel='Paired seed index',xticks=range(1,6));ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axs[0].set_ylabel('Near-set head-minus-raw regret')
    savefig(fig,'same_model_regret')
    manifest={'scientific_source_commit':'c6d7d85cbbd5f533742c32da31cd861d087cca32','completion_commit':'a5338e868f28915b87c14da2024be55edefbe65e','inputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in INPUTS+EVIDENCE],'generator_sha256':sha(Path(__file__)),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'scope':'Saved-data manuscript assembly only; no training or inference','checks':{'slots':648,'derived':180,'trained':468,'synthetic':980,'primary_box_passes':15,'secondary_passes':sum(r['official_gate']=='True' for r in rows if r['phase']=='pixel_affine_secondary_v1'),'cold_replay':replay['status']},'outputs':[]}
    for folder in ['data','tables','figures']:
        for p in sorted((HERE/folder).iterdir()):manifest['outputs'].append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
    (HERE/'asset_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['checks']))

if __name__=='__main__':
    main()
