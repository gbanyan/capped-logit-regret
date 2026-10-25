"""Recompute descriptive seed summaries from supplied all648.csv; no neural run.

Run with --resource-root pointing to an extracted ESM. The feature-precision
diagnostic additionally requires original training-input NPZ files and is not
recomputed by this lightweight script. Raw data and gates are never modified.
"""
import argparse
import csv
from pathlib import Path
import statistics


def main(root):
    with (root / "all648.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==648
    natural=[]; paired=[]; affine=[]
    for arch in ("fashion_cnn","fashion_mlp","resnet18"):
        groups={}
        for head in ("raw","box","tanh","ball"):
            group=sorted([r for r in rows if r["kind"]=="natural" and r["architecture"]==arch and r["head"]==head],key=lambda r:int(r["seed"]))
            assert len(group)==5
            groups[head]=group
            error=[100*float(r["official_error"]) for r in group]
            ce=[float(r["official_CE"]) for r in group]
            natural.append(dict(architecture=arch,head=head,seeds=5,error_mean_percent=statistics.mean(error),error_SD_pp=statistics.stdev(error),error_min_percent=min(error),error_max_percent=max(error),CE_mean=statistics.mean(ce),CE_SD=statistics.stdev(ce)))
        for head in ("box","tanh","ball"):
            assert [r["seed"] for r in groups[head]]==[r["seed"] for r in groups["raw"]]
            delta=[100*(float(h["official_error"])-float(r["official_error"])) for h,r in zip(groups[head],groups["raw"])]
            paired.append(dict(architecture=arch,head_minus_raw=head,paired_seeds=5,mean_difference_pp=statistics.mean(delta),SD_difference_pp=statistics.stdev(delta),SE_difference_pp=statistics.stdev(delta)/(5**.5),min_difference_pp=min(delta),max_difference_pp=max(delta)))
        for phase in ("pixel_affine_primary_v1","pixel_affine_secondary_v1"):
            for head in ("raw","box","tanh","ball"):
                group=[r for r in rows if r["phase"]==phase and r["architecture"]==arch and r["head"]==head]
                assert len(group)==5
                record=dict(architecture=arch,phase=phase,head=head,seeds=5)
                for key in ("official_mean_D","official_near","official_regret"):
                    values=[float(r[key]) for r in group]
                    record[key+"_mean"]=statistics.mean(values)
                    record[key+"_SD"]=statistics.stdev(values)
                affine.append(record)
    folder=root/"diagnostics";folder.mkdir(exist_ok=True)
    for name,records in (("natural_seed_dispersion.csv",natural),("paired_natural_error.csv",paired),("affine_seed_dispersion.csv",affine)):
        with (folder/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--resource-root",type=Path,required=True)
    main(p.parse_args().resource_root.resolve())
