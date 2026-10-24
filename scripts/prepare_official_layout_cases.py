"""Prepare official non-Open layout lists without launching a robot or model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.official_layouts import TASK_GROUP, OPEN_TASKS, CATALOG_URL, POLICY, make_case, validate_plan

def prepare(source, tasks=None, collection_seed=0, collection_layout=0,
            validation_seeds=(0, 1, 2), validation_layout=1):
    selected = sorted(TASK_GROUP) if tasks is None else list(tasks)
    if not selected or len(set(selected)) != len(selected) or any(t not in TASK_GROUP for t in selected):
        raise ValueError("Select unique tasks from the official non-Open catalog")
    if len(set(validation_seeds)) < 3:
        raise ValueError("Keep the existing requirement of at least three validation seeds")
    collection = [make_case(source, t, collection_seed, collection_layout) for t in selected]
    validation = [make_case(source, t, seed, validation_layout)
                  for t in selected for seed in validation_seeds]
    split_report = validate_plan(collection, validation)
    return dict(layout_policy=POLICY, layout_split=split_report, collection_cases=collection, validation_cases=validation,
        required_capabilities=sorted({TASK_GROUP[t] for t in selected}),
        required_seeds=list(validation_seeds), excluded_open_tasks=sorted(OPEN_TASKS),
        catalog_source=CATALOG_URL, training_rollout_export=False,
        note="Configuration fragment only. Public layout source is preserved; no official hidden score claim.")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--tasks",nargs="+")
    parser.add_argument("--collection-seed",type=int,default=0)
    parser.add_argument("--collection-layout",type=int,default=0)
    parser.add_argument("--validation-seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--validation-layout",type=int,default=1)
    args=parser.parse_args()
    plan=prepare(args.robodojo_root,args.tasks,args.collection_seed,args.collection_layout,
                 args.validation_seeds,args.validation_layout)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(output=str(path),collection_cases=len(plan["collection_cases"]),
                         validation_cases=len(plan["validation_cases"]),open_cases=0)))
