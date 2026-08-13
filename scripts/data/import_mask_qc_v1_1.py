#!/usr/bin/env python3
"""Import explicit human mask QC decisions for a frozen manifest subset."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CRITERIA=("target_object_coverage","background_leakage","non_target_object_inclusion","thin_structure_preservation","occlusion_boundary_handling","object_identity_consistency")
NOTE="Human review confirmed adequate A/B target coverage, no material leakage or non-target inclusion, acceptable boundaries, no improper overlap, and correct object selection."

def args_parse():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--generated-results",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--subset-name",required=True); g=p.add_mutually_exclusive_group(required=True); g.add_argument("--review-input",type=Path); g.add_argument("--confirm-all-pass",action="store_true"); p.add_argument("--reviewer"); p.add_argument("--review-timestamp"); return p.parse_args()
def rows(path):
 with path.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def stamp(value):
 if value is None:return datetime.now(timezone.utc).isoformat()
 d=datetime.fromisoformat(value.replace("Z","+00:00"))
 if d.tzinfo is None:raise ValueError("timestamp requires timezone")
 return d.isoformat()
def write_jsonl(path,data):
 if path.exists():raise FileExistsError(path)
 t=path.with_name(f".{path.name}.tmp")
 with t.open("x",encoding="utf-8") as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 t.replace(path)
def write_json(path,data):
 if path.exists():raise FileExistsError(path)
 t=path.with_name(f".{path.name}.tmp"); t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); t.replace(path)
def write_csv(path,data):
 if path.exists():raise FileExistsError(path)
 fields=("candidate_id","mask_qc_status","reviewer","review_timestamp","review_note","failure_reasons","object_a_mask_status","object_b_mask_status")
 t=path.with_name(f".{path.name}.tmp")
 with t.open("x",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in data:w.writerow({"candidate_id":r["candidate_id"],"mask_qc_status":r["mask_qc_status"],"reviewer":r["mask_qc_reviewer"],"review_timestamp":r["mask_qc_timestamp"],"review_note":r["mask_qc_note"],"failure_reasons":json.dumps(r["failure_reasons"]),"object_a_mask_status":r["object_a_mask_status"],"object_b_mask_status":r["object_b_mask_status"]})
 t.replace(path)
def aggregate_status(criteria):
 values=set(criteria.values())
 if "fail" in values:return "fail"
 if "human_review" in values:return "human_review"
 return "pass"
def validate_reviews(review,ids,subset):
 m={r.get("candidate_id"):r for r in review}
 if len(m)!=len(review) or set(m)!=ids:raise ValueError("review IDs differ from generated IDs")
 allowed={"pass","fail","human_review"}
 for r in review:
  if r.get("subset_name")!=subset:raise ValueError(f"subset mismatch: {r['candidate_id']}")
  if r.get("mask_qc_status") not in allowed:raise ValueError("incomplete aggregate mask review")
  for side in ("object_a_qc","object_b_qc"):
   if set(r.get(side,{}))!=set(CRITERIA) or any(v not in allowed for v in r[side].values()):raise ValueError(f"incomplete {side}: {r['candidate_id']}")
  if r.get("pair_qc",{}).get("ab_mask_overlap") not in allowed:raise ValueError("incomplete pair review")
  reasons=r.get("failure_reasons",r.get("mask_failure_reasons",[]))
  if r["mask_qc_status"]=="pass" and reasons:raise ValueError("pass forbids failure reasons")
  if r["mask_qc_status"]=="fail" and not reasons:raise ValueError("fail requires reason")
  if r["mask_qc_status"] in {"fail","human_review"} and not r.get("mask_qc_note"):raise ValueError("fail/human_review requires note")
  if not r.get("mask_qc_reviewer") or not r.get("mask_qc_timestamp"):raise ValueError("reviewer/timestamp required")
def main():
 a=args_parse(); generated=rows(a.generated_results); ids={r.get("candidate_id") for r in generated}
 if not generated or len(ids)!=len(generated) or None in ids:raise ValueError("generated results require unique candidates")
 reviewable=[]; failures=[]
 for r in generated:
  if r.get("pilot_subset")!=a.subset_name or r.get("mask_qc_status")!="not_tested" or r.get("edit_qc_status")!="not_tested" or r.get("object_identity_preservation_status")!="not_tested":raise ValueError(f"invalid generated state: {r.get('candidate_id')}")
  status=r.get("mask_generation_status")
  if status=="generated":reviewable.append(r)
  elif status=="segmentation_failed":failures.append(r)
  else:raise ValueError(f"unknown mask_generation_status for {r.get('candidate_id')}: {status!r}")
 reviewable_ids={r["candidate_id"] for r in reviewable}
 if a.confirm_all_pass:
  if not (a.reviewer or "").strip():raise ValueError("--confirm-all-pass requires --reviewer")
  ts=stamp(a.review_timestamp); review=[]
  for r in reviewable: review.append({"schema_version":"1.1","subset_name":a.subset_name,"mask_test_id":r["mask_test_id"],"candidate_id":r["candidate_id"],"object_a_qc":{c:"pass" for c in CRITERIA},"object_b_qc":{c:"pass" for c in CRITERIA},"pair_qc":{"ab_mask_overlap":"pass"},"object_a_mask_status":"pass","object_b_mask_status":"pass","mask_qc_status":"pass","failure_reasons":[],"mask_failure_reasons":[],"mask_qc_note":NOTE,"mask_qc_reviewer":a.reviewer,"mask_qc_timestamp":ts,"technical_status_before":r["technical_status_before"],"technical_status_after_mask_qc":None,"technical_status_update_approved":False})
 else:review=rows(a.review_input)
 validate_reviews(review,reviewable_ids,a.subset_name)
 normalized=[]
 for q in review:
  q=deepcopy(q); reasons=q.get("failure_reasons",q.get("mask_failure_reasons",[]))
  q["failure_reasons"]=list(reasons); q["mask_failure_reasons"]=list(reasons)
  q["object_a_mask_status"]=q.get("object_a_mask_status",aggregate_status(q["object_a_qc"]))
  q["object_b_mask_status"]=q.get("object_b_mask_status",aggregate_status(q["object_b_qc"]))
  normalized.append(q)
 review=normalized; by={r["candidate_id"]:r for r in review}; reviewed=[]
 for r in generated:
  if r.get("mask_generation_status")=="segmentation_failed":reviewed.append(deepcopy(r));continue
  q=by[r["candidate_id"]]; x=deepcopy(r); reasons=q.get("failure_reasons",q.get("mask_failure_reasons",[])); x.update({"mask_qc_status":q["mask_qc_status"],"object_a_mask_status":q.get("object_a_mask_status",aggregate_status(q["object_a_qc"])),"object_b_mask_status":q.get("object_b_mask_status",aggregate_status(q["object_b_qc"])),"object_a_qc":q["object_a_qc"],"object_b_qc":q["object_b_qc"],"pair_qc":q["pair_qc"],"failure_reasons":reasons,"mask_failure_reasons":reasons,"mask_qc_note":q.get("mask_qc_note"),"mask_qc_reviewer":q["mask_qc_reviewer"],"mask_qc_timestamp":q["mask_qc_timestamp"]}); reviewed.append(x)
 out={"csv":a.output_dir/f"mask_qc_review_{a.subset_name}_v1.1.csv","review":a.output_dir/f"mask_qc_review_{a.subset_name}_v1.1.jsonl","reviewed":a.output_dir/f"mask_results_{a.subset_name}_v1.1_reviewed.jsonl","summary":a.output_dir/f"mask_qc_summary_{a.subset_name}_v1.1.json"}
 if any(p.exists() for p in out.values()):raise FileExistsError("QC output exists")
 write_csv(out["csv"],review);write_jsonl(out["review"],review);write_jsonl(out["reviewed"],reviewed);summary={"schema_version":"1.1","subset_name":a.subset_name,"source_candidate_count":len(generated),"segmentation_success_count":len(reviewable),"segmentation_failure_count":len(failures),"reviewed_count":len(review),"mask_qc_status_counts":dict(Counter(r["mask_qc_status"] for r in review)),"segmentation_failure_records_preserved":len(failures),"asset_modifications":0,"generated_results_sha256":sha(a.generated_results)};write_json(out["summary"],summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
