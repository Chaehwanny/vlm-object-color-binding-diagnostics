#!/usr/bin/env python3
"""Import explicit human Natural Edited QC decisions for a frozen subset."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path

CRITERIA={"target_color_change":"target_color_change_status","original_color_residue":"original_color_residue_status","outside_mask_preservation":"outside_mask_preservation_status","color_leakage":"color_leakage_status","boundary_quality":"boundary_quality_status","texture_shading_pattern_preservation":"texture_shading_preservation_status","binding_swap_achieved":"target_binding_achievement_status","prompt_answer_validity":"prompt_answer_validity_after_edit_status"}
NOTE="Human review confirmed the A/B color swap, sufficient target-color movement, acceptable residue, outside-mask preservation, no material leakage or boundary artifact, preserved identity/texture/shading, and unambiguous answers."
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--generated-results",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--subset-name",required=True);g=p.add_mutually_exclusive_group(required=True);g.add_argument("--review-input",type=Path);g.add_argument("--confirm-all-pass",action="store_true");p.add_argument("--reviewer");p.add_argument("--review-timestamp");return p.parse_args()
def rows(p):
 with p.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def ts(v):
 if v is None:return datetime.now(timezone.utc).isoformat()
 d=datetime.fromisoformat(v.replace("Z","+00:00"))
 if d.tzinfo is None:raise ValueError("timestamp requires timezone")
 return d.isoformat()
def jsonl(p,data):
 if p.exists():raise FileExistsError(p)
 t=p.with_name(f".{p.name}.tmp")
 with t.open("x",encoding="utf-8") as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 t.replace(p)
def js(p,data):
 if p.exists():raise FileExistsError(p)
 t=p.with_name(f".{p.name}.tmp");t.write_text(json.dumps(data,indent=2)+"\n");t.replace(p)
def csvout(p,data):
 if p.exists():raise FileExistsError(p)
 fields=("candidate_id","edit_qc_status","object_identity_preservation_status","reviewer","review_timestamp","review_note","failure_reasons",*CRITERIA.values());t=p.with_name(f".{p.name}.tmp")
 with t.open("x",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in data:w.writerow({"candidate_id":r["candidate_id"],"edit_qc_status":r["edit_qc_status"],"object_identity_preservation_status":r["object_identity_preservation_status"],"reviewer":r["reviewer"],"review_timestamp":r["review_timestamp"],"review_note":r.get("review_note"),"failure_reasons":json.dumps(r["failure_reasons"]),**{flat:r.get(flat,r["criteria"].get(c)) for c,flat in CRITERIA.items()}})
 t.replace(p)
def validate(review,ids,subset):
 m={r.get("candidate_id"):r for r in review}
 if len(m)!=len(review) or set(m)!=ids:raise ValueError("review IDs differ")
 allowed={"pass","fail","human_review"}
 for r in review:
  if r.get("subset_name")!=subset:raise ValueError("subset mismatch")
  if r.get("edit_qc_status") not in allowed or r.get("object_identity_preservation_status") not in allowed:raise ValueError("incomplete aggregate review")
  c=r.get("criteria",{});required={*CRITERIA,"object_identity_preservation"}
  if set(c)!=required or any(v not in allowed for v in c.values()):raise ValueError("incomplete criteria")
  if r["edit_qc_status"]=="pass" and r.get("failure_reasons",[]):raise ValueError("pass forbids reasons")
  if r["edit_qc_status"]=="fail" and not r.get("failure_reasons"):raise ValueError("fail requires reason")
  if not r.get("reviewer") or not r.get("review_timestamp"):raise ValueError("reviewer/timestamp required")
def main():
 a=parse();gen=rows(a.generated_results);ids={r.get("candidate_id") for r in gen}
 if not gen or len(ids)!=len(gen) or None in ids:raise ValueError("generation results require unique IDs")
 generated=[r for r in gen if r.get("generation_status")=="generated"]
 failed=[r for r in gen if r.get("generation_status")=="failed"]
 if len(generated)+len(failed)!=len(gen):raise ValueError("unknown generation status")
 for r in gen:
  if r.get("pilot_subset",a.subset_name)!=a.subset_name or r.get("mask_qc_status")!="pass" or r.get("edit_qc_status")!="not_tested" or r.get("object_identity_preservation_status")!="not_tested":raise ValueError(f"invalid generation state: {r.get('candidate_id')}")
 generated_ids={r["candidate_id"] for r in generated}
 if a.confirm_all_pass:
  if not(a.reviewer or "").strip():raise ValueError("all-pass requires reviewer")
  stamp=ts(a.review_timestamp);review=[]
  for r in generated:review.append({"schema_version":"1.1","subset_name":a.subset_name,"candidate_id":r["candidate_id"],"criteria":{**{c:"pass" for c in CRITERIA},"object_identity_preservation":"pass"},"edit_qc_status":"pass","object_identity_preservation_status":"pass","failure_reasons":[],"review_note":NOTE,"reviewer":a.reviewer,"review_timestamp":stamp,**{flat:"pass" for flat in CRITERIA.values()}})
 else:review=rows(a.review_input)
 validate(review,generated_ids,a.subset_name)
 if [r["candidate_id"] for r in review]!=[r["candidate_id"] for r in generated]:raise ValueError("review order differs from generated subset order")
 by={r["candidate_id"]:r for r in review};reviewed=[]
 for r in gen:
  if r["generation_status"]=="failed":reviewed.append(deepcopy(r));continue
  q=by[r["candidate_id"]];x=deepcopy(r);x.update({"edit_qc_status":q["edit_qc_status"],"object_identity_preservation_status":q["object_identity_preservation_status"],"natural_edit_qc_criteria":q["criteria"],"natural_edit_qc_reviewer":q["reviewer"],"natural_edit_qc_timestamp":q["review_timestamp"],"natural_edit_qc_note":q.get("review_note"),"failure_reasons":q.get("failure_reasons",[]),"controlled_original_qc_status":"not_tested","controlled_edited_qc_status":"not_tested","four_image_qc_status":"not_tested",**{flat:q.get(flat,q["criteria"].get(c)) for c,flat in CRITERIA.items()}});reviewed.append(x)
 out={"csv":a.output_dir/f"natural_edit_qc_review_{a.subset_name}_v1.1.csv","review":a.output_dir/f"natural_edit_qc_review_{a.subset_name}_v1.1.jsonl","reviewed":a.output_dir/f"color_edit_results_{a.subset_name}_v1.1_reviewed.jsonl","summary":a.output_dir/f"natural_edit_qc_summary_{a.subset_name}_v1.1.json"}
 if any(p.exists() for p in out.values()):raise FileExistsError("QC output exists")
 generation_failures_preserved=all(reviewed[i]==gen[i] for i in range(len(gen)) if gen[i]["generation_status"]=="failed")
 csvout(out["csv"],review);jsonl(out["review"],review);jsonl(out["reviewed"],reviewed);summary={"schema_version":"1.1","subset_name":a.subset_name,"candidate_count":len(gen),"generated_count":len(generated),"generation_failed_count":len(failed),"generation_failed_candidate_ids":[r["candidate_id"] for r in failed],"reviewed_count":len(review),"edit_qc_status_counts":dict(Counter(r["edit_qc_status"] for r in review)),"identity_status_counts":dict(Counter(r["object_identity_preservation_status"] for r in review)),"lineage_edit_qc_status_counts":dict(Counter(r["edit_qc_status"] for r in reviewed)),"lineage_identity_status_counts":dict(Counter(r["object_identity_preservation_status"] for r in reviewed)),"asset_modifications":0,"generation_failures_preserved":generation_failures_preserved,"generated_results_sha256":sha(a.generated_results)};js(out["summary"],summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
