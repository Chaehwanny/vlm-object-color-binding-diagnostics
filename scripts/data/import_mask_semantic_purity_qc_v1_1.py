#!/usr/bin/env python3
"""Import explicit human semantic-purity reviews; no automatic pass mode exists."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path

STATUSES={"pass","fail","human_review"}
LABELS={"skin","arm","hand","neck","face","hair","person_leg","pants","shoe","adjacent_object","background_region","other"}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--review-template",type=Path,required=True);p.add_argument("--review-input",type=Path,required=True);p.add_argument("--asset-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--subset-name",required=True);return p.parse_args()
def rows(p):
 with p.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def validate(r,template,subset,root):
 cid=r.get("candidate_id")
 if r.get("subset_name")!=subset:raise ValueError(f"{cid}: subset mismatch")
 for field in ("object_a_label","object_b_label","original_color_a","original_color_b","contact_sheet_path","contact_sheet_sha256"):
  if r.get(field)!=template.get(field):raise ValueError(f"{cid}: frozen field changed: {field}")
 path=root/r["contact_sheet_path"]
 if not path.is_file() or sha(path)!=r["contact_sheet_sha256"]:raise ValueError(f"{cid}: contact sheet missing/checksum mismatch")
 for side in ("a","b"):
  status=r.get(f"object_{side}_semantic_purity_status");inc=r.get(f"non_target_inclusion_{side}");labels=r.get(f"contamination_labels_{side}")
  if status not in STATUSES:raise ValueError(f"{cid}: incomplete semantic status {side}")
  if not isinstance(inc,bool) or not isinstance(labels,list) or len(labels)!=len(set(labels)) or not set(labels).issubset(LABELS):raise ValueError(f"{cid}: invalid contamination fields {side}")
  if status=="pass" and (inc or labels):raise ValueError(f"{cid}: pass conflicts with contamination {side}")
  if status=="fail" and (not inc or not labels):raise ValueError(f"{cid}: fail requires inclusion and label {side}")
 if not r.get("reviewer") or not r.get("review_timestamp") or not r.get("review_note"):raise ValueError(f"{cid}: reviewer, timestamp, note required")
def write_jsonl(p,data):
 if p.exists():raise FileExistsError(p)
 t=p.with_name(f".{p.name}.tmp")
 with t.open("x",encoding="utf-8") as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False)+"\n")
 t.replace(p)
def write_json(p,data):
 if p.exists():raise FileExistsError(p)
 t=p.with_name(f".{p.name}.tmp");t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");t.replace(p)
def write_csv(p,data):
 if p.exists():raise FileExistsError(p)
 fields=("candidate_id","object_a_semantic_purity_status","object_b_semantic_purity_status","non_target_inclusion_a","non_target_inclusion_b","contamination_labels_a","contamination_labels_b","reviewer","review_timestamp","review_note")
 t=p.with_name(f".{p.name}.tmp")
 with t.open("x",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in data:w.writerow({**{k:r[k] for k in fields if not k.startswith("contamination_labels")},"contamination_labels_a":json.dumps(r["contamination_labels_a"]),"contamination_labels_b":json.dumps(r["contamination_labels_b"])})
 t.replace(p)
def main():
 a=parse();template=rows(a.review_template);review=rows(a.review_input);tm={r["candidate_id"]:r for r in template};rm={r.get("candidate_id"):r for r in review}
 if len(tm)!=len(template) or len(rm)!=len(review) or set(tm)!=set(rm):raise ValueError("review must cover template IDs exactly")
 for cid in tm:validate(rm[cid],tm[cid],a.subset_name,a.asset_root)
 ordered=[rm[r["candidate_id"]] for r in template];a.output_dir.mkdir(parents=True,exist_ok=True)
 out_jsonl=a.output_dir/f"mask_semantic_purity_review_{a.subset_name}_v1.1.jsonl";out_csv=a.output_dir/f"mask_semantic_purity_review_{a.subset_name}_v1.1.csv";out_summary=a.output_dir/f"mask_semantic_purity_review_summary_{a.subset_name}_v1.1.json"
 write_jsonl(out_jsonl,ordered);write_csv(out_csv,ordered);write_json(out_summary,{"schema_version":"1.1","subset_name":a.subset_name,"candidate_count":len(ordered),"object_a_status_counts":dict(Counter(r["object_a_semantic_purity_status"] for r in ordered)),"object_b_status_counts":dict(Counter(r["object_b_semantic_purity_status"] for r in ordered)),"non_target_inclusion_candidate_count":sum(r["non_target_inclusion_a"] or r["non_target_inclusion_b"] for r in ordered),"automatic_pass_assignment":False,"review_template_sha256":sha(a.review_template),"review_input_sha256":sha(a.review_input)})
 print(json.dumps({"subset_name":a.subset_name,"candidate_count":len(ordered),"output":out_jsonl.as_posix()},indent=2))
if __name__=="__main__":main()
