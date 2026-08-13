#!/usr/bin/env python3
"""Validate reviewed mask semantic-purity QC and enhanced review assets."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
STATUSES={"pass","fail","human_review"};LABELS={"skin","arm","hand","neck","face","hair","person_leg","pants","shoe","adjacent_object","background_region","other"}
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mask-results",type=Path,required=True);p.add_argument("--review-template",type=Path,required=True);p.add_argument("--review-manifest",type=Path,required=True);p.add_argument("--asset-root",type=Path,required=True);p.add_argument("--subset-name",required=True);return p.parse_args()
def rows(p):
 with p.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 a=parse();errors=[];source_masks=rows(a.mask_results);masks=[r for r in source_masks if r.get("mask_generation_status")=="generated"];failed_masks=[r for r in source_masks if r.get("mask_generation_status")=="segmentation_failed"];templates=rows(a.review_template);reviews=rows(a.review_manifest)
 if len(masks)+len(failed_masks)!=len(source_masks):errors.append("mask results contain unknown generation status")
 def index(label,data):
  d={r.get("candidate_id"):r for r in data}
  if not data or len(d)!=len(data) or None in d:errors.append(f"{label} IDs missing/duplicated")
  return d
 mi,ti,ri=index("mask",masks),index("template",templates),index("review",reviews)
 if set(mi)!=set(ti) or set(mi)!=set(ri):errors.append("mask/template/review candidate sets differ")
 for cid in sorted(set(mi)&set(ti)&set(ri)):
  m,t,r=mi[cid],ti[cid],ri[cid]
  if m.get("pilot_subset")!=a.subset_name or r.get("subset_name")!=a.subset_name:errors.append(f"{cid}: subset mismatch")
  expected={"object_a_label":m["object_a"]["label"],"object_b_label":m["object_b"]["label"],"original_color_a":m["object_a"]["original_color"],"original_color_b":m["object_b"]["original_color"],"contact_sheet_path":t.get("contact_sheet_path"),"contact_sheet_sha256":t.get("contact_sheet_sha256")}
  for f,v in expected.items():
   if r.get(f)!=v:errors.append(f"{cid}: {f} mismatch")
  p=a.asset_root/r.get("contact_sheet_path","")
  if not p.is_file() or sha(p)!=r.get("contact_sheet_sha256"):errors.append(f"{cid}: contact sheet invalid")
  for side in ("a","b"):
   status=r.get(f"object_{side}_semantic_purity_status");inc=r.get(f"non_target_inclusion_{side}");labels=r.get(f"contamination_labels_{side}")
   if status not in STATUSES:errors.append(f"{cid}: invalid status {side}")
   if not isinstance(inc,bool) or not isinstance(labels,list) or len(labels)!=len(set(labels)) or not set(labels).issubset(LABELS):errors.append(f"{cid}: invalid contamination {side}")
   if status=="pass" and (inc or labels):errors.append(f"{cid}: pass contamination conflict {side}")
   if status=="fail" and (not inc or not labels):errors.append(f"{cid}: fail lacks contamination {side}")
  if not r.get("reviewer") or not r.get("review_timestamp") or not r.get("review_note"):errors.append(f"{cid}: review provenance incomplete")
 report={"subset_name":a.subset_name,"source_candidate_count":len(source_masks),"segmentation_success_count":len(masks),"segmentation_failure_count":len(failed_masks),"candidate_count":len(reviews),"object_a_status_counts":dict(Counter(r.get("object_a_semantic_purity_status") for r in reviews)),"object_b_status_counts":dict(Counter(r.get("object_b_semantic_purity_status") for r in reviews)),"contact_sheets_verified":len(reviews),"validation_errors":len(errors),"errors":errors}
 print(json.dumps(report,ensure_ascii=False,indent=2))
 if errors:sys.exit(1)
if __name__=="__main__":main()
