#!/usr/bin/env python3
"""Audit current Round 1 artifacts against Section 9 file/report contracts.

Missing measurements stay missing. This script inventories readiness; it does
not synthesize table values or modify coordinator/measurement artifacts.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
import math
from datetime import datetime, timezone
from pathlib import Path

TABLES = {
    "conditions.csv": {"run_id","model","seed","condition","rendering","judge","lambda_p","lambda_d","steps","sample_count","status","raw_output_hash"},
    "behavior.csv": {"MR_numerator","MR_denominator","missing_parses","alignment","coherence","refusals","code","off_topic","length","absolute_effect","relative_effect","ci_low","ci_high"},
    "routes.csv": {"step","condition","region","S_C","S_X","S_G","S_HG","S_HC","Delta","T","D","M","removed_fraction","coverage","reverse_necessity","subset","identity_error"},
    "utility.csv": {"harmful_NLL","good_NLL","initial_NLL","E_NLL","U_NLL","U_preference","IFBench_strict","IFBench_loose","neutral_quality","source_count"},
    "factorial.csv": {"outcome","contrast","estimate","ci_low","ci_high","bootstrap_unit","bootstrap_count","bootstrap_seed","null_reason"},
    "training_diagnostics.csv": {"step","condition","harmful_loss","benign_loss","drift","learning_rate","grad_norm_inside","grad_norm_outside","update_norm_inside","update_norm_outside","clipping","wall_time","gpu_memory"},
    "generation_bridge.csv": {"condition","contrast","judge","estimate","ci_low","ci_high","sample_count","identity_check","prefix_check","donor_patch_norm"},
    "theory_checks.csv": {"endpoint","step","displacement_scale","observed_change","linear_prediction","absolute_residual","residual_over_delta_norm_sq","assumptions","null_reason"},
    "artifact_index.json": set(),
}
REPORT_SECTIONS = (
    "Decision summary", "Protocol and integrity", "Results", "Mechanism interpretation",
    "Theory", "Claim ledger", "Reproduction",
)
FIGURES = {
    "stage_evidence_decomposition": "route_decomposition_schematic",
    "cross_model_mechanism": "cross_model_route_decomposition",
    "first_seed_behavior_utility": "first_seed_behavior_utility",
    "absolute_route_factorial": "absolute_route_factorial",
    "training_trajectories": "training_trajectories",
}
TABLE_LOCATIONS = {
    "theory_checks.csv": (Path("theory/theory_checks.csv"), Path("theory_checks.csv")),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
PROMPT_COLUMNS = {"run_id","model","seed","condition","rendering","prompt_id",
                  "cluster_id","sample_id","judge","outcome","value","validity","source_hash"}


def _csv_header(path: Path) -> set[str]:
    with path.open(newline="") as f:
        return set(next(csv.reader(f), []))


def audit(run_dir: Path, report_path: Path, figure_dir: Path) -> dict:
    tables = {}
    for name, required in TABLES.items():
        candidates = [run_dir / relative for relative in TABLE_LOCATIONS.get(name, (Path(name),))]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        entry = {"path":str(path),"exists":path.is_file(),
                 "status":"present" if path.is_file() else "missing"}
        if path.is_file():
            entry["sha256"] = sha256(path)
        if path.is_file() and name.endswith(".csv"):
            fields = _csv_header(path)
            entry.update({"columns":sorted(fields),"missing_columns":sorted(required-fields),
                          "schema_status":"pass" if required <= fields else "missing_columns"})
            if name == "theory_checks.csv":
                with path.open(newline="") as f:
                    rows=list(csv.DictReader(f))
                measured_fields=("observed_change","linear_prediction","absolute_residual")
                complete=[]
                for row in rows:
                    numeric=True
                    for field in measured_fields:
                        try:
                            numeric = numeric and math.isfinite(float(row[field]))
                        except (KeyError, TypeError, ValueError):
                            numeric=False
                    complete.append(numeric and not str(row.get("status", "")).startswith("pending"))
                complete_count=sum(complete)
                entry.update({
                    "measurement_status":("ready" if rows and complete_count==len(rows) else
                                          "partial" if complete_count else "pending"),
                    "measurement_rows_total":len(rows),
                    "measurement_rows_complete":complete_count,
                    "measurement_rows_pending":len(rows)-complete_count,
                    "row_status_counts":{status:sum(row.get("status")==status for row in rows)
                                          for status in sorted({row.get("status", "") for row in rows})},
                })
        elif path.is_file() and name == "artifact_index.json":
            try:
                value=json.loads(path.read_text())
                required_keys={"files","execution_commands","plotting_commands"}
                entry.update({"keys":sorted(value),"missing_keys":sorted(required_keys-set(value)),
                              "schema_status":"pass" if required_keys <= set(value) else "missing_keys"})
            except Exception as exc:
                entry.update({"schema_status":"invalid_json","error":str(exc)})
        else:
            entry["schema_status"]="missing_file"
        tables[name]=entry

    report={"path":str(report_path),"exists":report_path.is_file(),"missing_sections":list(REPORT_SECTIONS),
            "status":"present" if report_path.is_file() else "missing"}
    if report_path.is_file():
        report["sha256"] = sha256(report_path)
        text=report_path.read_text()
        present=[s for s in REPORT_SECTIONS if s.lower() in text.lower()]
        report["present_sections_in_order"]=present
        report["missing_sections"]=[s for s in REPORT_SECTIONS if s not in present]
        report["section_order_status"]="pass" if present == list(REPORT_SECTIONS) else "missing_or_out_of_order"

    figures={}
    for name, stem in FIGURES.items():
        files={ext:(figure_dir/f"{stem}.{ext}").is_file() for ext in ("png","pdf")}
        paths={ext:str(figure_dir/f"{stem}.{ext}") for ext in files}
        hashes={ext:sha256(figure_dir/f"{stem}.{ext}") for ext, present in files.items() if present}
        figures[name]={"stem":stem,"paths":paths,"files":files,"sha256":hashes,
                       "status":"present" if all(files.values()) else "missing_png_or_pdf"}
    human_path=run_dir/"measurement"/"human_review_status.json"
    if not human_path.is_file() and (run_dir/"human_review_status.json").is_file():
        human_path=run_dir/"human_review_status.json"
    human={"path":str(human_path),"exists":human_path.is_file(),
           "status":"present" if human_path.is_file() else "missing"}
    if human_path.is_file():
        doc=json.loads(human_path.read_text())
        human["status"]="present"
        human["sha256"] = sha256(human_path)
        human["schema_fields"]={k:(k in doc) for k in ("eligible_rows","reviewed_rows","disagreement_counts_available","pending_labels","provenance","sampling_design")}
        human["numeric_disagreement_count_present"]="disagreement_counts" in doc
        human["review_status"]="pending" if doc.get("pending_labels") else "complete_or_unknown"
        human["note"]="Plan asks for disagreement counts; this file currently has availability boolean but no numeric disagreement count." if not human["numeric_disagreement_count_present"] else ""
    audit_source=Path(__file__).resolve()
    present_tables=sum(entry["exists"] for entry in tables.values())
    schema_valid_tables=sum(entry.get("schema_status")=="pass" for entry in tables.values())
    content_ready_tables=sum(entry.get("measurement_status")=="ready" for entry in tables.values())
    present_figures=sum(entry["status"]=="present" for entry in figures.values())
    return {"readiness_version":"section9_readiness_v3",
            "status":"audit_only","observed_at_utc":datetime.now(timezone.utc).isoformat(),
            "audit_source":{"path":str(audit_source),"sha256":sha256(audit_source)},
            "tables":tables,"report":report,"figures":figures,
            "counts":{"table_files_present":present_tables,"table_files_missing":len(tables)-present_tables,
                      "table_schemas_valid":schema_valid_tables,"tables_content_ready":content_ready_tables,
                      "figures_present":present_figures,"figures_missing":len(figures)-present_figures},
            "human_review_status":human,
            "prompt_level_schema_required":sorted(PROMPT_COLUMNS),
            "prompt_level_schema_check":"Apply to prompt-level tables when generated; raw and prompt-level data are not replaced by aggregates.",
            "null_policy":"Missing values stay null with a reason; no measurement values are synthesized."}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--figure-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    result=audit(a.run_dir.resolve(),a.report.resolve(),a.figure_dir.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"status":result["status"],
                      "table_files_present":sum(v["exists"] for v in result["tables"].values()),
                      "missing_figures":sum(v["status"]!="present" for v in result["figures"].values()),
                      "report_exists":result["report"]["exists"]},indent=2))


if __name__=="__main__": main()
