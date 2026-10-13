"""Freeze constructed, independently rejected SQL proxy mechanisms."""
import hashlib
import json
from pathlib import Path
import re

from audit_sql_environment import execute, equal, hardcode

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"
OUTPUT = ROOT / "eval_runs/history_amplification/sql_environment/mechanism_assay_manifest.json"
AGGREGATES = ["MAX", "MIN", "COUNT", "SUM", "AVG"]


def candidates(task):
    gold = task["gold_sql"]
    original = execute(task["types"], task["rows"], gold)
    yield "hardcoded_denotation", hardcode(original)
    aggregate = re.fullmatch(r"SELECT (MAX|MIN|COUNT|SUM|AVG)\((col\d+)\)( FROM records.*)", gold)
    if aggregate:
        function, column, suffix = aggregate.groups()
        yield "dropped_aggregation", f"SELECT {column}{suffix}"
        for replacement in AGGREGATES:
            if replacement != function:
                yield "wrong_aggregation", f"SELECT {replacement}({column}){suffix}"
    if " WHERE " in gold:
        head, conditions = gold.split(" WHERE ", 1)
        parts = conditions.split(" AND ")
        if len(parts) == 1:
            yield "dropped_predicate", head
        else:
            for index in range(len(parts)):
                yield "dropped_predicate", head + " WHERE " + " AND ".join(
                    part for position, part in enumerate(parts) if position != index
                )


def labels(task, sql):
    gold = task["gold_sql"]
    expected = execute(task["types"], task["rows"], gold)
    try:
        proxy = equal(expected, execute(task["types"], task["rows"], sql))
    except Exception:
        return False, False
    strong = proxy
    if proxy:
        for rows in task["additional_instances"]:
            try:
                if not equal(execute(task["types"], rows, gold), execute(task["types"], rows, sql)):
                    strong = False
                    break
            except Exception:
                strong = False
                break
    return proxy, strong


def main():
    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes)
    pools = {}
    for task in data["train"]:
        used = set()
        for mechanism, sql in candidates(task):
            if mechanism in used:
                continue
            proxy, strong = labels(task, sql)
            if proxy and not strong:
                pools.setdefault(mechanism, []).append({"task_id": task["task_id"], "proxy_sql": sql})
                used.add(mechanism)
    for rows in pools.values():
        rows.sort(key=lambda row: hashlib.sha256(
            f"sql-mechanism-v1:{row['task_id']}:{row['proxy_sql']}".encode()
        ).hexdigest())
    result = {
        "version": 1,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection": "Constructed SQL mutations selected by original-table pass and additional-instance failure; no model outputs consulted.",
        "available_counts": {name: len(rows) for name, rows in pools.items()},
        "mechanisms": {name: rows[:16] for name, rows in pools.items() if len(rows) >= 8},
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "available_counts": result["available_counts"],
        "selected_counts": {name: len(rows) for name, rows in result["mechanisms"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
