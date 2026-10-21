#!/usr/bin/env python3
"""Summarize completed lookup ABBA runs and statically check their saved code."""
import argparse
import ast
import json
import re
import statistics
from pathlib import Path


def tree(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def check_output(probe, text, source):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    result = {"nonempty": bool(text.strip()), "error": text.startswith("ERROR ")}
    if probe == "prose":
        result["words"] = len(text.split())
        return result
    if len(re.findall(r"^\s*```", text, re.M)) % 2:
        result.update(syntax_ok=False, incomplete_code_fence=True)
        return result
    if not blocks:
        blocks = [text]
    try:
        parsed = [ast.parse(block) for block in blocks]
    except SyntaxError as exc:
        result.update(syntax_ok=False, syntax_error=str(exc))
        return result
    result["syntax_ok"] = True
    if probe == "edit-rename":
        expected = source.replace("GGUFFile", "GgufReader")
        result["requested_edit_only_ast"] = len(blocks) == 1 and tree(blocks[0]) == tree(expected)
        result["requested_edit_only_text"] = len(blocks) == 1 and blocks[0].strip() == expected.strip()
    elif probe == "edit-log":
        module = parsed[0]
        log_assign = tree("log = logging.getLogger(__name__)")
        kept = []
        imports = assignments = 0
        for node in module.body:
            if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "logging" and node.names[0].asname is None:
                imports += 1
            elif isinstance(node, ast.Assign) and ast.dump(ast.Module(body=[node], type_ignores=[]), include_attributes=False) == log_assign:
                assignments += 1
            else:
                kept.append(node)
        module.body = kept
        result["requested_edit_only_ast"] = len(blocks) == 1 and imports == assignments == 1 and ast.dump(module, include_attributes=False) == tree(source)
    else:
        result["qualification"] = "syntax only; generated tests not executed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    legs = []
    expected_probes = {"edit-rename", "edit-log", "fresh-code", "prose"}
    for number, arm in enumerate(("base", "candidate", "candidate", "base"), 1):
        directory = args.directory / f"{number}-{arm}"
        rows = json.loads((directory / "results.json").read_text())
        assert len(rows) == 4 and {r["probe"] for r in rows} == expected_probes
        for row in rows:
            assert row["gen"] > 0 and row["dec_ms"] > 0
            text = (directory / f"results.json.0-{row['probe']}.txt").read_text()
            row["output_checks"] = check_output(row["probe"], text, source)
            row["output_checks"]["reached_token_cap"] = row["gen"] >= 2600
            test_log = args.directory / "generated-tests" / directory.name / "pytest.log"
            if row["probe"] == "fresh-code" and test_log.exists():
                row["output_checks"]["generated_pytest_log"] = str(test_log)
                row["output_checks"]["generated_pytest_result"] = test_log.read_text().strip().splitlines()[-1]
                row["output_checks"]["qualification"] = "generated tests run separately after source inspection; not an independent correctness oracle"
            row["tokens_per_second"] = row["gen"] * 1000 / row["dec_ms"]
            row["acceptance"] = row["acc"] / row["off"] if row["off"] else None
        legs.append({"leg": number, "arm": arm, "rows": rows})
    summary = {}
    for probe in sorted(expected_probes):
        by_arm = {arm: [next(r for r in leg["rows"] if r["probe"] == probe)
                        for leg in legs if leg["arm"] == arm]
                  for arm in ("base", "candidate")}
        base, candidate = by_arm["base"], by_arm["candidate"]
        changes = [100 * (c["tokens_per_second"] / b["tokens_per_second"] - 1)
                   for b, c in zip(base, candidate)]
        summary[probe] = {
            "base_median_tok_s": statistics.median(r["tokens_per_second"] for r in base),
            "candidate_median_tok_s": statistics.median(r["tokens_per_second"] for r in candidate),
            "paired_percent_change": changes,
            "paired_percent_range": [min(changes), max(changes)],
            "base_hashes": [r["sha"] for r in base],
            "candidate_hashes": [r["sha"] for r in candidate],
            "output_checks": {arm: [r["output_checks"] for r in rows] for arm, rows in by_arm.items()},
        }
    result = {"summary": summary, "legs": legs,
              "limits": "Two observations per arm; syntax/AST checks do not establish model quality or numerical equivalence."}
    (args.directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
