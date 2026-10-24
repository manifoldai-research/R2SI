"""Paired held-out development evidence; mock/replay tests cannot promote models."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

from .storage import canonical
from .official_layouts import MODE, POLICY, EVIDENCE, validate_case, reject_open, uses_official_layouts


def compare(candidate_id, baseline_id, baseline, candidate, *,
            required_capabilities, required_seeds=(0, 1, 2), trained_cases=(),
            score_tolerance=0.0, sr_tolerance=0.0, action_ratio_limit=1.15):
    if not required_capabilities or len(set(required_seeds)) < 3:
        raise ValueError("Declare held-out capability coverage and at least three seeds")
    def index(rows, release):
        result = {}
        for row in rows:
            reject_open(row)
            official = uses_official_layouts(row)
            if official:
                validate_case(row)
            expected_kind = EVIDENCE if official else "development_closed_loop"
            expected_mode = MODE if official else "development"
            key = (row["case_id"], row["seed"])
            if key in result or row["case_id"] in trained_cases:
                raise ValueError("Duplicate evaluation case or train/validation overlap")
            if (row.get("release_id") != release or row.get("mode") != expected_mode
                    or row.get("evidence_kind") != expected_kind
                    or row.get("terminal_confirmed") is not True
                    or row.get("category") not in {"standard", "random"}):
                raise ValueError("Validation requires real terminal development runs")
            if row.get("infrastructure_error") or row.get("unsafe_action"):
                raise ValueError("Infrastructure/safety failure disqualifies a candidate comparison")
            if (type(row.get("success")) is not bool
                    or type(row.get("actions")) is not int or row["actions"] <= 0
                    or not isinstance(row.get("score"), (float, int))
                    or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 100
                    or not row.get("trace_sha256")):
                raise ValueError("Invalid native result or missing trace")
            result[key] = row
        return result
    b, c = index(baseline, baseline_id), index(candidate, candidate_id)
    if not b or b.keys() != c.keys():
        raise ValueError("Baseline/candidate must run the exact same held-out cases/seeds")
    kinds = {row["evidence_kind"] for row in [*b.values(), *c.values()]}
    if len(kinds) != 1:
        raise ValueError("Do not mix official-layout and development evidence")
    official = EVIDENCE in kinds
    coverage = defaultdict(set)
    for key in b:
        if official and any(b[key][f] != c[key][f] for f in ("task", "eval_seed", "layout_id", "layout_file", "layout_sha256")):
            raise ValueError("Paired official layouts differ in identity or bytes")
        if b[key]["capability"] != c[key]["capability"]:
            raise ValueError("Capability label changed between paired cases")
        coverage[b[key]["capability"]].add(key[1])
    if any(not set(required_seeds) <= coverage[x] for x in required_capabilities):
        raise ValueError("Missing capability/seed coverage")
    groups = {}
    failures = []
    for capability in sorted(coverage):
        keys = [k for k in b if b[k]["capability"] == capability]
        def metrics(rows):
            return {field: sum(float(rows[k][field]) for k in keys) / len(keys)
                    for field in ("score", "success", "actions")}
        before, after = metrics(b), metrics(c)
        passed = (after["score"] + score_tolerance >= before["score"]
                  and after["success"] + sr_tolerance >= before["success"]
                  and after["actions"] <= before["actions"] * action_ratio_limit)
        groups[capability] = {"n": len(keys), "baseline": before, "candidate": after, "passed": passed}
        if not passed:
            failures.append(capability)
    equal = all(g["candidate"] == g["baseline"] for g in groups.values())
    improved = (not failures and all(
        g["candidate"]["score"] >= g["baseline"]["score"]
        and g["candidate"]["success"] >= g["baseline"]["success"]
        and g["candidate"]["actions"] <= g["baseline"]["actions"] for g in groups.values())
        and not equal)
    return {"outcome_classification": ("rejected" if failures else "equal" if equal else
                                      "improved" if improved else "within_regression_budget"),
            "demonstrated_improvement": improved,
            "candidate": candidate_id, "baseline": baseline_id, "passed": not failures,
            "evidence_kind": EVIDENCE if official else "development_closed_loop", "groups": groups,
            "layout_policy": POLICY if official else "development",
            "training_eligible": False,
            "comparison_scope": "separate public layouts; not a hidden official score" if official else "development",
            "regressions": failures, "case_count": len(b),
            "trace_set_sha256": hashlib.sha256(canonical([baseline, candidate])).hexdigest(),
            "official_eligibility": "unverified"}
