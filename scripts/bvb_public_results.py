#!/usr/bin/env python3
"""Check the BankerVerifierBench paper results, offline.

`check` uses the standard library only and makes no network or model calls. It
recomputes every statistic in the results bundle from the bundle's own
criterion predictions and fails on any difference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results/paper"
SCHEMA = "bvb-paper-results/v1"
KEYS = ("tp", "fp", "fn", "tn")

OPC_PREDICTIONS = "OpenClaw verdicts: Gandalf, Archipelago, AAAJ and AutoRubric on Gemini 3 Flash, three passes"

CONDITION_BOOTSTRAP = dict(seed=20261006, draws=2000)
REPEAT1_BOOTSTRAP = dict(seed=20261007, draws=10000)
BATCHING_BOOTSTRAP = dict(seed=20261007, draws=2000)
CAPABILITY_BOOTSTRAP = dict(seed=20261102, draws=2000)
REPEAT1_PAIRS = [("M04", "M01"), ("M04", "M05"), ("M03", "M01"), ("M04", "M10")]
BATCHING_REFERENCE = "b16"

# Capability analysis under the paper's nine capabilities.
CORE = ["M01", "M13", "M16", "M19"]
APPENDIX = CORE + ["M09"]
CAPABILITIES = [
    "audit_quantitative_metric_calculation_capability",
    "verify_literal_data_point_accuracy_capability",
    "reconcile_cross_component_data_linkage_capability",
    "verify_structural_component_inventory_capability",
    "inspect_visual_presentation_and_style_capability",
    "audit_accounting_logic_and_convention_capability",
    "validate_document_labeling_and_metadata_capability",
    "audit_governance_and_strategic_narrative_capability",
    "validate_temporal_and_sequential_integrity_capability",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read(path: Path) -> str:
    return path.read_bytes().decode()


def write(path: Path, text: str) -> None:
    path.write_bytes(text.encode())


# Measurement metrics ------------------------------------------------------------


def total(rows) -> Counter:
    counts = Counter({k: 0 for k in KEYS})
    for row in rows:
        for k in KEYS:
            counts[k] += row[k]
    return counts


def summary_metrics(cc) -> dict:
    """Condition summary metrics."""
    tp, fp, fn, tn = (cc[k] for k in KEYS)
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return dict(counts={k: cc[k] for k in KEYS},
                precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                accuracy=(tp + tn) / sum(cc[k] for k in KEYS),
                mcc=(tp * tn - fp * fn) / math.sqrt(den) if den else 0)


def scoring_metrics(cc, f1_key="f1_unmet") -> dict:
    """`verifier_runner.scoring.metrics_from_counts`."""
    tp, fp, fn, tn = (cc[k] for k in KEYS)
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    out = dict(counts={k: cc[k] for k in KEYS}, precision=precision, recall=recall, f1=f1,
               accuracy=(tp + tn) / n if n else 0.0,
               mcc=(tp * tn - fp * fn) / math.sqrt(den) if den else 0.0,
               unmet_rate=(tp + fp) / n if n else 0.0)
    if f1_key != "f1":
        out[f1_key] = out.pop("f1")
    return out


def sensitivity_metrics(cc) -> dict:
    tp, fp, fn, tn = (cc[k] for k in KEYS)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(counts={k: cc[k] for k in KEYS}, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                accuracy=(tp + tn) / sum(cc[k] for k in KEYS))


def bootstrap_interval(draws, count):
    draws.sort()
    return [draws[count // 40 - 1], draws[count - count // 40 - 1]]


def derive(data: dict) -> dict:
    """Every derived statistic of a results bundle, from its runs and predictions."""
    tasks = sorted({row[0] for row in data["rubric"]})
    runs = defaultdict(list)
    for run in data["runs"]:
        runs[run["condition"]].append(run)
    slots = {cid: {(r["repeat"], r["task"]): r for r in rows} for cid, rows in runs.items()}
    votes = defaultdict(list)
    for cid, task, index, human, *predictions in data["criteria"]:
        votes[cid].append((human, predictions))

    def majority(cid):
        cc, unanimous, n = Counter({k: 0 for k in KEYS}), 0, 0
        for human, predictions in votes[cid]:
            require(all(p is not None for p in predictions), f"{cid}: incomplete repeats")
            vote = Counter(predictions).most_common(1)[0][0]
            cc[_direction(human, vote)] += 1
            unanimous += len(set(predictions)) == 1
            n += 1
        return scoring_metrics(cc)["f1_unmet"], unanimous / n

    conditions = []
    for meta in data["conditions"]:
        cid, rs = meta["id"], runs[meta["id"]]
        timed = [r for r in rs if r["primary_latency_eligible"] and r["duration_sec"] is not None]
        entry = {k: meta[k] for k in ("id", "verifier", "model", "study", "scope", "target", "effort",
                                      "partition", "status")}
        entry.update(runs=len(rs), tasks=len({r["task"] for r in rs}),
                     repeat_ids=",".join(map(str, sorted({r["repeat"] for r in rs}))),
                     **summary_metrics(total(rs)), total_cost=sum(r["cost_usd"] or 0 for r in rs),
                     mean_seconds=statistics.mean(r["duration_sec"] for r in timed),
                     median_seconds=statistics.median(r["duration_sec"] for r in timed),
                     timing_runs=len(timed), overall_mae=statistics.mean(r["abs_error"] for r in rs),
                     overall_bias=statistics.mean(r["difference"] for r in rs), per_repeat=[])
        for repeat in sorted({r["repeat"] for r in rs}):
            selected = [r for r in rs if r["repeat"] == repeat]
            entry["per_repeat"].append(dict(repeat=repeat, runs=len(selected), **summary_metrics(total(selected)),
                                            cost=sum(r["cost_usd"] or 0 for r in selected)))
        full_cohort = meta["study"] == "Model comparison" and meta["scope"] == "Selected" and len(rs) == 3 * len(tasks)
        entry.update(majority_f1=None, unanimity=None, ci_low=None, ci_high=None)
        if full_cohort or meta["study"] == "Batching":
            entry["majority_f1"], entry["unanimity"] = majority(cid)
        if full_cohort:
            by_task = {t: total(slots[cid][rep, t] for rep in (1, 2, 3)) for t in tasks}
            rng, draws = random.Random(CONDITION_BOOTSTRAP["seed"]), []
            for _ in range(CONDITION_BOOTSTRAP["draws"]):
                draws.append(scoring_metrics(total(by_task[t] for t in rng.choices(tasks, k=len(tasks))))["f1_unmet"])
            entry["ci_low"], entry["ci_high"] = bootstrap_interval(draws, CONDITION_BOOTSTRAP["draws"])
        conditions.append(entry)
    by_id = {c["id"]: c for c in conditions}

    paired = []
    for left, right in REPEAT1_PAIRS:
        a = {t: total([slots[left][1, t]]) for t in tasks}
        b = {t: total([slots[right][1, t]]) for t in tasks}
        rng, draws = random.Random(REPEAT1_BOOTSTRAP["seed"]), []
        for _ in range(REPEAT1_BOOTSTRAP["draws"]):
            chosen = rng.choices(tasks, k=len(tasks))
            draws.append(scoring_metrics(total(a[t] for t in chosen))["f1_unmet"] -
                         scoring_metrics(total(b[t] for t in chosen))["f1_unmet"])
        delta = scoring_metrics(total(a.values()))["f1_unmet"] - scoring_metrics(total(b.values()))["f1_unmet"]
        paired.append(["Model repeat1", _label(by_id[left]), _label(by_id[right]), delta,
                       *bootstrap_interval(draws, REPEAT1_BOOTSTRAP["draws"]), len(tasks), 1,
                       REPEAT1_BOOTSTRAP["draws"], "Exploratory fixed-repeat task interval; no multiplicity correction"])
    batching = {c["partition"]: c["id"] for c in conditions if c["study"] == "Batching"}
    reference = batching[BATCHING_REFERENCE]
    rng = random.Random(BATCHING_BOOTSTRAP["seed"])
    for partition in sorted(batching):
        cid = batching[partition]
        if partition == BATCHING_REFERENCE or len(runs[cid]) != 3 * len(tasks):
            continue
        groups = {t: (total(slots[cid][rep, t] for rep in (1, 2, 3)),
                      total(slots[reference][rep, t] for rep in (1, 2, 3))) for t in tasks}
        draws = []
        for _ in range(BATCHING_BOOTSTRAP["draws"]):
            chosen = rng.choices(tasks, k=len(tasks))
            draws.append(scoring_metrics(total(groups[t][0] for t in chosen))["f1_unmet"] -
                         scoring_metrics(total(groups[t][1] for t in chosen))["f1_unmet"])
        delta = scoring_metrics(total(runs[cid]))["f1_unmet"] - scoring_metrics(total(runs[reference]))["f1_unmet"]
        paired.append(["Batching", partition, BATCHING_REFERENCE, delta,
                       *bootstrap_interval(draws, BATCHING_BOOTSTRAP["draws"]), len(tasks), 3,
                       BATCHING_BOOTSTRAP["draws"], "Paired task clusters; all three repetitions retained"])

    human = Counter({"tp": sum(r[5] == "UNMET" for r in data["rubric"]), "fp": 0, "fn": 0,
                     "tn": sum(r[5] == "MET" for r in data["rubric"])})
    weights, met = defaultdict(float), defaultdict(float)
    for task, _, _, _, weight, label, *_ in data["rubric"]:
        weights[task] += weight
        met[task] += weight * (label == "MET")
    human_overall = [met[t] / weights[t] for t in tasks]
    baselines = {}
    for name, cc in (("always-unmet", Counter(tp=human["tp"], fp=human["tn"], fn=0, tn=0)),
                     ("always-met", Counter(tp=0, fp=0, fn=human["tp"], tn=human["tn"]))):
        baseline = scoring_metrics(cc, f1_key="f1")
        predicts_met = name == "always-met"
        baseline.update(overall_mae=statistics.mean(abs(float(predicts_met) - h) for h in human_overall),
                        mean_seconds=0, model_cost_usd=0)
        baselines[name] = baseline

    return dict(conditions=conditions, paired=paired, baselines=baselines,
                batch_sensitivity=_batch_sensitivity(data, tasks, batching, slots))


def _label(condition):
    return f"{condition['verifier']} {condition['model']}"


def _direction(human, predicted):
    require(human in {"MET", "UNMET"} and predicted in {"MET", "UNMET"}, "Invalid class")
    if human == "UNMET":
        return "tp" if predicted == "UNMET" else "fn"
    return "fp" if predicted == "UNMET" else "tn"


def _batch_sensitivity(data, tasks, batching, slots):
    recorded = data["batch_sensitivity"]
    sessions = defaultdict(lambda: [0, 0])
    for s in data["sessions"]:
        key = (s["condition"], s["repeat"], s["task"])
        sessions[key][0] += 1
        sessions[key][1] += s["initial"] is True
    out = dict(risk_records=recorded["risk_records"], sensitivities={})
    for name, recorded_sensitivity in recorded["sensitivities"].items():
        removed_slots = {tuple(x) for x in recorded_sensitivity.get("excluded_task_repeats", [])}
        removed_tasks = set(recorded_sensitivity.get("excluded_tasks", []))
        result = {k: v for k, v in recorded_sensitivity.items() if k not in ("conditions", "b16_deltas")}
        result["conditions"] = {}
        for partition in recorded_sensitivity["conditions"]:
            kept = [r for (rep, t), r in slots[batching[partition]].items()
                    if (t, rep) not in removed_slots and t not in removed_tasks]
            result["conditions"][partition] = dict(slots=len(kept), **sensitivity_metrics(total(kept)))
        result["b16_deltas"] = {p: result["conditions"][p]["f1"] - result["conditions"][BATCHING_REFERENCE]["f1"]
                                for p in recorded_sensitivity["b16_deltas"]}
        out["sensitivities"][name] = result
    retry = []
    reference = slots[batching[BATCHING_REFERENCE]]

    def no_retry(partition, slot):
        total_sessions, initial = sessions[partition, slot[0], slot[1]]
        return total_sessions == initial

    for recorded_row in recorded["native_retry_sensitivity"]:
        partition = recorded_row["condition"]
        rows = slots[batching[partition]]
        without = {slot: r for slot, r in rows.items() if no_retry(partition, slot)}
        row = dict(condition=partition, complete_condition=len(rows) == 3 * len(tasks), accepted=len(rows),
                   native_retry_runs=len(rows) - len(without), no_native_retry_slots=len(without),
                   task_coverage=len({t for _, t in without}))
        if without:
            row["pooled"] = scoring_metrics(total(without.values()))
        if partition != BATCHING_REFERENCE and len(rows) == 3 * len(tasks) and len(reference) == 3 * len(tasks):
            matched = {slot: r for slot, r in without.items() if no_retry(BATCHING_REFERENCE, slot)}
            row["matched_b16_comparison"] = dict(
                slots=len(matched), task_coverage=len({t for _, t in matched}),
                condition_metrics=scoring_metrics(total(matched.values())),
                reference_metrics=scoring_metrics(total(reference[slot] for slot in matched)))
        retry.append(row)
    out["native_retry_sensitivity"] = retry
    return out


# Capability analysis ------------------------------------------------------------


def _capability_label(name):
    return name.removesuffix("_capability").replace("_", " ").capitalize()


def _error_metrics(cc):
    tp, fp, fn, tn = (cc[k] for k in KEYS)
    n = tp + fp + fn + tn

    def div(a, b):
        return a / b if b else None
    return dict(n_judgments=n, tp=tp, fp=fp, fn=fn, tn=tn, n_errors=fp + fn,
                error_rate=div(fp + fn, n), over_rejection_rate_all=div(fp, n), under_rejection_rate_all=div(fn, n),
                over_rejection_rate_given_met=div(fp, fp + tn), under_rejection_rate_given_unmet=div(fn, tp + fn),
                f1_unmet=div(2 * tp, 2 * tp + fp + fn))


def _best(row, verifiers):
    m = min(row[v] for v in verifiers)
    return [v for v in verifiers if abs(row[v] - m) < 1e-9]


def _tex_row(name, n, values, bold):
    cells = [f"\\textbf{{{v:.1f}}}" if b else f"{v:.1f}" for v, b in zip(values, bold)]
    label = _capability_label(name) if not name.startswith("__") else "Aggregate"
    return f"{label} & {n:,} & ".replace(",", "{,}") + " & ".join(cells) + r" \\"


def _csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _read_csv_text(text):
    return list(csv.DictReader(io.StringIO(text)))


def _join(names, tags, data):
    rubric = {}
    for task, index, _, _, _, human, text, digest in data["rubric"]:
        require(sha(text.encode()) == digest, "Canonical criterion hash")
        rubric[task, index] = dict(human=human, caps=tags[task, str(index)], text=text)
    conditions = {c["id"]: c for c in data["conditions"]}
    cc, task_cc, multi_cc, seen = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), set()
    for cid, task, index, human, *predictions in data["criteria"]:
        r = rubric[task, index]
        require(human == r["human"], "Prediction identity")
        for repeat, predicted in enumerate(predictions, 1):
            if predicted is None:
                continue
            require((cid, task, index, repeat) not in seen, "Duplicate prediction")
            seen.add((cid, task, index, repeat))
            d = _direction(human, predicted)
            cc[cid, r["caps"][0]][d] += 1
            task_cc[cid, r["caps"][0], task][d] += 1
            for cap in r["caps"]:
                multi_cc[cid, cap][d] += 1
    for cid, c in conditions.items():
        counts = Counter()
        for cap in names:
            counts.update(cc[cid, cap])
        require({k: counts[k] for k in KEYS} == c["counts"], f"{cid}: capability confusion does not reconcile")
    primary = Counter(r["caps"][0] for r in rubric.values())
    return rubric, conditions, cc, task_cc, multi_cc, primary, len(seen)


def _error_table(names, conditions, cc, primary, cids):
    rows = []
    for cap in names:
        row = dict(capability=cap, display_name=_capability_label(cap), n_unique_criteria=primary[cap])
        for cid in cids:
            m = _error_metrics(cc[cid, cap])
            if cid != "M09" or m["n_judgments"]:
                require(m["n_judgments"] == 3 * primary[cap], f"{cid}/{cap}: judgment coverage")
            row[conditions[cid]["verifier"]] = 100 * m["error_rate"] if m["n_judgments"] else None
        row["best_flash_verifiers"] = ";".join(_best(row, [conditions[c]["verifier"] for c in CORE]))
        rows.append(row)
    agg = dict(capability="__aggregate__", display_name="Aggregate", n_unique_criteria=sum(primary.values()))
    for cid in cids:
        agg[conditions[cid]["verifier"]] = 100 * (1 - conditions[cid]["accuracy"])
    agg["best_flash_verifiers"] = ";".join(_best(agg, [conditions[c]["verifier"] for c in CORE]))
    return rows, agg


def _paired_capability(names, conditions, task_cc, tasks):
    rng = random.Random(CAPABILITY_BOOTSTRAP["seed"])
    draws = [Counter(rng.choices(tasks, k=len(tasks))) for _ in range(CAPABILITY_BOOTSTRAP["draws"])]
    out = []
    for cap in names:
        for cid in ["M13", "M16", "M19", "M09"]:
            deltas = []
            for weights in draws:
                a, b = Counter(), Counter()
                for task, k in weights.items():
                    for d, v in task_cc[cid, cap, task].items():
                        a[d] += k * v
                    for d, v in task_cc["M01", cap, task].items():
                        b[d] += k * v
                ma, mb = _error_metrics(a), _error_metrics(b)
                if ma["n_judgments"] and mb["n_judgments"]:
                    deltas.append(100 * (ma["error_rate"] - mb["error_rate"]))
            deltas.sort()
            pa = _error_metrics(sum((task_cc[cid, cap, t] for t in tasks), Counter()))
            pb = _error_metrics(sum((task_cc["M01", cap, t] for t in tasks), Counter()))
            lo, hi = deltas[int(.025 * (len(deltas) - 1))], deltas[int(.975 * (len(deltas) - 1))]
            out.append(dict(capability=cap, display_name=_capability_label(cap), comparator=cid,
                            verifier=conditions[cid]["verifier"], model=conditions[cid]["model"], reference="M01",
                            delta_error_pp=round(100 * (pa["error_rate"] - pb["error_rate"]), 2),
                            ci95_low_pp=round(lo, 2), ci95_high_pp=round(hi, 2), bootstrap_draws=len(deltas),
                            resolved=lo > 0 or hi < 0, gandalf_better_resolved=lo > 0,
                            matching_model=cid != "M09", seed=CAPABILITY_BOOTSTRAP["seed"], multiplicity_adjusted=False))
    return out


def _kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return (po - pe) / (1 - pe)


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _tags(text, key_cols):
    return {tuple(r[k] for k in key_cols): json.loads(r["capabilities_raw"]) for r in _read_csv_text(text)}


def capabilities(data, taxonomy, tags_text, replicate_text, opc_text, opc_paired_text) -> dict[str, str]:
    """Capability-conditioned results under the paper's nine capabilities, as file texts."""
    names = list(CAPABILITIES)
    require({c["name"] for c in taxonomy} == set(names), "Taxonomy does not match the paper's nine capabilities")
    tags = _tags(tags_text, ["task_id", "evaluation_criterion_index"])
    rubric, conditions, cc, task_cc, multi_cc, primary, n_predictions = _join(names, tags, data)
    require(len(tags) == len(rubric), "Tag coverage")
    n = len(rubric)
    files = {}

    dist = []
    for cap in names:
        dist.append(dict(capability=cap, display_name=_capability_label(cap), n_primary=primary[cap],
                         primary_share_pct=round(100 * primary[cap] / n, 1),
                         n_human_unmet=sum(r["human"] == "UNMET" for r in rubric.values() if r["caps"][0] == cap),
                         n_tasks=len({t for (t, _), r in rubric.items() if r["caps"][0] == cap}),
                         description=next(c["description"] for c in taxonomy if c["name"] == cap)))
    dist.sort(key=lambda r: -r["n_primary"])
    order = [r["capability"] for r in dist]
    files["table1_capability_distribution.csv"] = _csv_text(dist)

    rows, agg = _error_table(order, conditions, cc, primary, APPENDIX)
    files["bvb_primary_error_rates.csv"] = _csv_text(rows + [agg])
    pairs = _paired_capability(order, conditions, task_cc, sorted({t for t, _ in rubric}))
    files["paired_error_differences_vs_gandalf.csv"] = _csv_text(pairs)
    multi = []
    for cap in order:
        row = dict(capability=cap, display_name=_capability_label(cap),
                   n_any_tag=sum(cap in r["caps"] for r in rubric.values()))
        for cid in APPENDIX:
            row[conditions[cid]["verifier"]] = 100 * _error_metrics(multi_cc[cid, cap])["error_rate"]
        row["best_flash_verifiers"] = ";".join(_best(row, [conditions[c]["verifier"] for c in CORE]))
        multi.append(row)
    files["bvb_multi_tag_error_rates.csv"] = _csv_text(multi)

    by_text = _tags(replicate_text, ["task_id", "criterion"])
    replicate_tags = {(t, str(i)): by_text[t, r["text"]] for (t, i), r in rubric.items()}
    _, _, rcc, _, _, rprimary, _ = _join(names, replicate_tags, data)
    rep_rows, rep_agg = _error_table(order, conditions, rcc, rprimary, CORE)
    files["bvb_primary_error_rates_replicate_tags.csv"] = _csv_text(rep_rows + [rep_agg])

    arch = []
    for cap in order:
        m = _error_metrics(cc["M13", cap])
        arch.append(dict(capability=cap, display_name=_capability_label(cap), n_unique_criteria=primary[cap],
                         fp_pct_all=round(100 * m["over_rejection_rate_all"], 1),
                         fn_pct_all=round(100 * m["under_rejection_rate_all"], 1),
                         fn_minus_fp_pp=round(100 * (m["under_rejection_rate_all"] - m["over_rejection_rate_all"]), 1),
                         fp_pct_given_met=round(100 * m["over_rejection_rate_given_met"], 1),
                         fn_pct_given_unmet=round(100 * m["under_rejection_rate_given_unmet"], 1)))
    arch.sort(key=lambda r: -r["fn_minus_fp_pp"])
    files["archipelago_fp_fn.csv"] = _csv_text(arch)

    verifiers = ["gandalf", "archipelago", "aaaj", "autorubric", "rewardkit"]

    # OpenClaw (OPC) rows are a separate benchmark; only its aggregates are released.
    opc_rows = {}
    for r in _read_csv_text(opc_text):
        opc_rows[r["capability"]] = dict(r, n_unique_criteria=int(r["n_unique_criteria"]),
                                         **{v: float(r[v]) if r[v] else None for v in verifiers[:4]})
    opc_label = opc_rows["__aggregate__"]["predictions"]
    require(set(opc_rows) == set(names) | {"__aggregate__"} and
            {r["predictions"] for r in opc_rows.values()} == {OPC_PREDICTIONS}, "Unexpected OPC rates")
    four = verifiers[:4]
    opc_capabilities = [c for c in order if opc_rows[c]["n_unique_criteria"] >= 5]
    opc_pairs = _read_csv_text(opc_paired_text)
    require([(p["capability"], p["comparator"]) for p in opc_pairs] ==
            [(c, v) for c in opc_capabilities + ["__aggregate__"] for v in four[1:]], "OPC interval coverage")
    for p in opc_pairs:
        o = opc_rows[p["capability"]]
        lo, delta, hi = (float(p[k]) for k in ("ci95_low_pp", "delta_error_pp", "ci95_high_pp"))
        require(abs(delta - (o[p["comparator"]] - o["gandalf"])) <= 0.005 + 1e-9 and lo <= hi and
                p["resolved"] == str(lo > 0 or hi < 0) and p["gandalf_better_resolved"] == str(lo > 0),
                f"OPC interval does not match the OPC rates: {p['capability']}/{p['comparator']}")
    lines = []
    for r in rows + [agg]:
        if r["capability"] == "__aggregate__":
            lines.append(r"\midrule")
        bold = r["best_flash_verifiers"].split(";")
        o = opc_rows[r["capability"]]
        bvb_part = _tex_row(r["capability"], r["n_unique_criteria"], [r[x] for x in four], [x in bold for x in four])
        if o["n_unique_criteria"] >= 5 or r["capability"] == "__aggregate__":
            ob = _best(o, four)
            opc_part = f" & {o['n_unique_criteria']} & " + " & ".join(
                (f"\\textbf{{{o[x]:.1f}}}" if x in ob else f"{o[x]:.1f}") for x in four)
        else:
            opc_part = " & - & - & - & - & -"
        lines.append(bvb_part[:-3] + opc_part + r" \\")
    files["tab_combined_rows.tex"] = (
        "% Generated by scripts/bvb_public_results.py; OPC columns from " + opc_label
        + '\n% OPC rows with n<5 shown as "-".\n' + "\n".join(lines) + "\n")
    lines = []
    for r in rows + [agg]:
        if r["capability"] == "__aggregate__":
            lines.append(r"\midrule")
        bold = r["best_flash_verifiers"].split(";")
        lines.append(_tex_row(r["capability"], r["n_unique_criteria"], [r[x] for x in verifiers],
                              [x in bold for x in verifiers]))
    files["tab_combined_appendix_bvb_rows.tex"] = (
        "% BVB columns only; RK = RewardKit GPT-5.4 (M09), not model-matched; bold among the four Flash verifiers.\n"
        + "\n".join(lines) + "\n")
    files["tab_taxonomy_rows.tex"] = "\n".join(
        f"{r['display_name']} & {r['description']} & {r['primary_share_pct']:.1f} \\\\" for r in dist) + "\n"
    files["archi_bimodal_rows.tex"] = "\n".join(
        f"{r['display_name']} & {r['n_unique_criteria']} & {r['fp_pct_all']:.1f} & {r['fn_pct_all']:.1f} & "
        f"${r['fn_minus_fp_pp']:+.1f}$ \\\\" for r in arch) + "\n"

    wins = [r for r in rows if "gandalf" in r["best_flash_verifiers"].split(";")]
    resolved = [cap for cap in order if all(p["gandalf_better_resolved"] for p in pairs
                                            if p["capability"] == cap and p["comparator"] in ["M13", "M16", "M19"])]
    run1 = {key: tags[key] for key in replicate_tags}
    agreement_pairs = [(run1[k][0], replicate_tags[k][0]) for k in run1]
    agree = sum(a == b for a, b in agreement_pairs)
    lo, hi = _wilson(agree, n)
    summary = dict(
        n_criteria=n, n_tasks=len({t for t, _ in rubric}), n_predictions=n_predictions,
        confusion_counts_reconciled=True,
        primary_counts={_capability_label(c): primary[c] for c in order},
        gandalf_lowest_point_estimate=f"{len(wins)} of 9",
        gandalf_lowest_capabilities=[r["display_name"] for r in wins],
        gandalf_not_lowest={r["display_name"]: r["best_flash_verifiers"] for r in rows if r not in wins},
        gandalf_resolved_better_than_all_three_flash_baselines=[_capability_label(c) for c in resolved],
        gandalf_vs_archipelago_resolved=[f"{p['display_name']} ({p['delta_error_pp']:+.1f} pp)"
                                         for p in pairs if p["comparator"] == "M13" and p["resolved"]],
        multi_tag_best_same_as_primary=sum(m["best_flash_verifiers"] == r["best_flash_verifiers"]
                                           for m, r in zip(multi, rows)),
        archipelago_fn_dominant=[r["display_name"] for r in arch if r["fn_minus_fp_pp"] > 0],
        opc_predictions=opc_label,
        opc_primary_counts={_capability_label(c): opc_rows[c]["n_unique_criteria"]
                            for c in names if opc_rows[c]["n_unique_criteria"]},
        opc_n_tasks=int(next(p["n_tasks_with_capability"] for p in opc_pairs if p["capability"] == "__aggregate__")),
        opc_gandalf_lowest=f"{sum('gandalf' in _best(opc_rows[c], four) for c in opc_capabilities)} of "
                           f"{len(opc_capabilities)} capabilities with n>=5",
        opc_gandalf_not_lowest={_capability_label(c): ";".join(_best(opc_rows[c], four))
                                for c in opc_capabilities if "gandalf" not in _best(opc_rows[c], four)},
        opc_gandalf_resolved_better_than_all_three=[
            _capability_label(c) for c in opc_capabilities
            if all(p["gandalf_better_resolved"] == "True" for p in opc_pairs if p["capability"] == c)],
        seed=CAPABILITY_BOOTSTRAP["seed"], bootstrap_draws=CAPABILITY_BOOTSTRAP["draws"],
        replicate_tags_gandalf_lowest=f"{sum('gandalf' in r['best_flash_verifiers'].split(';') for r in rep_rows)} of 9",
        replicate_tags_max_abs_error_shift_pp=round(max(abs(a[x] - b[x]) for a, b in zip(rows, rep_rows) for x in four), 1),
        replicate_primary_agreement=round(agree / n, 3),
        replicate_primary_agreement_ci95=[round(lo, 3), round(hi, 3)],
        replicate_primary_kappa=round(_kappa(agreement_pairs), 3),
        replicate_primary_in_any_tag=round(sum(run1[k][0] in replicate_tags[k] for k in run1) / n, 3),
    )
    files["summary.json"] = json.dumps(summary, indent=2) + "\n"
    return files


# Public bundle check -----------------------------------------------------------


def _close(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        require(a.keys() == b.keys(), f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        require(len(a) == len(b), f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        require(isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), f"{path}: {a} != {b}")
    else:
        require(a == b, f"{path}: {a!r} != {b!r}")


def _conditions_csv(conditions):
    keys = ["id", "verifier", "model", "effort", "study", "scope", "partition", "runs", "target", "status",
            "f1", "precision", "recall", "accuracy", "overall_mae", "majority_f1", "total_cost", "mean_seconds"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(conditions)
    return stream.getvalue()


def _derived_files(data, taxonomy, tags_text, replicate_text, opc_text, opc_paired_text) -> dict[str, str]:
    files = {"conditions.csv": _conditions_csv(data["conditions"]),
             "paired-comparisons.json": json.dumps(data["paired"], indent=2) + "\n",
             "batching-sensitivities.json": json.dumps(data["batch_sensitivity"], indent=2) + "\n",
             "baselines.json": json.dumps(data["baselines"], indent=2) + "\n"}
    for name, text in capabilities(data, taxonomy, tags_text, replicate_text, opc_text, opc_paired_text).items():
        files["capabilities/" + name] = text
    return files


def _validate_release_binding(data, release) -> None:
    """Require every run to name the release's selected rollout and canonical rubric."""
    for run in data["runs"]:
        require(run["task"] in release.tasks, "Archived run references an unknown task")
        task = release.tasks[run["task"]]
        require(
            run["selected_rollout_trial"] == task.selected_rollout.trial_name
            and run["canonical_rubric_sha256"] == task.manifest["evaluation_rubric"]["sha256"],
            f"Run evidence identity differs from the release: {run['condition']}/{run['task']}",
        )


def check(bundle: Path, release_dir: Path | None = None) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    require(manifest["schema"] == SCHEMA, "Unknown bundle schema")
    names = set()
    for record in manifest["files"]:
        name = PurePosixPath(record["path"])
        require(not name.is_absolute() and ".." not in name.parts and str(name) not in names, "Unsafe/duplicate path")
        names.add(str(name))
        payload = (bundle / str(name)).read_bytes()
        require(len(payload) == record["bytes"] and sha(payload) == record["sha256"], f"File changed: {name}")
    present = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
               if p.is_file() and p.relative_to(bundle).as_posix() not in ("manifest.json", "README.md")}
    require(names == present, f"Unmanifested or missing files: {sorted(names ^ present)}")
    data = json.loads(gzip.decompress((bundle / "experiment-data.json.gz").read_bytes()))
    require(data["dataset_revision"] == manifest["dataset_revision"], "Dataset revision mismatch")
    rubric = {(r[0], r[1]): r for r in data["rubric"]}
    require(len(rubric) == len(data["rubric"]), "Duplicate criterion")
    tasks = {t for t, _ in rubric}
    human = Counter(r[5] for r in rubric.values())
    require(dict(tasks=len(tasks), criteria=len(rubric), human_unmet=human["UNMET"], human_met=human["MET"])
            == manifest["counts"]["release"], "Release counts")

    per_run = defaultdict(Counter)
    scores = defaultdict(lambda: [0.0, 0.0, 0.0])
    decisions = 0
    for cid, task, index, label, *predictions in data["criteria"]:
        r = rubric[task, index]
        require(label == r[5] and len(predictions) == 3, "Prediction identity")
        for repeat, predicted in enumerate(predictions, 1):
            if predicted is None:
                continue
            decisions += 1
            per_run[cid, task, repeat][_direction(label, predicted)] += 1
            w, hs, vs = scores[cid, task, repeat]
            scores[cid, task, repeat] = [w + r[4], hs + r[4] * (label == "MET"), vs + r[4] * (predicted == "MET")]
    runs = {(r["condition"], r["task"], r["repeat"]): r for r in data["runs"]}
    require(len(runs) == len(data["runs"]) and runs.keys() == per_run.keys(), "Run coverage")
    for key, run in runs.items():
        require({k: per_run[key][k] for k in KEYS} == {k: run[k] for k in KEYS}, f"Run confusion {key}")
        w, hs, vs = scores[key]
        require(math.isclose(hs / w, run["human_overall"], rel_tol=1e-10, abs_tol=1e-10) and
                math.isclose(vs / w, run["verifier_overall"], rel_tol=1e-10, abs_tol=1e-10), f"Weighted score {key}")
        require(math.isclose(run["difference"], run["verifier_overall"] - run["human_overall"], abs_tol=1e-10) and
                math.isclose(run["abs_error"], abs(run["difference"]), abs_tol=1e-12), f"Score error {key}")
    for condition in data["conditions"]:
        if condition["status"] == "Complete":
            repeats = condition["target"] // len(tasks)
            expected = {(condition["id"], t, rep) for t in tasks for rep in range(1, repeats + 1)}
            require({k for k in runs if k[0] == condition["id"]} == expected, f"{condition['id']}: coverage")
    require(decisions == manifest["counts"]["criterion_decisions"], "Decision count")

    derived = derive(data)
    for key in ("conditions", "paired", "baselines", "batch_sensitivity"):
        _close(derived[key], data[key], key)
    capability_dir = bundle / "capabilities"
    files = _derived_files(data, json.loads(read(capability_dir / "capability_taxonomy.json")),
                           read(capability_dir / "criterion_capabilities.csv"),
                           read(capability_dir / "criterion_capabilities_replicate.csv"),
                           read(capability_dir / "opc_primary_error_rates.csv"),
                           read(capability_dir / "opc_paired_error_differences_vs_gandalf.csv"))
    for name, text in files.items():
        require(read(bundle / name) == text, f"{name} does not match its recomputation")
    if release_dir is not None:
        _check_release(data, release_dir)
    return dict(status="passed", tasks=len(tasks), criteria=len(rubric), conditions=len(data["conditions"]),
                runs=len(runs), decisions=decisions, recomputed_files=len(files), paid_model_calls=0)


def _check_release(data, release_dir):
    sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]
    from adapters.bvb.release import BVBRelease
    from verifier_runner.scoring import load_ground_truth

    release = BVBRelease.open(release_dir)
    require(sorted(release.tasks) == sorted({r[0] for r in data["rubric"]}), "Task set differs from the release")
    expected = {}
    for task_id, task in release.tasks.items():
        for c, label in zip(task.criteria, load_ground_truth(release, task_id)):
            expected[task_id, c.evaluation_index] = [c.source_index, c.category, float(c.weight), label["decision"],
                                                     c.text, c.text_sha256]
    require({(r[0], r[1]): r[2:] for r in data["rubric"]} == expected, "Rubric or labels differ from the release")
    _validate_release_binding(data, release)
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="Recompute and verify a public results bundle.")
    check_parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    check_parser.add_argument("--release", type=Path, help="Also bind archived evidence to this public release.")
    args = parser.parse_args()
    result = check(args.bundle.resolve(), args.release and args.release.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
