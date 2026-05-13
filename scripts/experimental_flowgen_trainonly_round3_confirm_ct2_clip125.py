from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import experimental_flowgen_trainonly_round3 as round3


POLICY_ID = "R3A2_t06_clip125"
BASE_TOKEN = "candidate_trainonly_2"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Single-run confirmation wrapper for the train-only round3 family "
            "R3A2_t06_clip125 on candidate_trainonly_2."
        )
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def _apply_confirm_overrides() -> None:
    round3.OUTPUT_SUBDIR_ROOT = "round3_confirm"
    round3.SUMMARY_ROOT = round3.FLOWGEN_TRAINONLY_ROOT / "campaign_summaries" / "round3_confirm"
    round3.CONTRACT_ID = "flowgen_trainonly_round3_confirm_ct2_clip125_v1"
    round3.OBJECTIVE_METRIC_ID = "flowgen_trainonly_realism_round3_confirm_ct2_clip125"
    round3.PREFERRED_POLICY_ORDER = [POLICY_ID]

    spec = copy.deepcopy(round3.FLOWGEN_TRAINONLY_POLICY_SPECS[POLICY_ID])
    spec["allowed_base_tokens"] = [BASE_TOKEN]
    spec["historical_origin"] = (
        str(spec["historical_origin"])
        + " Confirmation-only rerun on candidate_trainonly_2 after round3 showed the strongest X gain on "
        + "candidate_trainonly_1 but left this transfer check unresolved."
    )
    round3.FLOWGEN_TRAINONLY_POLICY_SPECS = {POLICY_ID: spec}

    def _confirm_run_id_for(base: round3.BaseContext, policy: round3.TrainOnlyPolicy, run_seed: int) -> str:
        token = round3.BASE_TOKENS[base.token]
        policy_slug = round3._policy_slug_for_run_id(policy.policy_id)
        return f"flowgen_trainonly_tpv1_{token}_round3confirm_{policy_slug}_seed{int(run_seed)}_v1"

    round3._run_id_for = _confirm_run_id_for

    original_build_config_payload = round3._build_config_payload
    original_build_evaluation_context = round3._build_evaluation_context

    def _confirm_build_config_payload(*, base, policy, run_seed):
        payload = original_build_config_payload(base=base, policy=policy, run_seed=run_seed)
        block = payload.get("trainonly_training") or {}
        block["mode"] = "flowgen_trainonly_round3_confirm_finetune"
        block["policy_set_id"] = "flowgen_trainonly_round3_confirm_ct2_clip125_v1"
        payload["trainonly_training"] = block
        return payload

    def _confirm_build_evaluation_context(*, base, policy, run_seed):
        ctx = original_build_evaluation_context(base=base, policy=policy, run_seed=run_seed)
        ctx["contract_id"] = round3.CONTRACT_ID
        ctx["seed_set_id"] = f"{base.token}_round3confirm_seed{int(run_seed)}"
        ctx["objective_metric_id"] = round3.OBJECTIVE_METRIC_ID
        axes = ctx.get("run_level_axes") or {}
        axes["phase"] = "trainonly_round3_confirm_finetune"
        ctx["run_level_axes"] = axes
        return ctx

    round3._build_config_payload = _confirm_build_config_payload
    round3._build_evaluation_context = _confirm_build_evaluation_context


def main() -> int:
    args = _parse_args()
    _apply_confirm_overrides()
    verbose = not args.quiet

    bases = [round3._resolve_base_context(BASE_TOKEN)]
    policies = round3._build_policies()
    entries = round3._build_plan(
        bases=bases,
        policies=policies,
        allowed_policy_ids={POLICY_ID},
    )
    if not entries:
        raise RuntimeError("No confirmation run was built for ct2 + R3A2_t06_clip125.")

    round3.SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    campaign_id = f"flowgen_trainonly_round3_confirm_{round3._campaign_timestamp()}"
    plan_json_path = round3.SUMMARY_ROOT / f"{campaign_id}_plan.json"
    plan_csv_path = round3.SUMMARY_ROOT / f"{campaign_id}_plan.csv"

    plan_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "seed_policy": "per_base_flowgen_seed",
        "model_family": round3.MODEL_FAMILY,
        "contract_id": round3.CONTRACT_ID,
        "output_root": str(round3.FLOWGEN_TRAINONLY_ROOT / round3.OUTPUT_SUBDIR_ROOT),
        "bases": [round3._base_plan_payload(base) for base in bases],
        "policy_set": [
            {
                "policy_id": policy.policy_id,
                "allowed_base_tokens": list(policy.allowed_base_tokens),
                "policy_signature": policy.policy_signature,
                "historical_origin": policy.historical_origin,
                "historical_source_run_ids": list(policy.historical_source_run_ids),
            }
            for policy in policies
        ],
        "runs": [round3.asdict(entry) for entry in entries],
    }
    round3._write_json(plan_json_path, plan_payload)
    round3._write_rows_csv(plan_csv_path, [round3.asdict(entry) for entry in entries])

    round3._print_plan_summary(
        bases=bases,
        policies=policies,
        entries=entries,
        summary_paths={"plan_json": plan_json_path, "plan_csv": plan_csv_path},
    )

    if args.summary_only:
        print("\nSummary-only mode enabled. No training started.")
        return 0

    entry = entries[0]
    if entry.existing_status == "complete":
        print(f"\nRun already complete: {entry.run_id} -> {entry.output_dir}")
        return 0
    if entry.existing_status == "incomplete":
        removed = round3._reset_incomplete_run_dir(entry.output_dir)
        print(f"\nRemoved incomplete run directory before retry: {removed}")

    print("\n" + "=" * 100)
    print("Starting FlowGen train-only round3 confirmation | planned runs=1")
    print("\n" + "-" * 100)
    print(f"[1/1] {entry.run_id}")
    print(
        f"  base={entry.base_run_id} | policy={entry.policy_id} | "
        f"seed={entry.run_seed} | existing={entry.existing_status}"
    )

    try:
        entry.result_paths = round3._run_one(
            base=bases[0],
            policy=policies[0],
            entry=entry,
            device=args.device,
            verbose=verbose,
        )
        entry.status = "completed"
        completed = 1
        failed = 0
        print(f"  status=completed | output={entry.output_dir}")
    except Exception as exc:
        entry.status = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"
        completed = 0
        failed = 1
        print(f"  status=failed | error={entry.error}")
    finally:
        round3._release_process_memory()

    results_payload = {
        "campaign_id": campaign_id,
        "script": str(Path(__file__).resolve()),
        "seed_policy": "per_base_flowgen_seed",
        "contract_id": round3.CONTRACT_ID,
        "output_root": str(round3.FLOWGEN_TRAINONLY_ROOT / round3.OUTPUT_SUBDIR_ROOT),
        "completed": completed,
        "skipped_existing": 0,
        "failed": failed,
        "runs": [round3.asdict(entry)],
    }
    results_json_path = round3.SUMMARY_ROOT / f"{campaign_id}_results.json"
    results_csv_path = round3.SUMMARY_ROOT / f"{campaign_id}_results.csv"
    round3._write_json(results_json_path, results_payload)
    round3._write_rows_csv(results_csv_path, [round3.asdict(entry)])

    print("\n" + "=" * 100)
    print("FlowGen train-only round3 confirmation finished")
    print(f"Completed: {completed}")
    print("Skipped existing: 0")
    print(f"Failed: {failed}")
    print(f"Results JSON: {results_json_path}")
    print(f"Results CSV:  {results_csv_path}")
    if completed:
        print("\nOutput paths:")
        print(f"  - {entry.run_id}: {entry.output_dir}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
