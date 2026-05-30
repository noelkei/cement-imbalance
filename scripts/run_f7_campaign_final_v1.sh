#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
CLEANUP_EVERY="${CLEANUP_EVERY:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib-codex}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"

PRIMARY_SPEC="config/f7_campaign_spec_v1.yaml"
EXTENSION1_SPEC="config/f7_campaign_extension1_v1.yaml"
EXTENSION2_SPEC="config/f7_campaign_extension2_v1.yaml"
EXTENSION3_SPEC="config/f7_campaign_extension3_v1.yaml"

PRIMARY_ID="f7_campaign_v1"
EXTENSION1_ID="f7_campaign_extension1_v1"
EXTENSION2_ID="f7_campaign_extension2_v1"
EXTENSION3_ID="f7_campaign_extension3_v1"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_f7_campaign_final_v1.sh <command>

Commands:
  materialize            Materialize primary and all three extension inventories.
  readiness              Regenerate the canonical launch-readiness report.
  preflight-primary      Preflight the primary campaign.
  preflight-all          Preflight the launch chain safely. Primary is checked now; extensions are checked only if their parent is already closed_success.
  run-primary            Run the primary campaign only.
  run-all                Run primary, then extension 1, then extension 2, then extension 3.
  run-extension1         Run extension 1 after the primary is closed_success.
  run-extension2         Run extension 2 after extension 1 is closed_success.
  run-extension3         Run extension 3 after extension 2 is closed_success.
  resume-primary         Resume the primary campaign.
  resume-extension1      Resume extension 1.
  resume-extension2      Resume extension 2.
  resume-extension3      Resume extension 3.
  close-primary          Write closeout for the primary campaign.
  close-extension1       Write closeout for extension 1.
  close-extension2       Write closeout for extension 2.
  close-extension3       Write closeout for extension 3.
  reports                Generate campaign, lineage, readiness, and storage reports.

Environment overrides:
  PYTHON_BIN=<python executable>   Default: python
  DEVICE=<device>                  Default: cpu
  CLEANUP_EVERY=<int>              Default: 1
  MPLCONFIGDIR=<path>              Default: /private/tmp/matplotlib-codex
  XDG_CACHE_HOME=<path>            Default: /private/tmp
EOF
}

require_closed_success() {
  local campaign_id="$1"
  local manifest_path="outputs/campaigns/${campaign_id}/campaign_manifest.json"
  local status
  status="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path("${manifest_path}")
if not path.exists():
    raise SystemExit("missing manifest for ${campaign_id}")
print(json.loads(path.read_text(encoding="utf-8")).get("campaign_status", ""))
PY
)"
  if [[ "${status}" != "closed_success" ]]; then
    echo "${campaign_id} did not close successfully: ${status}" >&2
    exit 1
  fi
}

campaign_status_or_missing() {
  local campaign_id="$1"
  local manifest_path="outputs/campaigns/${campaign_id}/campaign_manifest.json"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
path = Path("${manifest_path}")
if not path.exists():
    print("missing")
else:
    print(json.loads(path.read_text(encoding="utf-8")).get("campaign_status", "unknown"))
PY
}

materialize_one() {
  local spec_path="$1"
  "${PYTHON_BIN}" scripts/materialize_f7_campaign_spec.py --spec-path "${spec_path}"
}

report_one_campaign() {
  local campaign_id="$1"
  "${PYTHON_BIN}" scripts/report_f7_campaign.py --campaign-id "${campaign_id}"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cd "${ROOT_DIR}"

case "$1" in
  materialize)
    materialize_one "${PRIMARY_SPEC}"
    materialize_one "${EXTENSION1_SPEC}"
    materialize_one "${EXTENSION2_SPEC}"
    materialize_one "${EXTENSION3_SPEC}"
    ;;
  readiness)
    "${PYTHON_BIN}" scripts/report_f7_launch_readiness.py
    ;;
  preflight-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${PRIMARY_SPEC}"
    ;;
  preflight-all)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${PRIMARY_SPEC}"
    primary_status="$(campaign_status_or_missing "${PRIMARY_ID}")"
    if [[ "${primary_status}" == "closed_success" ]]; then
      "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${EXTENSION1_SPEC}"
    else
      echo "Skipping extension1 preflight until ${PRIMARY_ID} is closed_success (current: ${primary_status})."
    fi
    extension1_status="$(campaign_status_or_missing "${EXTENSION1_ID}")"
    if [[ "${extension1_status}" == "closed_success" ]]; then
      "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${EXTENSION2_SPEC}"
    else
      echo "Skipping extension2 preflight until ${EXTENSION1_ID} is closed_success (current: ${extension1_status})."
    fi
    extension2_status="$(campaign_status_or_missing "${EXTENSION2_ID}")"
    if [[ "${extension2_status}" == "closed_success" ]]; then
      "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${EXTENSION3_SPEC}"
    else
      echo "Skipping extension3 preflight until ${EXTENSION2_ID} is closed_success (current: ${extension2_status})."
    fi
    ;;
  run-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${PRIMARY_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-all)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${PRIMARY_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}" --chain-next-spec "${EXTENSION1_SPEC}"
    require_closed_success "${EXTENSION1_ID}"
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION2_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    require_closed_success "${EXTENSION2_ID}"
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION3_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extension1)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION1_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION2_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extension3)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION3_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${PRIMARY_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-extension1)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${EXTENSION1_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${EXTENSION2_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-extension3)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${EXTENSION3_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  close-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${PRIMARY_ID}"
    ;;
  close-extension1)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${EXTENSION1_ID}"
    ;;
  close-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${EXTENSION2_ID}"
    ;;
  close-extension3)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${EXTENSION3_ID}"
    ;;
  reports)
    report_one_campaign "${PRIMARY_ID}"
    report_one_campaign "${EXTENSION1_ID}"
    report_one_campaign "${EXTENSION2_ID}"
    report_one_campaign "${EXTENSION3_ID}"
    "${PYTHON_BIN}" scripts/report_f7_lineage.py --root-campaign-id "${PRIMARY_ID}"
    "${PYTHON_BIN}" scripts/report_f7_storage_footprint.py --root-campaign-id "${PRIMARY_ID}"
    "${PYTHON_BIN}" scripts/report_f7_launch_readiness.py
    ;;
  *)
    usage
    exit 1
    ;;
esac
