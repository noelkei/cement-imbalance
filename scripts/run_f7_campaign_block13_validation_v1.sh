#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
CLEANUP_EVERY="${CLEANUP_EVERY:-1}"

PRIMARY_SPEC="config/f7_campaign_block13_validation_primary_v1.yaml"
EXTENSION_SPEC="config/f7_campaign_block13_validation_extension_v1.yaml"
EXTENSION2_SPEC="config/f7_campaign_block13_validation_extension2_v1.yaml"
PRIMARY_ID="f7_campaign_block13_validation_primary_v1"
EXTENSION_ID="f7_campaign_block13_validation_extension_v1"
EXTENSION2_ID="f7_campaign_block13_validation_extension2_v1"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_f7_campaign_block13_validation_v1.sh <command>

Commands:
  materialize            Materialize primary and extension inventories.
  preflight-primary      Preflight the primary validation campaign.
  preflight-extension    Preflight the extension campaign. Requires parent campaign to exist and be closed.
  preflight-extension2   Preflight the second extension campaign. Requires first extension campaign to exist and be closed.
  run-primary            Run the primary validation campaign.
  run-primary-chain      Run the primary campaign and chain the extension immediately after success.
  run-all                Run primary, then extension 1, then extension 2 as one sequential validation flow.
  run-extensions         Run extension 1 and then extension 2, assuming primary is already closed_success.
  resume-primary         Resume the primary validation campaign.
  resume-extension       Resume the extension validation campaign.
  resume-extension2      Resume the second extension validation campaign.
  run-extension          Run the extension campaign as a standalone follow-up after the primary is closed.
  run-extension2         Run the second extension campaign after extension 1 is closed.
  close-primary          Write closeout for the primary campaign.
  close-extension        Write closeout for the extension campaign.
  close-extension2       Write closeout for the second extension campaign.

Environment overrides:
  PYTHON_BIN=<python executable>   Default: python
  DEVICE=<device>                  Default: cpu
  CLEANUP_EVERY=<int>              Default: 1
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cd "${ROOT_DIR}"

case "$1" in
  materialize)
    "${PYTHON_BIN}" scripts/materialize_f7_campaign_block13_validation_v1.py
    ;;
  preflight-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${PRIMARY_SPEC}"
    ;;
  preflight-extension)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${EXTENSION_SPEC}"
    ;;
  preflight-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py preflight --campaign-spec "${EXTENSION2_SPEC}"
    ;;
  run-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${PRIMARY_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-primary-chain)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${PRIMARY_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}" --chain-next-spec "${EXTENSION_SPEC}"
    ;;
  run-all)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${PRIMARY_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}" --chain-next-spec "${EXTENSION_SPEC}"
    extension_status="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("outputs/campaigns/f7_campaign_block13_validation_extension_v1/campaign_manifest.json")
if not path.exists():
    raise SystemExit("missing extension manifest after run-primary-chain")
print(json.loads(path.read_text(encoding="utf-8")).get("campaign_status", ""))
PY
)"
    if [[ "${extension_status}" != "closed_success" ]]; then
      echo "extension 1 did not close successfully: ${extension_status}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION2_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extensions)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    extension_status="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("outputs/campaigns/f7_campaign_block13_validation_extension_v1/campaign_manifest.json")
if not path.exists():
    raise SystemExit("missing extension manifest after run-extension")
print(json.loads(path.read_text(encoding="utf-8")).get("campaign_status", ""))
PY
)"
    if [[ "${extension_status}" != "closed_success" ]]; then
      echo "extension 1 did not close successfully: ${extension_status}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION2_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${PRIMARY_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-extension)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${EXTENSION_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  resume-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py resume --campaign-id "${EXTENSION2_ID}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extension)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  run-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py run --campaign-spec "${EXTENSION2_SPEC}" --device "${DEVICE}" --cleanup-every "${CLEANUP_EVERY}"
    ;;
  close-primary)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${PRIMARY_ID}"
    ;;
  close-extension)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${EXTENSION_ID}"
    ;;
  close-extension2)
    "${PYTHON_BIN}" scripts/run_f7_campaign.py close --campaign-id "${EXTENSION2_ID}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
