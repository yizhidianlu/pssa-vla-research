#!/usr/bin/env bash
# Package smoke results and push to a results-smoke-<ts> branch on origin.
# Run from repo root on AutoDL after smoke_run.sh.
#
#   bash experiment/code/scripts/autodl/package_results.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

TS=$(date +%Y%m%d-%H%M%S)
BRANCH="results-smoke-$TS"

# Make a stable copy outside experiment/runs (which is .gitignored except
# for the smoke-* dirs we explicitly include).
RESULT_DIR="experiment/runs"
LATEST=$(ls -1dt "$RESULT_DIR"/smoke-* 2>/dev/null | head -n1 || true)
if [[ -z "$LATEST" ]]; then
    echo "FATAL: no smoke-* dir found under $RESULT_DIR — run smoke_run.sh first."
    exit 1
fi

git checkout -b "$BRANCH"
# Force-add the smoke run dir even if .gitignore excludes runs/
git add -f "$LATEST"
git -c user.email=autodl@pssa-vla -c user.name="autodl-runner" \
    commit -m "smoke results $TS

Captured by experiment/code/scripts/autodl/smoke_run.sh on AutoDL.
GPU: $(nvidia-smi -L | head -1)
"
git push -u origin "$BRANCH"

# Also produce a tarball as a fallback in case the user wants to download
# directly from AutoDL's file manager instead of via git.
tar -C "$RESULT_DIR" -czf "/root/autodl-tmp/${BRANCH}.tar.gz" "$(basename "$LATEST")"
echo "==> branch: $BRANCH (pushed to origin)"
echo "==> tarball: /root/autodl-tmp/${BRANCH}.tar.gz"
echo "==> tell Claude Code: 'pull branch $BRANCH'"
