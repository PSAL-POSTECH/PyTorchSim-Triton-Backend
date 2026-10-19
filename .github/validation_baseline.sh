#!/usr/bin/env bash
# From main's newest images run: OUT/<suite>/results.json, OUT/commit and OUT/error-targets.json. Leaves OUT
# empty when main has none yet, and never fails the job.
#   GH_TOKEN=... REPO=owner/name .github/validation_baseline.sh OUT
set -uo pipefail
: "${GH_TOKEN:?a token that reads the actions of this repository}"
: "${REPO:?owner/name}"
OUT=${1:?the directory to write}
SUITES="gemm vector attention"
mkdir -p "$OUT"

api() { curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" "$@"; }

RUNS=$(api "https://api.github.com/repos/$REPO/actions/workflows/images.yml/runs?branch=main&event=push&per_page=30") \
  || { echo "::warning::could not list main's images runs: no baseline"; exit 0; }
for run in $(jq -r --arg me "${GITHUB_RUN_ID:-0}" '.workflow_runs[] | select((.id|tostring) != $me) | .id' <<<"$RUNS"); do
  ARTS=$(api "https://api.github.com/repos/$REPO/actions/runs/$run/artifacts?per_page=100") || continue
  SHA=$(jq -r --arg r "$run" '.workflow_runs[] | select((.id|tostring) == $r) | .head_sha' <<<"$RUNS")
  have=1
  for s in $SUITES; do
    jq -e --arg n "$s-validation-$SHA" '.artifacts[] | select(.name == $n and (.expired|not))' <<<"$ARTS" >/dev/null || have=0
  done
  [ "$have" = 1 ] || continue
  for s in $SUITES; do
    id=$(jq -r --arg n "$s-validation-$SHA" '.artifacts[] | select(.name == $n) | .id' <<<"$ARTS" | head -1)
    api -o "$OUT/$s.zip" "https://api.github.com/repos/$REPO/actions/artifacts/$id/zip" || { have=0; break; }
    mkdir -p "$OUT/$s"
    python3 -c 'import sys, zipfile; open(sys.argv[2], "wb").write(zipfile.ZipFile(sys.argv[1]).read("results.json"))' \
      "$OUT/$s.zip" "$OUT/$s/results.json" || { have=0; break; }
    rm -f "$OUT/$s.zip"
  done
  if [ "$have" = 1 ]; then
    id=$(jq -r '.artifacts[] | select(.name == "error-targets" and (.expired|not)) | .id' <<<"$ARTS" | head -1)
    if [ -n "$id" ] && api -o "$OUT/et.zip" "https://api.github.com/repos/$REPO/actions/artifacts/$id/zip"; then
      python3 -c 'import sys, zipfile; open(sys.argv[2], "wb").write(zipfile.ZipFile(sys.argv[1]).read("error-targets.json"))' \
        "$OUT/et.zip" "$OUT/error-targets.json" || rm -f "$OUT/error-targets.json"
      rm -f "$OUT/et.zip"
    fi
    echo "$SHA" > "$OUT/commit"
    echo "baseline: main $SHA (run $run)"
    exit 0
  fi
  rm -rf "${OUT:?}"/*
done
echo "::warning::no images run on main has all of $SUITES validation results: no baseline"
exit 0
