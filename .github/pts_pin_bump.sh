#!/usr/bin/env bash
# Proposes PyTorchSim's compiler pin at this repository's newest main commit with a published image:
# one branch (BUMP_BRANCH) force-pushed from the tip of PTS_BASE, one open pull request on it.
# DRY_RUN=1 makes only read calls and prints the writes it would make.
set -euo pipefail

: "${PTB:=PSAL-POSTECH/PyTorchSim-Triton-Backend}"
: "${PTS:=PSAL-POSTECH/PyTorchSim}"
: "${PTS_BASE:=pytorchsim-triton-backend}"
: "${BUMP_BRANCH:=bot/ptb-bump}"
: "${PIN_FILE:=third_party/pytorchsim-triton-backend.json}"
: "${IMAGE:=ghcr.io/psal-postech/pytorchsim-triton-backend}"
: "${DRY_RUN:=0}"
: "${PTB_TOKEN:?read token for $PTB (contents, packages)}"
: "${PTS_TOKEN:?write token for $PTS (contents, pull requests)}"
: "${NEW:?the main commit this run published}"
MARKER='<!-- ptb-pin-bump -->'
NAME='Wonhyuk Yang'
EMAIL='wonhyuk@postech.ac.kr'

api() {  # token method path [json]
  local tok=$1 method=$2 path=$3 data=${4:-}
  if [ "$method" != GET ] && [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: $method $path ${data:0:300}" >&2; echo '{}'; return
  fi
  curl -fsSL -X "$method" -H "Authorization: Bearer $tok" -H "Accept: application/vnd.github+json" \
    ${data:+-d "$data"} "https://api.github.com/$path"
}
ptb() { api "$PTB_TOKEN" GET "repos/$PTB/$1"; }
pts() { api "$PTS_TOKEN" "$@"; }

published() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -I \
    -H "Authorization: Bearer $(printf %s "$PTB_TOKEN" | base64 -w0)" \
    -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json" \
    "https://ghcr.io/v2/${IMAGE#ghcr.io/}/manifests/$1")
  [ "$code" = 200 ]
}
relation() { ptb "compare/$1...$2" | jq -r .status; }  # how $2 stands to $1: ahead behind identical diverged
pin_at() { pts GET "repos/$PTS/contents/$PIN_FILE?ref=$1" | jq -r .content | base64 -d | jq -r .pytorchsim_triton_backend.ref; }

# The newest main commit with an image: main's head when its image is up (a later run may have been
# cancelled for this one), else the commit this run published.
TIP=$(ptb "commits/main" | jq -r .sha)
TARGET=$NEW
if [ "$TIP" != "$NEW" ] && [ "$(relation "$NEW" "$TIP")" = ahead ] && published "$TIP"; then TARGET=$TIP; fi
published "$TARGET" || { echo "::error::$IMAGE:$TARGET is not published"; exit 1; }
echo "target: $TARGET (this run: $NEW, main: $TIP)"

OPEN=$(pts GET "repos/$PTS/pulls?state=open&base=$PTS_BASE&per_page=100")
OURS=$(jq -r --arg b "$BUMP_BRANCH" --arg o "${PTS%%/*}" \
  '[.[] | select(.head.ref == $b and .head.repo.owner.login == $o)][0].number // empty' <<<"$OPEN")

# Older per-commit bump pull requests this automation opened (the marker says so); a hand-made one stays.
jq -r --arg m "$MARKER" --arg n "$OURS" \
  '.[] | select((.head.ref | startswith("pin/ptb-")) and ((.body // "") | contains($m)) and ((.number|tostring) != $n)) | .number' \
  <<<"$OPEN" | while read -r n; do
  echo "superseded: #$n"
  pts POST "repos/$PTS/issues/$n/comments" "$(jq -nc --arg b "Superseded by the bump on \`$BUMP_BRANCH\` (PTB main ${TARGET:0:7})." '{body:$b}')" >/dev/null
  pts PATCH "repos/$PTS/pulls/$n" '{"state":"closed"}' >/dev/null
done

OLD=$(pin_at "$PTS_BASE")
echo "$PTS_BASE pins $OLD"
case $(relation "$OLD" "$TARGET") in
  ahead) ;;
  *)
    echo "the pin is already at or past ${TARGET:0:7}: nothing to propose"
    if [ -n "$OURS" ]; then
      pts POST "repos/$PTS/issues/$OURS/comments" "$(jq -nc --arg b "$PTS_BASE already pins ${OLD:0:7}, at or past PTB main ${TARGET:0:7}." '{body:$b}')" >/dev/null
      pts PATCH "repos/$PTS/pulls/$OURS" '{"state":"closed"}' >/dev/null
    fi
    exit 0 ;;
esac

BASE_SHA=$(pts GET "repos/$PTS/branches/$PTS_BASE" | jq -r .commit.sha)
if BRANCH=$(pts GET "repos/$PTS/branches/$BUMP_BRANCH" 2>/dev/null); then
  PREV=$(pin_at "$BUMP_BRANCH")
  if [ "$PREV" = "$TARGET" ] && [ "$(jq -r '.commit.parents[0].sha' <<<"$BRANCH")" = "$BASE_SHA" ] && [ -n "$OURS" ]; then
    echo "$BUMP_BRANCH already proposes ${TARGET:0:7} on $PTS_BASE's tip"; exit 0
  fi
  if [ "$PREV" != "$TARGET" ] && [ "$(relation "$PREV" "$TARGET")" != ahead ]; then
    echo "$BUMP_BRANCH proposes ${PREV:0:7}, not behind ${TARGET:0:7}: a newer run got there"; exit 0
  fi
fi

S=${TARGET:0:7}
COMPARE="https://github.com/$PTB/compare/$OLD...$TARGET"
LOG=$(ptb "compare/$OLD...$TARGET" | jq -r '.commits[] | "- \(.sha[0:7]) \(.commit.message | split("\n")[0])"')
N=$(wc -l <<<"$LOG")
[ "$N" -le 40 ] || LOG="$(head -40 <<<"$LOG")"$'\n'"- ... $((N - 40)) more in the compare view"

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 --branch "$PTS_BASE" "https://x-access-token:$PTS_TOKEN@github.com/$PTS.git" "$WORK/pts"
cd "$WORK/pts"
sed -i "s/$OLD/$TARGET/" "$PIN_FILE"
[ "$(jq -r .pytorchsim_triton_backend.ref "$PIN_FILE")" = "$TARGET" ] || { echo "::error::$PIN_FILE did not take $TARGET"; exit 1; }
[ "$(git diff --numstat | awk '{print $1 "+" $2 " " $3}')" = "1+1 $PIN_FILE" ] || { git diff; echo "::error::the bump touched more than the ref"; exit 1; }
git add "$PIN_FILE"
GIT_AUTHOR_NAME=$NAME GIT_AUTHOR_EMAIL=$EMAIL GIT_COMMITTER_NAME=$NAME GIT_COMMITTER_EMAIL=$EMAIL \
  git commit -q -F - <<EOF
[Simulator] The compiler pin is PTB main $S

$PIN_FILE moves from ${OLD:0:7} to PTB main $S, whose image is published.
$COMPARE
EOF
git show --stat HEAD
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: git push --force origin HEAD:refs/heads/$BUMP_BRANCH"
else git push -q --force origin "HEAD:refs/heads/$BUMP_BRANCH"; fi

TITLE="[Simulator] Compiler pin: PTB main $S"
BODY="Moves \`$PIN_FILE\` from ${OLD:0:7} to PTB main $S ([compare]($COMPARE)). The PTB image for $S is published.

PTB commits in the range, oldest first:
$LOG

PyTorchSim-side changes the new compiler needs, if any, are not included.

Opened by PTB's images workflow. A newer PTB main commit force-pushes \`$BUMP_BRANCH\` and rewrites this pull request, so only the newest bump is ever open.
$MARKER"
if [ -n "$OURS" ]; then
  echo "updating #$OURS"
  pts PATCH "repos/$PTS/pulls/$OURS" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" '{title:$t, body:$b}')" >/dev/null
else
  echo "opening a pull request"
  pts POST "repos/$PTS/pulls" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" --arg h "$BUMP_BRANCH" --arg base "$PTS_BASE" \
    '{title:$t, body:$b, head:$h, base:$base}')" | jq -r '.html_url // empty'
fi
[ "$DRY_RUN" != 1 ] || printf '%s\n---\n%s\n' "$TITLE" "$BODY"
