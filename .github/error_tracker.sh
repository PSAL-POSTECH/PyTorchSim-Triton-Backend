#!/usr/bin/env bash
# Posts validation/error_targets.py's table; never fails the job.
#   push          tracker issue body = latest table, plus a comment (the history) with the commits since PREV_SHA.
#   pull_request  one comment on the pull request, rewritten by every run.
#   probe         one comment on the tracker (REF = the claude/* branch).
#   GH_TOKEN=... REPO=owner/name EVENT=push|pull_request [PR=n] SHA=... [PREV_SHA=...] RUN_URL=... TABLE=md JSON=json \
#     .github/error_tracker.sh
set -uo pipefail
: "${GH_TOKEN:?}" "${REPO:?}" "${EVENT:?}" "${SHA:?}" "${TABLE:?}" "${JSON:?}"
TITLE="Validation error tracker"
MARK='<!-- validation-error-tracker -->'

api() {  # method path [json]
  curl -fsSL -X "$1" -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github+json" \
    ${3:+-d "$3"} "https://api.github.com/repos/$REPO/$2"
}

HEAD="Main \`${SHA:0:12}\`${RUN_URL:+ ([run]($RUN_URL))}"
[ "$EVENT" != probe ] || HEAD="Probe of \`${REF:-?}\` at \`${SHA:0:12}\`${RUN_URL:+ ([run]($RUN_URL))}, against main \`${PREV_SHA:0:12}\`"
BODY="$(printf '%s\n\n%s\n%s\n' "$HEAD" "$(cat "$TABLE")" "$MARK")"

if [ "$EVENT" = pull_request ]; then
  : "${PR:?}"
  id=$(api GET "issues/$PR/comments?per_page=100" | jq -r --arg m "$MARK" '[.[] | select(.body | contains($m))][0].id // empty') || exit 0
  BODY="${BODY/Main /This pull request }"
  if [ -n "$id" ]; then api PATCH "issues/comments/$id" "$(jq -nc --arg b "$BODY" '{body:$b}')" >/dev/null
  else api POST "issues/$PR/comments" "$(jq -nc --arg b "$BODY" '{body:$b}')" >/dev/null; fi
  echo "error targets posted on #$PR"
  exit 0
fi

issue=$(api GET "issues?state=open&per_page=100&creator=github-actions%5Bbot%5D" \
        | jq -r --arg t "$TITLE" '[.[] | select(.title == $t and (.pull_request | not))][0].number // empty') || exit 0
if [ "$EVENT" = probe ]; then
  [ -n "$issue" ] || exit 0
  api POST "issues/$issue/comments" "$(jq -nc --arg b "$BODY" '{body:$b}')" >/dev/null
  echo "probe result posted on #$issue"
  exit 0
fi
if [ -z "$issue" ]; then
  issue=$(api POST issues "$(jq -nc --arg t "$TITLE" --arg b "$BODY" '{title:$t, body:$b}')" | jq -r .number) || exit 0
else
  api PATCH "issues/$issue" "$(jq -nc --arg b "$BODY" '{body:$b}')" >/dev/null || exit 0
fi

SINCE=""
if [ -n "${PREV_SHA:-}" ] && [ "$PREV_SHA" != "$SHA" ]; then
  log=$(api GET "compare/$PREV_SHA...$SHA" | jq -r '.commits[] | "- \(.sha[0:7]) \(.commit.message | split("\n")[0])"') || log=""
  [ -z "$log" ] || SINCE=$'\n\nSince main `'"${PREV_SHA:0:12}"$'` (the previous column):\n'"$log"
fi
off=$(jq -r '[.groups | to_entries[] | select(.value.on_target == false and (.value.excluded | not)) | .key] | join(", ")' "$JSON")
NOTE=""
[ -z "$off" ] || NOTE=$'\n\nOff target: '"$off"'.'
api POST "issues/$issue/comments" "$(jq -nc --arg b "$BODY$SINCE$NOTE" '{body:$b}')" >/dev/null
echo "error targets posted on #$issue (off target: ${off:-none})"
exit 0
