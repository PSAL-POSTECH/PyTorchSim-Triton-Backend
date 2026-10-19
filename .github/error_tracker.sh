#!/usr/bin/env bash
# Posts the validation error targets (validation/error_targets.py's TABLE and JSON) where people and the loop see
# them, and never fails the job:
#   main push     the tracker issue's body is the latest table; each run adds it as a comment, so the issue is
#                 the history. The loop's next iteration is started by claude.yml when this run completes.
#   pull request  one comment on the pull request, rewritten by every run.
#   GH_TOKEN=... REPO=owner/name EVENT=push|pull_request [PR=n] SHA=... RUN_URL=... TABLE=md JSON=json \
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
if [ -z "$issue" ]; then
  issue=$(api POST issues "$(jq -nc --arg t "$TITLE" --arg b "$BODY" '{title:$t, body:$b}')" | jq -r .number) || exit 0
else
  api PATCH "issues/$issue" "$(jq -nc --arg b "$BODY" '{body:$b}')" >/dev/null || exit 0
fi

off=$(jq -r '[.groups | to_entries[] | select((.value.on_target | not) and (.value.excluded | not)) | .key] | join(", ")' "$JSON")
NOTE=""
[ -z "$off" ] || NOTE=$'\n\nOff target: '"$off"'.'
api POST "issues/$issue/comments" "$(jq -nc --arg b "$BODY$NOTE" '{body:$b}')" >/dev/null
echo "error targets posted on #$issue (off target: ${off:-none})"
exit 0
