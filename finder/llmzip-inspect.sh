#!/bin/zsh
# Finder Quick Action "Inspect with llmzip": open Terminal with the llmzip TUI on the
# selected file's heatmap. Only the first selected file is opened.
source "${0:A:h}/common.sh"

if (( $# == 0 )); then
  notify "llmzip ✗" "Select a file to inspect."; exit 1
fi
file=${1:A}  # absolute: the command cd's into the repo
(( $# > 1 )) && notify "llmzip" "Inspecting only the first selected file (${file:t})."
cmd="cd ${(qq)REPO} && ${(qq)UV} run --frozen llmzip tui ${(qq)file} --tab heatmap"
log "inspect: $cmd"
if [[ -n ${LLMZIP_DRY_RUN:-} ]]; then
  print -r -- "$cmd"; exit 0
fi
# The command goes in via argv, so no AppleScript escaping is needed.
/usr/bin/osascript -e 'on run argv' -e 'tell application "Terminal"' -e 'activate' \
  -e 'do script (item 1 of argv)' -e 'end tell' -e 'end run' "$cmd" >/dev/null
