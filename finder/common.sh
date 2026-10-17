# Shared helpers for the llmzip Finder Quick Action scripts (zsh; sourced, not run).
#
# Configuration (all optional; install.sh bakes LLMZIP_UV and LLMZIP_REPO into each
# Quick Action because Automator runs with a minimal PATH):
#   LLMZIP_UV          absolute path of uv
#   LLMZIP_REPO        absolute path of the llmzip repo
#   LLMZIP_LOG_FILE    log file (default ~/Library/Logs/llmzip/finder.log)
#   LLMZIP_NOTIFY_CMD  executable called as: CMD TITLE MESSAGE (tests use a recorder)
#   LLMZIP_DRY_RUN     inspect only: print the Terminal command instead of opening it

REPO=${LLMZIP_REPO:-${${(%):-%x}:A:h:h}}
UV=${LLMZIP_UV:-${commands[uv]:-/opt/homebrew/bin/uv}}
LOG_FILE=${LLMZIP_LOG_FILE:-$HOME/Library/Logs/llmzip/finder.log}

notify() {  # notify TITLE MESSAGE
  if [[ -n ${LLMZIP_NOTIFY_CMD:-} ]]; then
    "$LLMZIP_NOTIFY_CMD" "$1" "$2"
  else
    # Text goes in via argv, so quotes/apostrophes/unicode need no escaping.
    /usr/bin/osascript -e 'on run argv' \
      -e 'display notification (item 2 of argv) with title (item 1 of argv)' \
      -e 'end run' "$1" "$2" >/dev/null 2>&1
  fi
}

log() {
  mkdir -p -- "${LOG_FILE:h}" 2>/dev/null
  print -r -- "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG_FILE" 2>/dev/null
}

# run_llmzip DIR ARGS... : run the CLI inside DIR; sets OUT (stdout+stderr) and RC.
run_llmzip() {
  local dir=$1; shift
  OUT=$(cd -- "$dir" && "$UV" run --project "$REPO" --frozen --quiet llmzip "$@" 2>&1)
  RC=$?
  log "llmzip ${(j: :)${(qq)@}}  (in ${(qq)dir}) -> exit $RC"
  local line
  for line in "${(@f)OUT}"; do log "    $line"; done
}

# last non-empty line of OUT (the CLI's error message on failure)
last_line() {
  local lines=("${(@f)OUT}")
  lines=(${lines:#})
  print -r -- "${lines[-1]:-unknown error (exit $RC)}"
}
