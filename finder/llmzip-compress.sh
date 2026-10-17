#!/bin/zsh
# Finder Quick Action "Compress with llmzip": compress each selected file (auto mode).
# Never overwrites: if FILE.lz exists, the CLI's --unique picks FILE-2.ext.lz, etc.
source "${0:A:h}/common.sh"

log "compress action: $# item(s)"
for f in "$@"; do
  name=${f:t}
  if [[ -d $f ]]; then
    notify "llmzip" "Skipped folder “$name” (select files to compress)."
    log "skip folder $f"; continue
  fi
  if [[ $f == *.lz ]]; then
    notify "llmzip" "Skipped “$name”: it is already a .lz file (use Decompress)."
    log "skip .lz $f"; continue
  fi
  if [[ ! -f $f ]]; then
    notify "llmzip ✗ $name" "File not found."
    log "missing $f"; continue
  fi
  notify "llmzip" "Compressing $name… you'll get a notification when it's done."
  run_llmzip "${f:A:h}" compress "$name" --unique
  if (( RC == 0 )); then
    summary=${(M)${(@f)OUT}:#compress:*}
    notify "llmzip ✓ compressed $name" "${summary#compress: }"
  else
    notify "llmzip ✗ $name" "$(last_line)"
  fi
done
