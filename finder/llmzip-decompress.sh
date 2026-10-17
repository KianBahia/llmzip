#!/bin/zsh
# Finder Quick Action "Decompress with llmzip": restore each selected .lz as a copy
# (NAME.restored.EXT, numbered if it exists) and report whether it is IDENTICAL to the
# original when the original is still next to it.
source "${0:A:h}/common.sh"

log "decompress action: $# item(s)"
for f in "$@"; do
  name=${f:t}
  if [[ $f != *.lz || ! -f $f ]]; then
    notify "llmzip ✗ $name" "Not a .lz file; only .lz files can be decompressed."
    log "skip non-.lz $f"; continue
  fi
  notify "llmzip" "Decompressing $name… you'll get a notification when it's done."
  run_llmzip "${f:A:h}" decompress "$name" --restored --unique
  if (( RC == 0 )); then
    lines=("${(@f)OUT}")
    summary=${(M)lines:#decompress:*}
    identity=${(M)lines:#(Original vs restored:*|Original not present*)}
    title="llmzip ✓ restored $name"
    [[ $identity == *DIFFERENT* ]] && title="llmzip ⚠ restored $name, but DIFFERENT from the original"
    notify "$title" "${identity}"$'\n'"${summary#decompress: }"
  else
    notify "llmzip ✗ $name" "$(last_line)"
  fi
done
