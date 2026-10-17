#!/bin/zsh
# Install (or remove) the llmzip Finder Quick Actions:
#   "Compress with llmzip", "Decompress with llmzip", "Inspect with llmzip".
#
#   finder/install.sh                 install into ~/Library/Services
#   finder/install.sh --uninstall     remove them
#   finder/install.sh --dest DIR      write/remove the bundles in DIR instead (testing)
#
# The absolute paths of uv and of this repo are baked into each Quick Action, because
# Finder runs them with a minimal PATH. Rerun this after moving the repo.
set -e

REPO=${0:A:h:h}
DEFAULT_DEST=$HOME/Library/Services
DEST=$DEFAULT_DEST
UNINSTALL=0
while (( $# )); do
  case $1 in
    --uninstall) UNINSTALL=1 ;;
    --dest) shift; DEST=${1:?--dest needs a directory} ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) print -u2 "unknown option: $1"; exit 2 ;;
  esac
  shift
done
DEST=${DEST:A}

NAMES=("Compress with llmzip" "Decompress with llmzip" "Inspect with llmzip")
SCRIPTS=(llmzip-compress.sh llmzip-decompress.sh llmzip-inspect.sh)

if (( UNINSTALL )); then
  for name in $NAMES; do
    bundle="${DEST:?}/$name.workflow"
    if [[ -d $bundle ]]; then rm -rf -- "$bundle"; print "removed $bundle"; fi
  done
  [[ $DEST == ${DEFAULT_DEST:A} ]] && /System/Library/CoreServices/pbs -flush 2>/dev/null || true
  exit 0
fi

UV=${commands[uv]:-}
if [[ -z $UV ]]; then
  print -u2 "error: uv not found on PATH; install it (brew install uv) and rerun."; exit 1
fi
UV=${UV:a}  # absolute, but keep the symlink (/opt/homebrew/bin/uv survives brew upgrades)
chmod +x "$REPO"/finder/*.sh
mkdir -p -- "$DEST"

for i in {1..3}; do
  script="$REPO/finder/${SCRIPTS[i]}"
  command="LLMZIP_UV=${(qq)UV} LLMZIP_REPO=${(qq)REPO} ${(qq)script} \"\$@\""
  bundle=$("$UV" run --project "$REPO" --frozen --quiet python "$REPO/finder/make_workflow.py" \
    --dest "$DEST" --name "${NAMES[i]}" --command "$command")
  plutil -lint -s "$bundle/Contents/Info.plist" "$bundle/Contents/document.wflow"
  print "installed $bundle"
done

if [[ $DEST == ${DEFAULT_DEST:A} ]]; then
  /System/Library/CoreServices/pbs -flush 2>/dev/null || true
  print "Done. Right-click a file in Finder → Quick Actions (or Services) to use them."
fi
