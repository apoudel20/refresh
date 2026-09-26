#!/bin/zsh
# Usage: aux/iterate.sh NAME [extra build args]  -> render + patch report + silhouette rows
cd "$(dirname "$0")/.."
name=$1; shift
rm -f aux/renders/$name.png
for attempt in 1 2 3; do
  blender -b --factory-startup -P aux/build_dog.py -- --nosave --cam photo --strands ${STRANDS:-450000} --samples ${SAMPLES:-64} --pct 50 --render aux/renders/$name.png "$@" 2>&1 | grep -E "Error|Traceback|line [0-9]"
  [ -f aux/renders/$name.png ] && break
done
TOPN=${TOPN:-15} blender -b --factory-startup -P aux/patch_compare.py -- aux/renders/$name.png aux/renders/$name 2>&1 | grep -E "PATCH|rank|^ +[0-9]+  \("
[ -n "$ROWS" ] && blender -b --factory-startup -P aux/silhouette_rows.py -- aux/renders/$name.png 2>&1 | grep -E "^ *[0-9]+ |top"
