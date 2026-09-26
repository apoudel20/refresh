#!/usr/bin/env bash
# Install the imagegen toolkit and the blender-atlas-retexture skill into a project.
#
#   ./install.sh <project-dir> [--mcp] [--codex] [--no-tool]
#   ./install.sh --uninstall <project-dir>
#
#   (default)  install `imagegen` + `imagegen-mcp` on PATH (uv tool) and copy the skill to
#              <project>/.claude/skills/blender-atlas-retexture
#   --mcp      also register the MCP server in <project>/.mcp.json
#   --codex    also copy the skill to ${CODEX_HOME:-~/.codex}/skills
#   --no-tool  skip the uv tool install (skill files only)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL=blender-atlas-retexture
TARGET="" MCP=0 CODEX=0 TOOL=1 UNINSTALL=0

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

for arg in "$@"; do
  case "$arg" in
    --mcp) MCP=1 ;;
    --codex) CODEX=1 ;;
    --no-tool) TOOL=0 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $arg" >&2; usage 1 ;;
    *) [ -z "$TARGET" ] && TARGET="$arg" || { echo "only one project dir, got '$TARGET' and '$arg'" >&2; exit 1; } ;;
  esac
done
[ -n "$TARGET" ] || usage 1
[ -d "$TARGET" ] || { echo "not a directory: $TARGET" >&2; exit 1; }
TARGET="$(cd "$TARGET" && pwd)"
CODEX_SKILLS="${CODEX_HOME:-$HOME/.codex}/skills"

# Merge or remove the imagegen entry in <project>/.mcp.json without touching other servers.
mcp_json() {  # $1 = add | remove
  python3 - "$TARGET/.mcp.json" "$1" <<'PY'
import json, os, sys
path, action = sys.argv[1], sys.argv[2]
data = json.load(open(path)) if os.path.exists(path) else {}
servers = data.setdefault("mcpServers", {})
if action == "add":
    servers["imagegen"] = {"command": "imagegen-mcp"}
else:
    servers.pop("imagegen", None)
json.dump(data, open(path, "w"), indent=2)
open(path, "a").write("\n")
PY
}

if [ "$UNINSTALL" = 1 ]; then
  rm -rf "$TARGET/.claude/skills/$SKILL"
  [ -f "$TARGET/.mcp.json" ] && mcp_json remove
  [ -d "$CODEX_SKILLS/$SKILL" ] && rm -rf "$CODEX_SKILLS/$SKILL"
  [ "$TOOL" = 1 ] && command -v uv >/dev/null && uv tool uninstall imagegen >/dev/null 2>&1 || true
  echo "removed imagegen skill from $TARGET"
  exit 0
fi

# 1. CLI + MCP server on PATH
if [ "$TOOL" = 1 ]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
  echo "installing imagegen commands (uv tool)..."
  if ! out="$(uv tool install --force --reinstall "$HERE" 2>&1)"; then echo "$out" >&2; exit 1; fi
  BIN="$(uv tool dir --bin)"
  case ":$PATH:" in *":$BIN:"*) ;; *) echo "  note: add $BIN to your PATH (or run: uv tool update-shell)";; esac
fi

# 2. The skill
mkdir -p "$TARGET/.claude/skills"
rm -rf "$TARGET/.claude/skills/$SKILL"
cp -R "$HERE/skills/$SKILL" "$TARGET/.claude/skills/$SKILL"
echo "skill  -> $TARGET/.claude/skills/$SKILL"

if [ "$CODEX" = 1 ]; then
  mkdir -p "$CODEX_SKILLS"
  rm -rf "$CODEX_SKILLS/$SKILL"
  cp -R "$HERE/skills/$SKILL" "$CODEX_SKILLS/$SKILL"
  echo "skill  -> $CODEX_SKILLS/$SKILL"
fi

if [ "$MCP" = 1 ]; then
  mcp_json add
  echo "mcp    -> $TARGET/.mcp.json (imagegen-mcp)"
fi

# 3. What's usable (informational, spends nothing)
echo
if command -v blender >/dev/null || [ -x /Applications/Blender.app/Contents/MacOS/Blender ] || [ -n "${BLENDER:-}" ]; then
  echo "blender: found"
else
  echo "blender: NOT found (needed for 'imagegen uv ...'; set BLENDER=/path/to/blender)"
fi
if command -v imagegen >/dev/null; then
  imagegen backends || true
else
  echo "imagegen not on PATH yet (see note above)"
fi
