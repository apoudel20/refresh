#!/usr/bin/env bash
# Install render-eval: the `render-eval` command and the render-eval agent skill.
#
#   ./install.sh <project-dir>              command on PATH + skill in <project>/.claude/skills/render-eval
#   ./install.sh --user                     command on PATH + skill in ~/.claude/skills/render-eval (all projects)
#   ./install.sh <project-dir> --no-tool    skill files only (the command is already installed)
#   ./install.sh --uninstall <project-dir>  remove the skill from the project (and the command)
#   ./install.sh --uninstall --user
#
# Requires uv (https://docs.astral.sh/uv/). The first install downloads PyTorch and friends
# (about 1 GB); the first eval run downloads about 3 GB of model weights into ~/.cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL=render-eval
TARGET="" USER_SCOPE=0 TOOL=1 UNINSTALL=0

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

for arg in "$@"; do
  case "$arg" in
    --user) USER_SCOPE=1 ;;
    --no-tool) TOOL=0 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $arg" >&2; usage 1 ;;
    *) [ -z "$TARGET" ] && TARGET="$arg" || { echo "only one project dir, got '$TARGET' and '$arg'" >&2; exit 1; } ;;
  esac
done

if [ "$USER_SCOPE" = 1 ]; then
  SKILLS_DIR="$HOME/.claude/skills"
else
  [ -n "$TARGET" ] || usage 1
  [ -d "$TARGET" ] || { echo "not a directory: $TARGET" >&2; exit 1; }
  SKILLS_DIR="$(cd "$TARGET" && pwd)/.claude/skills"
fi
DEST="$SKILLS_DIR/$SKILL"

if [ "$UNINSTALL" = 1 ]; then
  rm -rf "$DEST"
  [ "$TOOL" = 1 ] && command -v uv >/dev/null && uv tool uninstall render-eval >/dev/null 2>&1 || true
  echo "removed $DEST"
  exit 0
fi

# 1. The `render-eval` command, in its own environment managed by uv
if [ "$TOOL" = 1 ]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
  echo "installing the render-eval command (uv tool; the first time downloads PyTorch)..."
  uv tool install --quiet --force --reinstall "$HERE" >/dev/null
  BIN="$(uv tool dir --bin)"
  case ":$PATH:" in *":$BIN:"*) ;; *) echo "  note: add $BIN to your PATH (or run: uv tool update-shell)" ;; esac
fi

# 2. The skill
mkdir -p "$SKILLS_DIR"
rm -rf "$DEST"
cp -R "$HERE/skills/$SKILL" "$DEST"
echo "skill   -> $DEST"

# 3. Readiness (informational)
if command -v render-eval >/dev/null; then
  echo "command -> $(command -v render-eval)"
else
  echo "command -> not on PATH yet (see the note above)"
fi
if [ -n "${OPENROUTER_API_KEY:-}" ] || { [ -n "$TARGET" ] && grep -qs '^OPENROUTER_API_KEY=' "$TARGET/.env"; }; then
  echo "openrouter key: found"
else
  echo "openrouter key: not found. The embedding and critic steps need OPENROUTER_API_KEY"
  echo "  (environment, or a .env in the folder you run render-eval from). See $HERE/.env.example"
fi
