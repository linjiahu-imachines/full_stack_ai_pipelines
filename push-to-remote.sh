#!/usr/bin/env bash
# Stage, commit (if needed), and push monorepo changes to GitHub.
#
# Usage:
#   ./push-to-remote.sh
#   ./push-to-remote.sh "Fix vLLM CPU test script"
#
# Environment overrides:
#   REMOTE=origin   BRANCH=main

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-$(git branch --show-current)}"
COMMIT_MSG="${*:-Update $(date '+%Y-%m-%d %H:%M')}"

if ! git remote get-url "$REMOTE" &>/dev/null; then
  echo "error: remote '$REMOTE' is not configured." >&2
  echo "  git remote add origin https://github.com/linjiahu-imachines/full_stack_ai_pipelines.git" >&2
  exit 1
fi

if ! git config user.email &>/dev/null || ! git config user.name &>/dev/null; then
  echo "error: set git identity before committing:" >&2
  echo "  git config --global user.name \"Your Name\"" >&2
  echo "  git config --global user.email \"you@example.com\"" >&2
  exit 1
fi

echo "==> Repository: $REPO_ROOT"
echo "==> Remote:     $REMOTE ($(git remote get-url "$REMOTE"))"
echo "==> Branch:     $BRANCH"
echo

if [ -n "$(git status --porcelain)" ]; then
  echo "==> Staging changes..."
  git add -A
  git status --short
  echo
  echo "==> Committing: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG"
else
  echo "==> No uncommitted changes."
fi

echo
echo "==> Pushing to $REMOTE/$BRANCH ..."
git push -u "$REMOTE" "$BRANCH"

echo
echo "==> Done. https://github.com/linjiahu-imachines/full_stack_ai_pipelines"
