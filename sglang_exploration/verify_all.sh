#!/bin/bash
# Layout verification for sglang_exploration (extend with real checks later).

set -euo pipefail

echo "════════════════════════════════════════════════════════════"
echo "  SGLang exploration — scaffold verification"
echo "════════════════════════════════════════════════════════════"
echo ""

SGLANG_ROOT="/home/linhu/projects/sglang_exploration"
ERRORS=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "1. Checking main folder..."
if [ -d "$SGLANG_ROOT" ]; then
  echo -e "   ${GREEN}✓${NC} Main folder exists: $SGLANG_ROOT"
else
  echo -e "   ${RED}✗${NC} Main folder not found: $SGLANG_ROOT"
  ERRORS=$((ERRORS + 1))
fi

echo ""
echo "2. Checking subfolders..."
for d in sglang_test sglang_cpu_test sglang_gpu_test docs; do
  if [ -d "$SGLANG_ROOT/$d" ]; then
    echo -e "   ${GREEN}✓${NC} $d/"
  else
    echo -e "   ${RED}✗${NC} missing: $d/"
    ERRORS=$((ERRORS + 1))
  fi
done

echo ""
echo "3. Checking key files..."
for f in README.md .gitignore verify_all.sh docs/README.md docs/QUICKSTART.md; do
  if [ -f "$SGLANG_ROOT/$f" ]; then
    echo -e "   ${GREEN}✓${NC} $f"
  else
    echo -e "   ${RED}✗${NC} missing: $f"
    ERRORS=$((ERRORS + 1))
  fi
done

echo ""
if [ "$ERRORS" -eq 0 ]; then
  echo -e "${GREEN}All checks passed.${NC}"
  exit 0
else
  echo -e "${RED}$ERRORS check(s) failed.${NC}"
  exit 1
fi
