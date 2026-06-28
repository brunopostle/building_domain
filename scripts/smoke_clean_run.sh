#!/usr/bin/env bash
#
# smoke_clean_run.sh — clean-by-construction verification (epic building_domain-27w, acceptance (c))
#
# Runs the documented from-empty-DB extraction procedure on a small seed against a
# THROWAWAY database, then verifies the guardrails held with zero manual SQL.
# Your production bsos.db is never touched.
#
# Must run in a real terminal with a provider key set (cannot run inside a Claude
# Code session). Pick ONE provider before running:
#
#   DeepSeek (documented production model):
#     export OPENAI_API_KEY=sk-<deepseek-key>
#     export OPENAI_BASE_URL=https://api.deepseek.com
#     MODEL=deepseek-chat ./scripts/smoke_clean_run.sh
#
#   Anthropic Haiku (quality-equivalent fallback):
#     export ANTHROPIC_API_KEY=sk-ant-<key>
#     MODEL=claude-haiku-4-5-20251001 ./scripts/smoke_clean_run.sh
#
set -euo pipefail

MODEL="${MODEL:-deepseek-chat}"
DB="${DB:-/tmp/bsos_smoke.db}"
SEED="$(mktemp /tmp/bsos_smoke_seed.XXXX.txt)"

# One concept per type (component/space/material/activity) keeps Pass 1 bounded.
printf 'Wall\nWindow\nRoof\nConcrete\nCorridor\nConcrete Pouring\n' > "$SEED"

echo "== model=$MODEL  db=$DB  seed=$SEED =="
rm -f "$DB" "$DB"-* 2>/dev/null || true

# DeepSeek provider health check (no-op for Anthropic).
if [[ "$MODEL" == "deepseek-chat" ]]; then
  python scripts/smoke_test_deepseek.py
fi

# --- Documented clean-by-construction procedure ---
bsos init --db "$DB"
bsos seed-ifc-classes --db "$DB"
bsos seed-psets --db "$DB"
bsos extract --db "$DB" --seed "$SEED" --models "$MODEL" \
  --passes 1,2,3,4,5,6,7,8,9,10,11,12 \
  --framings 1 --workers 2

# --- Verification (acceptance (c)): all checks must pass with NO manual SQL ---
echo
echo "================ VERIFICATION ================"
bsos doctor --db "$DB" || true
bsos status --db "$DB" || true

fail=0

hallu=$(sqlite3 "$DB" "SELECT count(*) FROM entities WHERE entity_type='ifc_class' AND source_model!='ifc-schema';")
echo "(a) hallucinated ifc_class rows (expect 0): $hallu"
[[ "$hallu" == "0" ]] || { echo "  FAIL"; fail=1; }

echo "(b) patterns per entity (should vary, not all be exactly 8):"
sqlite3 "$DB" "SELECT subject_id, count(*) FROM patterns GROUP BY subject_id;"

unnorm=$(sqlite3 "$DB" "SELECT count(*) FROM patterns
  WHERE (force_descriptions NOT IN ('','[]') AND force_ids IN ('','[]'))
     OR (related_pattern_names NOT IN ('','[]') AND related_pattern_ids IN ('','[]'));")
echo "(c) un-normalized patterns (expect 0): $unnorm"
[[ "$unnorm" == "0" ]] || { echo "  FAIL"; fail=1; }

echo "(d) activities (should be on-site construction only, no off-site manufacture):"
sqlite3 "$DB" "SELECT name FROM entities WHERE entity_type='activity';"

echo "=============================================="
if [[ "$fail" == "0" ]]; then
  echo "RESULT: PASS — clean-by-construction verified. Review (b)/(d) by eye, then"
  echo "        close building_domain-27w and building_domain-efo."
else
  echo "RESULT: FAIL — a guardrail leaked; do NOT close the epic. Inspect above."
fi
echo "Cleanup when done:  rm -f $DB ${DB}-* $SEED"
exit "$fail"
