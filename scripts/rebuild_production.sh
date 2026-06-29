#!/usr/bin/env bash
#
# rebuild_production.sh — full clean rebuild of production bsos.db
#                         (building_domain-zmw, epic building_domain-ufy)
#
# WHY: the production bsos.db was built incrementally BEFORE the prompt-level
# guardrails (8tk quality guidance, 5if scope note) and the e9k activity-dedup
# code fix landed. Those guardrails are cache-keyed / post-Pass-5, so they only
# take effect on a FRESH extraction. Re-running from scratch reproduces a clean
# DB by construction (proven by scripts/smoke_clean_run.sh on a small seed).
#
# This script does the FULL-SCALE rebuild into a STAGING database and runs every
# acceptance check from building_domain-zmw. It NEVER touches your live bsos.db.
# Promotion (backup + swap + snapshot) is a separate, human-gated subcommand you
# run only after the acceptance checks pass and you have eyeballed the warnings.
#
# Must run in a real terminal with the DeepSeek key set (cannot run inside a
# Claude Code session — the API key cannot be exported there):
#
#     export OPENAI_API_KEY=sk-<deepseek-key>
#     export OPENAI_BASE_URL=https://api.deepseek.com
#     ./scripts/rebuild_production.sh build      # long run: full 12-pass extract
#     ./scripts/rebuild_production.sh verify     # re-run acceptance checks only
#     ./scripts/rebuild_production.sh promote     # swap staging -> bsos.db (gated)
#
# claude-haiku-4-5-20251001 is a quality-equivalent fallback:
#     export ANTHROPIC_API_KEY=sk-ant-<key>
#     MODEL=claude-haiku-4-5-20251001 ./scripts/rebuild_production.sh build
#
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-deepseek-chat}"
STAGE="${STAGE:-bsos_rebuild.db}"   # *.db is git-ignored — never committed
LIVE="bsos.db"

# ---------------------------------------------------------------------------
sql() { sqlite3 "$STAGE" "$1"; }

run_acceptance() {
  [[ -f "$STAGE" ]] || { echo "ERROR: staging DB $STAGE not found — run 'build' first."; exit 1; }

  echo
  echo "================ building_domain-zmw ACCEPTANCE ($STAGE) ================"
  local fail=0

  echo "--- bsos doctor ---"
  if bsos doctor --db "$STAGE" 2>&1 | tee /tmp/zmw_doctor.txt | grep -qiE '\bFAIL\b'; then
    echo "(1) bsos doctor: has FAIL  -> FAIL"; fail=1
  else
    echo "(1) bsos doctor: 0 FAIL  -> ok"
  fi

  local unnorm
  unnorm=$(sql "SELECT count(*) FROM patterns WHERE status!='deprecated' AND (
      (force_descriptions NOT IN ('','[]') AND force_ids IN ('','[]'))
   OR (related_pattern_names NOT IN ('','[]') AND related_pattern_ids IN ('','[]')));")
  echo "(2) un-normalized patterns (expect 0): $unnorm"
  [[ "$unnorm" == "0" ]] || { echo "    FAIL"; fail=1; }

  local dangling
  dangling=$(sql "SELECT
     (SELECT count(*) FROM patterns p WHERE p.status!='deprecated'
        AND EXISTS(SELECT 1 FROM entities e WHERE e.id=p.subject_id AND e.status IN ('merged','deprecated')))
    +(SELECT count(*) FROM constraints c WHERE c.status!='deprecated'
        AND EXISTS(SELECT 1 FROM entities e WHERE e.id=c.subject_id AND e.status IN ('merged','deprecated')));")
  echo "(3) live patterns/constraints referencing merged/deprecated subjects (expect 0): $dangling"
  [[ "$dangling" == "0" ]] || { echo "    FAIL"; fail=1; }

  local phantom
  phantom=$(sql "SELECT count(*) FROM pass_progress pp WHERE pp.pass_number='3'
      AND NOT EXISTS(SELECT 1 FROM assertions a WHERE a.subject_id=pp.entity_id);")
  echo "(4) phantom pass-3 progress records (expect 0): $phantom"
  [[ "$phantom" == "0" ]] || { echo "    FAIL"; fail=1; }

  local exactdup
  exactdup=$(sql "SELECT count(*) FROM (
      SELECT lower(trim(name)) ln, count(*) c FROM entities
       WHERE entity_type='activity' AND status NOT IN ('merged','deprecated')
       GROUP BY ln HAVING c>1);")
  echo "(5) activity exact-string near-duplicate name groups (expect 0): $exactdup"
  [[ "$exactdup" == "0" ]] || { echo "    FAIL"; fail=1; }

  # Pattern-count distribution must NOT be spiked at exactly 8 (the 8tk debt was 97%).
  local spike
  spike=$(sql "SELECT CASE WHEN count(*)=0 THEN '0.0'
      ELSE printf('%.3f', cast(sum(CASE WHEN n=8 THEN 1 ELSE 0 END) AS float)/count(*)) END
      FROM (SELECT subject_id, count(*) n FROM patterns WHERE status!='deprecated' GROUP BY subject_id);")
  echo "(6) fraction of entities with exactly 8 patterns (expect well below 0.5): $spike"
  awk "BEGIN{exit !($spike > 0.5)}" && { echo "    FAIL: distribution still spiked at 8"; fail=1; }

  # Domain-drift activities: 5if excludes OFF-SITE MANUFACTURING and consumer
  # objects, NOT on-site construction. Keep keywords high-precision: bare
  # 'furniture'/'cushion' false-flag legit "Furniture Installation", and
  # 'fabricat' false-flags legit trades (steel/rebar/ductwork fabrication).
  # The reliable drift signals are off-site manufacturing and composting.
  local DRIFT_FILTER="lower(name) LIKE '%manufactur%' OR lower(name) LIKE '%paintbrush%'
        OR lower(name) LIKE '%compost%'"
  local drift
  drift=$(sql "SELECT count(*) FROM entities WHERE entity_type='activity' AND status NOT IN ('merged','deprecated')
      AND ($DRIFT_FILTER);")
  echo "(7) domain-drift activity keyword hits (expect 0): $drift"
  if [[ "$drift" != "0" ]]; then
    echo "    FAIL — offending names (off-site manufacturing / composting):"
    sql "SELECT '      '||name FROM entities WHERE entity_type='activity' AND status NOT IN ('merged','deprecated')
      AND ($DRIFT_FILTER);"
    fail=1
  fi

  echo
  echo "--- manual review (zmw requires eyeballing these) ---"
  echo "patterns-per-entity histogram:"
  sql "SELECT n AS patterns, count(*) AS entities FROM
        (SELECT subject_id, count(*) n FROM patterns WHERE status!='deprecated' GROUP BY subject_id)
       GROUP BY n ORDER BY n;"
  echo "live activity count:"
  sql "SELECT count(*) FROM entities WHERE entity_type='activity' AND status NOT IN ('merged','deprecated');"

  echo "========================================================================"
  if [[ "$fail" == "0" ]]; then
    echo "RESULT: PASS — all automated acceptance checks green."
    echo "        Review the histogram + activity list above, then:"
    echo "            ./scripts/rebuild_production.sh promote"
  else
    echo "RESULT: FAIL — do NOT promote. Inspect the failing check(s) above."
  fi
  return "$fail"
}

cmd_build() {
  echo "== FULL REBUILD  model=$MODEL  staging=$STAGE  (live $LIVE untouched) =="
  rm -f "$STAGE" "$STAGE"-* 2>/dev/null || true

  if [[ "$MODEL" == "deepseek-chat" ]]; then
    python scripts/smoke_test_deepseek.py   # provider health gate (exit 0 = HEALTHY)
  fi

  # Documented clean-by-construction procedure (CLAUDE.md), full production seed.
  bsos init             --db "$STAGE"
  bsos seed-ifc-classes --db "$STAGE"
  bsos seed-psets       --db "$STAGE"
  bsos extract          --db "$STAGE" --seed-apl --models "$MODEL" \
      --passes 1,2,3,4,5,6,7,8,9,10,11,12 --framings 1 --workers 2

  run_acceptance
}

cmd_promote() {
  [[ -f "$STAGE" ]] || { echo "ERROR: staging DB $STAGE not found."; exit 1; }
  echo "Re-running acceptance before promotion..."
  if ! run_acceptance; then
    echo "ABORTING promote: acceptance checks did not pass."; exit 1
  fi
  local ts backup
  ts=$(date +%Y%m%d-%H%M%S)
  backup="${LIVE}.bak-${ts}"
  echo
  echo "Promoting $STAGE -> $LIVE  (old live -> $backup)"
  [[ -f "$LIVE" ]] && cp -a "$LIVE" "$backup" && echo "  backed up live DB to $backup"
  cp -a "$STAGE" "$LIVE"
  echo "./$LIVE" > .bsos_config
  echo "  .bsos_config now points at ./$LIVE"

  echo "Regenerating ALL data/snapshot/*.json from the new DB..."
  bsos export --format json --output data/snapshot/

  echo
  echo "DONE. Next (commit the tracked snapshot form):"
  echo "    git add data/snapshot/"
  echo "    git commit -m 'snapshot: clean rebuild of production bsos.db (zmw)'"
  echo "    git pull --rebase && git push"
  echo "    bd close building_domain-zmw building_domain-ufy"
  echo "Keep $backup until the rebuild is accepted (building_domain-lfy)."
}

case "${1:-build}" in
  build)   cmd_build ;;
  verify)  run_acceptance ;;
  promote) cmd_promote ;;
  *) echo "usage: $0 {build|verify|promote}"; exit 2 ;;
esac
