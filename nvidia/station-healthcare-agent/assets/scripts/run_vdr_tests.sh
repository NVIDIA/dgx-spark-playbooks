#!/bin/bash
export NODE_OPTIONS="--use-env-proxy"
# SECURITY: Disables TLS certificate verification for Node.js HTTP calls.
# Required only for local sandbox testing where the FHIR proxy uses self-signed
# certs. Do NOT carry this into production. Scoped to this script's process tree.
if [ "${LOCAL_TESTING:-1}" = "1" ]; then
  export NODE_TLS_REJECT_UNAUTHORIZED=0
fi
export PATH="/sandbox/.venv/bin:$PATH"

find ~/.openclaw -name '*.lock' -delete 2>/dev/null
rm -f /tmp/vdr*.out /tmp/t*start /tmp/t*end /tmp/*.py /tmp/chart*.png

echo "=== VDR1: Cohort ==="
date +%s > /tmp/t1start
openclaw agent --local --session-id t1 --thinking off --timeout 600 \
  --message "Find all diabetic patients, get their latest A1c and medications. Identify gap patients with A1c above 9 percent not on insulin or GLP-1. Show the A1c distribution as a histogram." \
  > /tmp/vdr1.out 2>&1
date +%s > /tmp/t1end
echo "VDR1 done: $(( $(cat /tmp/t1end) - $(cat /tmp/t1start) ))s"

echo "=== VDR2: Case Summary ==="
date +%s > /tmp/t2start
openclaw agent --local --session-id t2 --thinking off --timeout 600 \
  --message "Look up the first patient. Compile a case summary: demographics, conditions, recent labs, and medications." \
  > /tmp/vdr2.out 2>&1
date +%s > /tmp/t2end
echo "VDR2 done: $(( $(cat /tmp/t2end) - $(cat /tmp/t2start) ))s"

echo "=== VDR3: Cross-condition ==="
date +%s > /tmp/t3start
openclaw agent --local --session-id t3 --thinking off --timeout 600 \
  --message "Which patients have both diabetes and hypertension? For the overlap, get their latest HbA1c and blood pressure." \
  > /tmp/vdr3.out 2>&1
date +%s > /tmp/t3end
echo "VDR3 done: $(( $(cat /tmp/t3end) - $(cat /tmp/t3start) ))s"

echo "=== VDR4: Follow-up ==="
date +%s > /tmp/t4start
openclaw agent --local --session-id t4 --thinking off --timeout 600 \
  --message "Find all diabetic patients. Print the count and list their IDs." \
  > /tmp/vdr4.out 2>&1
date +%s > /tmp/t4end
echo "VDR4 done: $(( $(cat /tmp/t4end) - $(cat /tmp/t4start) ))s"

echo "=== ALL DONE ==="
touch /tmp/vdr-done
