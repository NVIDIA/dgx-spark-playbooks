#!/usr/bin/env bash
# Probe vLLM with a real completion; restart it only if it is genuinely wedged.
set -uo pipefail

URL=http://127.0.0.1:8000/v1/chat/completions
ENV_FILE=/etc/vllm/vllm.env          # holds VLLM_API_KEY=...
STATE=/var/lib/vllm-watchdog/state
PROBE_TIMEOUT=30
MIN_UPTIME=600                       # a loading model is not a wedged one
RESTART_COOLDOWN=2700                # 45 min hard floor between restarts
FAILS_BEFORE_RESTART=2               # one slow moment is not a wedge

say() { logger -t vllm-watchdog "$1"; echo "$1"; }
FAILS=0; LAST_RESTART=0
[ -f "$STATE" ] && . "$STATE" 2>/dev/null || true
save() { printf 'FAILS=%s\nLAST_RESTART=%s\nLAST_RESULT=%s\n' \
         "$FAILS" "$LAST_RESTART" "$1" > "$STATE"; }

# Derive the target. A hardcoded unit name silently disarms the watchdog the
# first time you swap models: the rail below sees an inactive unit, exits 0,
# and records "skipped" forever while nothing is guarded.
UNIT=$(systemctl list-units --state=active --no-legend 'vllm-*.service' \
       | awk '{print $1}' | head -1)
[ -n "${UNIT:-}" ] || { save "skipped: no vllm unit active"; exit 0; }

START=$(date -d "$(systemctl show "$UNIT" -p ActiveEnterTimestamp --value)" +%s)
NOW=$(date +%s)
[ $((NOW - START)) -lt "$MIN_UPTIME" ] && { save "skipped: still loading"; exit 0; }

# The key comes first: everything below, including asking which model is
# served, is an authenticated call.
KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$ENV_FILE" | tr -d "\"'" | head -1)
if [ -z "${KEY:-}" ]; then
    say "NO KEY in $ENV_FILE - refusing to probe; a restart cannot fix this"
    save "error: no api key"; exit 1
fi

# Take the FIRST "id" and nothing else — /v1/models also carries a permission
# object whose id a greedy match will return instead.
MODELS=$(curl -s -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/v1/models)
if echo "$MODELS" | grep -qi unauthorized; then
    say "the key in $ENV_FILE is not accepted; NOT restarting"
    save "error: unauthorized (not a wedge)"; exit 1
fi
MODEL=$(echo "$MODELS" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[ -n "${MODEL:-}" ] || { say "could not read the served model name"; save "error: no model"; exit 1; }

BODY=$(curl -s -m "$PROBE_TIMEOUT" "$URL" \
       -H 'Content-Type: application/json' \
       -H "Authorization: Bearer $KEY" \
       -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8,\"temperature\":0}")

if echo "$BODY" | grep -q '"choices"'; then
    [ "$FAILS" -ne 0 ] && say "recovered after $FAILS failed probe(s)"
    FAILS=0; save "ok ($MODEL)"; exit 0
fi

# A rejected key is a configuration fault and must never count toward the
# restart threshold. This is the branch whose absence caused the outages.
if echo "$BODY" | grep -qi unauthorized; then
    say "PROBE UNAUTHORIZED on $UNIT - the key is not accepted; NOT restarting"
    save "error: unauthorized (not a wedge)"; exit 1
fi

FAILS=$((FAILS + 1))
say "PROBE FAILED ($FAILS/$FAILS_BEFORE_RESTART) on $UNIT: no completion in ${PROBE_TIMEOUT}s"
if [ "$FAILS" -ge "$FAILS_BEFORE_RESTART" ]; then
    SINCE=$((NOW - LAST_RESTART))
    if [ "$SINCE" -lt "$RESTART_COOLDOWN" ]; then
        say "WEDGED but restarted ${SINCE}s ago - NOT restarting again; investigate"
        save "wedged: in cooldown"; exit 1
    fi
    say "restarting $UNIT"
    systemctl restart "$UNIT"
    FAILS=0; LAST_RESTART=$NOW; save "restarted $UNIT"
fi
save "failed ($FAILS)"
exit 1
