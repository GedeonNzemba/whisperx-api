#!/usr/bin/env bash
# test_deployment.sh — Validate a running WhisperX API deployment.
#
# Usage:
#   bash test_deployment.sh <BASE_URL> [TEST_AUDIO]
#
# Examples:
#   bash test_deployment.sh https://XXXXXX-8000.proxy.runpod.net
#   bash test_deployment.sh http://localhost:8000 ./audio.m4a

set -euo pipefail

BASE_URL="${1:?Usage: $0 <BASE_URL> [TEST_AUDIO]}"
BASE_URL="${BASE_URL%/}"  # strip trailing slash
TEST_AUDIO="${2:-}"

PASS=0
FAIL=0
ERRORS=()

GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"

pass() { echo -e "${GREEN}  PASS${RESET} — $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}  FAIL${RESET} — $1"; FAIL=$((FAIL + 1)); ERRORS+=("$1"); }

echo ""
echo "================================================"
echo "  WhisperX API Deployment Validation"
echo "  Target: ${BASE_URL}"
echo "================================================"
echo ""

# ── Test 1: GET /health ──────────────────────────────────────────────────────
echo "[ 1/3 ] GET /health"
HEALTH_RESP=$(curl -sf --max-time 15 "${BASE_URL}/health" 2>&1) || {
    fail "GET /health — connection failed or returned non-2xx"
    echo "       Error: ${HEALTH_RESP}"
    # Can't continue without a running server
    echo ""
    echo "Results: ${PASS} passed, ${FAIL} failed."
    exit 1
}

WHISPER_LOADED=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('whisper_loaded',''))" 2>/dev/null || echo "")
STATUS=$(echo "$HEALTH_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "")

if [ "$STATUS" = "ok" ]; then
    pass "GET /health → status=ok"
else
    fail "GET /health → expected status=ok, got: ${STATUS}"
fi

if [ "$WHISPER_LOADED" = "True" ] || [ "$WHISPER_LOADED" = "true" ]; then
    pass "GET /health → whisper_loaded=true"
else
    fail "GET /health → whisper_loaded is not true (got: '${WHISPER_LOADED}'). Model may still be loading."
fi

echo "       Full response: ${HEALTH_RESP}"
echo ""

# ── Prepare test audio ───────────────────────────────────────────────────────
TMP_AUDIO=""
CLEANUP_AUDIO=0

if [ -z "$TEST_AUDIO" ]; then
    # Generate a tiny silent WAV (1 second, 16kHz, mono) using Python
    echo "  [info] No test audio provided — generating a 1-second silent WAV..."
    TMP_AUDIO=$(mktemp /tmp/test_audio_XXXX.wav)
    CLEANUP_AUDIO=1
    python3 - <<'PYEOF'
import wave, struct, os, sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_audio.wav"
samples = [0] * 16000  # 1 second of silence at 16kHz
with wave.open(os.environ.get("TMP_AUDIO", "/tmp/test_audio.wav"), "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
PYEOF
    # Re-generate using shell since PYEOF env approach is cleaner this way
    python3 -c "
import wave, struct
samples = [0] * 16000
with wave.open('${TMP_AUDIO}', 'w') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
"
    TEST_AUDIO="$TMP_AUDIO"
elif [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${RED}  [warn]${RESET} Test audio file not found: ${TEST_AUDIO}"
    echo "         Generating silent WAV instead..."
    TMP_AUDIO=$(mktemp /tmp/test_audio_XXXX.wav)
    CLEANUP_AUDIO=1
    python3 -c "
import wave, struct
samples = [0] * 16000
with wave.open('${TMP_AUDIO}', 'w') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
"
    TEST_AUDIO="$TMP_AUDIO"
fi

echo "  [info] Using audio: ${TEST_AUDIO}"
echo ""

# ── Test 2: POST /transcribe ─────────────────────────────────────────────────
echo "[ 2/3 ] POST /transcribe"
TRANSCRIBE_RESP=$(curl -sf --max-time 120 \
    -X POST "${BASE_URL}/transcribe" \
    -F "audio=@${TEST_AUDIO}" \
    -F "language=en" \
    2>&1) || {
    fail "POST /transcribe — request failed"
    echo "       Error: ${TRANSCRIBE_RESP}"
    TRANSCRIBE_RESP=""
}

if [ -n "$TRANSCRIBE_RESP" ]; then
    SEGMENTS=$(echo "$TRANSCRIBE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(type(d.get('segments','')).__name__)" 2>/dev/null || echo "")
    if [ "$SEGMENTS" = "list" ]; then
        pass "POST /transcribe → returned segments array"
    else
        fail "POST /transcribe → expected 'segments' list in response"
        echo "       Response: ${TRANSCRIBE_RESP}"
    fi
fi
echo ""

# ── Test 3: POST /align ──────────────────────────────────────────────────────
echo "[ 3/3 ] POST /align"
ALIGN_RESP=$(curl -sf --max-time 120 \
    -X POST "${BASE_URL}/align" \
    -F "audio=@${TEST_AUDIO}" \
    -F "transcript=test" \
    -F "language=en" \
    2>&1) || {
    fail "POST /align — request failed"
    echo "       Error: ${ALIGN_RESP}"
    ALIGN_RESP=""
}

if [ -n "$ALIGN_RESP" ]; then
    WORD_SEGS=$(echo "$ALIGN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(type(d.get('word_segments','')).__name__)" 2>/dev/null || echo "")
    if [ "$WORD_SEGS" = "list" ]; then
        pass "POST /align → returned word_segments array"
    else
        fail "POST /align → expected 'word_segments' list in response"
        echo "       Response: ${ALIGN_RESP}"
    fi
fi
echo ""

# ── Cleanup ──────────────────────────────────────────────────────────────────
if [ "$CLEANUP_AUDIO" = "1" ] && [ -n "$TMP_AUDIO" ]; then
    rm -f "$TMP_AUDIO"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed checks:"
    for ERR in "${ERRORS[@]}"; do
        echo -e "    ${RED}✗${RESET} ${ERR}"
    done
fi
echo "================================================"
echo ""

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
