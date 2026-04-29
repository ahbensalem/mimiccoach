#!/usr/bin/env bash
# Cron-callable warm-keep for the Modal endpoint.
# Hits /healthz so the container stays warm during demo windows.
set -euo pipefail

URL="${MIMICCOACH_BACKEND_URL:-}"
if [ -z "$URL" ]; then
  echo "Set MIMICCOACH_BACKEND_URL to your Modal /healthz endpoint." >&2
  exit 2
fi

curl --max-time 30 --silent --show-error --fail "$URL" >/dev/null
echo "warm: ok @ $(date -u +%FT%TZ)"
