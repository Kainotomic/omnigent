#!/bin/sh
# Claude Code `apiKeyHelper`: prints the CLIProxy gateway key to stdout.
# Claude sends it as both `x-api-key` and `Authorization: Bearer`.
# OMNIGENT_GATEWAY_API_KEY reaches the harness via OMNIGENT_RUNNER_ENV_PASSTHROUGH
# (set by the entrypoint). No OPENAI_API_KEY fallback on purpose.
if [ -n "${OMNIGENT_GATEWAY_API_KEY:-}" ]; then
  printf '%s' "$OMNIGENT_GATEWAY_API_KEY"
else
  echo "omnigent-gateway-key: OMNIGENT_GATEWAY_API_KEY is not set" >&2
  exit 1
fi
