# Kainotomic deployment overlay

Deployment-only additions on top of upstream `deploy/docker/`. Nothing here
changes upstream files; the server image is upstream's `runtime` target
unmodified, the workspace image is upstream's `host` target plus this overlay.

## Build (contabo or CI, never on the Dokploy control plane)

```sh
# from the repo root, at the frozen revision
docker build -f deploy/docker/Dockerfile --target runtime -t omnigent-server:b203ba4c .
docker build -f deploy/docker/Dockerfile --target host \
    --build-arg EXTRA_HARNESS_CLIS=opencode@1.18.30 -t omnigent-host-base:b203ba4c .
docker build -f deploy/kainotomic/Dockerfile.host -t omnigent-host:b203ba4c-kt10 .
```

`Dockerfile.host` pins `@anthropic-ai/claude-code`, `@openai/codex`,
`@earendil-works/pi-coding-agent`, `opencode-ai` (ARGs at the top), keeps
upstream's sha256-pinned `agy`, adds `tini` + `ripgrep`, and installs the
harness templates and entrypoint. `.github/workflows/kainotomic-publish-images.yml`
runs the same three builds and pushes to `ghcr.io/kainotomic/omnigent-server`
and `ghcr.io/kainotomic/omnigent-host` (`workflow_dispatch` or a
`kainotomic-v*` tag).

## Workspace runtime contract

`workspace-entrypoint.py` runs under `tini`, renders the templates in
`harness-templates/`, waits until a usable login exists at
`$OMNIGENT_DATA_DIR/auth_tokens.json` or `~/.omnigent/auth_tokens.json` (same
path as `omnigent/cli_auth.py`), then execs `omnigent host --server
"$OMNIGENT_SERVER_URL" --non-interactive`. Without tokens the process stays
up (logs every 30s) so `docker exec` login is safe; `restart: unless-stopped`
does not bounce. Tokens already present skip the wait.

| Env var | Purpose |
|---|---|
| `OMNIGENT_SERVER_URL` | Omnigent server the host daemon dials (required) |
| `OMNIGENT_GATEWAY_BASE_URL` | CLIProxy root; default `https://openai.kainotomic.com` |
| `OMNIGENT_GATEWAY_API_KEY` | This workspace's CLIProxy key. Env only; never written to disk |
| `OMNIGENT_GATEWAY_ANTHROPIC_MODEL` | Overrides the catalog's anthropic default (Claude Code, Pi, Omnigent anthropic family) |
| `OMNIGENT_GATEWAY_OPENAI_MODEL` | Overrides the catalog's openai default (Codex, Omnigent openai family) |
| `OMNIGENT_GATEWAY_FORCE` | `1` overwrites existing files in the persistent home |

### Model catalog

`harness-templates/gateway-models.jsonl` is the single source for every
per-harness model block: one CLIProxy model per line (`id`, `family`,
`contextWindow`, `maxTokens`, `input`, `reasoning`, `thinkingLevelMap` as
served by the gateway) plus the allocation fields `default` (one per family),
`claudeCode` (Claude Code alias tiers), `codexProfile`, and `opencodeDefault`. It is JSON Lines
rather than a `.json` template so a model is one diffable line and so the
repo's hardcoded-model lint (which covers `.json`/`.yaml`/`.toml`/`.py`/`.sh`)
keeps the code and templates id-free; the ids live only in this data file.
Add or re-tier a model there, rebuild, and every harness picks it up:

| Harness | What it gets from the catalog |
|---|---|
| Claude Code (`/etc/claude-code/managed-settings.json`) | anthropic family only: `model` + `ANTHROPIC_MODEL` = anthropic default; `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU,FABLE}_MODEL` and `CLAUDE_CODE_SUBAGENT_MODEL` from `claudeCode` tiers (`haiku` is the current name for the small/fast slot; `ANTHROPIC_SMALL_FAST_MODEL` is deprecated); `availableModels` + `modelPicker.replaceBuiltInOptions` + `enforceAvailableModels` pin New Chat and `/model` to those factory IDs (not Claude Code's built-in opus 4.x / `[1m]` catalog); `CLAUDE_CODE_DISABLE_1M_CONTEXT=1` drops the 1M twins |
| Codex (`~/.codex/config.toml` + `~/.codex/model_catalog.json` + `~/.codex/<codexProfile>.config.toml`) | openai default as `model` (+ `model_context_window`), `wire_api = "responses"`; `model_catalog_json` replaces Codex's bundled catalog with the openai default plus every `codexProfile` row so New Chat / `model/list` does not show built-ins or other cliproxy families; one profile file per `codexProfile` model, selected with `codex --profile <name>` (codex 0.154.0 layers `$CODEX_HOME/<name>.config.toml` over `config.toml`; the legacy `[profiles.*]` tables are no longer applied) |
| Pi (`~/.pi/agent/models.json`, `settings.json`) | all models under one `cliproxy` provider, `api: openai-responses`, with `contextWindow`/`maxTokens`/`thinkingLevelMap`/`input`; default = anthropic default |
| OpenCode (`~/.config/opencode/opencode.json`) | all models under `provider.cliproxy` (`@ai-sdk/openai-compatible`) with `limit.context`/`limit.output`, `reasoning`, `attachment`, `modalities`; `model = cliproxy/<opencodeDefault>` |
| Omnigent (`~/.omnigent/config.yaml`) | `providers.cliproxy.anthropic.models` = default + tiers, `openai.models` = default + the remaining models, `context_window`/`max_output_tokens` of each default |

`OMNIGENT_GATEWAY_{ANTHROPIC,OPENAI}_MODEL` only move the defaults; an id
outside the catalog is rendered without limits (logged as a warning).

On each start the entrypoint hashes `gateway-models.jsonl` and compares it
to `~/.omnigent/.kainotomic-catalog-hash`. A missing or different hash
re-renders catalog-driven fields (model lists, defaults, limits) in the
existing home files and leaves user-added keys alone. A matching hash
skips those files. `OMNIGENT_GATEWAY_FORCE=1` still overwrites.

Rendered files (created when absent; catalog-hash updates the model bits):

- `~/.codex/model_catalog.json` — Codex picker allowlist (openai default +
  `codexProfile` ids). Written on first boot or catalog-hash change; a
  matching hash skips it. The host probe copies `model` +
  `model_catalog_json` into its isolated `CODEX_HOME`.
- `~/.omnigent/config.yaml` — Omnigent's own `providers.cliproxy` (`kind:
  gateway`, `anthropic` + `openai` families, `api_key_ref:
  env:OMNIGENT_GATEWAY_API_KEY`, `default: [anthropic, openai, pi]`). This is
  what Omnigent readiness and the native adapters consume: with it present the
  adapters inject the provider per session (Claude `--settings`, Codex `-c
  model_provider`, Pi managed `models.json`) and forward the referenced env
  var to the runner. `omnigent host` also stores the host identity in this
  file, so FORCE replaces only `providers.cliproxy`.
- `/etc/claude-code/managed-settings.json` (always; readiness credits the
  `apiKeyHelper`), `~/.codex/config.toml`, `~/.config/opencode/opencode.json`,
  `~/.pi/agent/models.json`, `~/.pi/agent/settings.json` — the harness-native
  configs, kept as the fallback for CLIs started directly in a terminal.
- `~/.local/share/opencode/auth.json` — registers `cliproxy` in OpenCode's
  credential store with a non-secret placeholder key, because Omnigent's
  OpenCode readiness only credits `auth.json` entries (or `OPENAI_API_KEY`-style
  env vars), never a custom `opencode.json` provider. OpenCode merges
  `opencode.json` `options.apiKey` (`{env:OMNIGENT_GATEWAY_API_KEY}`) over it,
  so the real key never touches disk (verified against a capturing gateway).

The entrypoint also exports the non-secret `OPENAI_BASE_URL` /
`ANTHROPIC_BASE_URL` and appends `OMNIGENT_GATEWAY_API_KEY` to
`OMNIGENT_RUNNER_ENV_PASSTHROUGH` so the key reaches harness processes. The
key is deliberately never mirrored into `OPENAI_API_KEY` (Pi/OpenCode would
auto-register a built-in `openai` provider and send it to api.openai.com).
Re-check the catalog against `GET $OMNIGENT_GATEWAY_BASE_URL/v1/models`
whenever the gateway's inventory changes.

Dry run without a server:

```sh
docker run --rm -e OMNIGENT_GATEWAY_API_KEY=dummy omnigent-host:b203ba4c-kt10 \
    sh -c 'cat /etc/claude-code/managed-settings.json ~/.codex/config.toml'
```

## Dokploy entries (raw Compose)

1. Server: project *Kainogent* → Create Compose → provider **Raw**, paste
   `docker-compose.server.yaml`, fill the `${VAR}`s from `.env.example` in the
   Environment tab. The image is pinned by digest (published tag
   `kainotomic-v0.14.0-kt4`). Volumes have fixed names (`omnigent-kt-postgres-data`, `omnigent-kt-data`)
   and cannot collide with the stopped `omnigent-server-do_*` volumes.
   Routing is **label-only**: do not add a Dokploy domain row to this compose
   (Dokploy would generate a second Traefik router for
   `kainogent.kainotomic.com` without the `kainogent-cloudflare-only@file` /
   `kainogent-no-store@file` middlewares). Leave the legacy domain row on
   `OmnigentDO20260719`; it is inert while that compose is stopped. Rollback:
   stop this compose first, then `docker start` the legacy containers — the
   two routers must never be live at the same time.
2. Workspaces: project *Kainogent Workspaces*, one raw compose per user from
   `docker-compose.workspace.yaml` with that user's `OMNIGENT_SERVER_URL` and
   `OMNIGENT_GATEWAY_API_KEY`; the host image is pinned by digest (published tag
   `kainotomic-v0.14.0-kt10`). The `egress` sidecar rejects traffic to `OMNIGENT_EGRESS_DENY_IP`
   (dokploy-root) exactly like the legacy `nft` script. After the first start
   the host stays up waiting for login (no crash-loop). Enroll with:

   ```sh
   docker exec -it omnigent-kt-ws-<slug>-host-1 omnigent login "$OMNIGENT_SERVER_URL"
   ```

   Example for ssah: `docker exec -it omnigent-kt-ws-ssah-host-1 omnigent login
   https://kainogent.kainotomic.com`. Finish Google in the printed URL. The
   waiter polls `auth_tokens.json` and execs the host daemon when the file
   is usable — do not stop the container or `docker run` a second login
   process. A restart after login is optional.

   The host image does not ship `kiro-cli`; the host hello omits kiro-native
   when that binary is absent, so New Chat cannot offer Kiro. Antigravity
   (`agy`) is selectable. Each user signs in with their own Google
   OAuth — there is no `agy login` subcommand and no shared CLIProxy Gemini key:

   ```sh
   docker exec -it omnigent-kt-ws-<slug>-host-1 agy
   ```

   Follow the printed Google sign-in. The token is written under `~/.gemini`
   on that workspace's `home` volume (`oauth_creds.json` or
   `antigravity-cli/antigravity-oauth-token`). Verify with `agy models` (exits 0
   when signed in; do not print the token). You can also start Antigravity from
   New Chat and complete the same browser sign-in in the session TUI.

   `host` shares `egress`'s network namespace. The `host` healthcheck passes
   when `omnigent host` is running and `curl "$OMNIGENT_SERVER_URL/health"`
   succeeds, **or** when the entrypoint is still waiting for login
   (`pgrep … workspace-entrypoint`). A dead namespace (or an unreachable
   server) after the daemon has started shows as `unhealthy` after
   `start_period` + 12 failed probes (~2.5 min). Docker does not restart on
   unhealthy (`restart: unless-stopped` only acts on process exit).
   `depends_on … restart: true` restarts `host` when a Compose
   command restarts/recreates `egress`; a Docker-runtime restart of `egress`
   (crash, `docker restart`) is not covered — `host` keeps the dead namespace
   and its daemon keeps running, so the unhealthy `host` is the only signal.
   A plain `docker compose up -d` does **not** fix it (the `egress` container
   id is unchanged, so Compose sees no drift); recover with `docker compose
   restart host` (or `up -d --force-recreate host`, or Dokploy Redeploy),
   which re-joins the live namespace and turns healthy within ~10 s. No
   autoheal container is used.

## Images and placeholders

Both compose files pin `ghcr.io/kainotomic/omnigent-{server,host}` by digest
(published from `kainotomic-publish-images.yml`; the packages are private, so
the pulling host needs a `docker login ghcr.io` with `read:packages`). To roll
a new build: run the workflow, take the digests from its "Report digests"
step, replace the `@sha256:` values (host appears twice), commit.

Still to fill per deployment: every value in `.env.example` (Dokploy
Environment tab, never committed).
