#!/opt/venv/bin/python
"""Kainotomic workspace entrypoint: render harness gateway configs, exec the host.

Runs under tini (PID 1). This script is tini's child and must handle SIGTERM
itself. Before exec'ing `omnigent host` it waits for a usable login at
``$OMNIGENT_DATA_DIR/auth_tokens.json`` or ``~/.omnigent/auth_tokens.json``
(same path as ``omnigent/cli_auth.py``) so an unenrolled workspace stays
up for ``docker exec … omnigent login`` instead of crash-looping. Reads
only these variables
(all secret-free except the API key, which is never written to disk or logged):

  OMNIGENT_GATEWAY_BASE_URL        CLIProxy root, default https://openai.kainotomic.com
  OMNIGENT_GATEWAY_API_KEY         per-workspace CLIProxy key (env only)
  OMNIGENT_GATEWAY_ANTHROPIC_MODEL overrides the catalog's anthropic default
                                   (Claude Code, Pi, Omnigent anthropic)
  OMNIGENT_GATEWAY_OPENAI_MODEL    overrides the catalog's openai default
                                   (Codex, Omnigent openai)
  OMNIGENT_GATEWAY_FORCE=1         overwrite files in the persistent home
  OMNIGENT_SERVER_URL              Omnigent server the host daemon dials

Model catalog: /opt/omnigent-kainotomic/harness-templates/gateway-models.jsonl,
one model per line with the ids CLIProxy serves, their limits and the
per-harness allocation (family, default, opencodeDefault, claudeCode tiers,
codexProfile). Every per-harness model block below is generated from it, so a
model is added or re-tiered in exactly one place.

Rendered from /opt/omnigent-kainotomic/harness-templates (string.Template for
the gateway scalars, then the model blocks are injected structurally):
  /etc/claude-code/managed-settings.json   always (image-owned, not in the home)
  ~/.omnigent/config.yaml                  create if absent; catalog-hash or FORCE
                                           updates providers.cliproxy catalog fields
  ~/.codex/config.toml                     create if absent; catalog-hash updates
                                           model + model_context_window + model_catalog_json
  ~/.codex/model_catalog.json              openai default + codexProfile rows; catalog-hash
                                           or first-boot writes it (Codex model/list pin)
  ~/.codex/<codexProfile>.config.toml      create/update on catalog-hash; `codex --profile <name>`
  ~/.config/opencode/opencode.json         create if absent; catalog-hash updates
                                           model + provider.cliproxy.models
  ~/.pi/agent/models.json                  create if absent; catalog-hash updates
                                           providers.cliproxy.models
  ~/.pi/agent/settings.json                create if absent; catalog-hash updates
                                           defaultProvider + defaultModel
  ~/.local/share/opencode/auth.json        only if absent (or FORCE); placeholder key

The SHA-256 of gateway-models.jsonl is stored at
~/.omnigent/.kainotomic-catalog-hash. A missing or different hash re-renders
the catalog-driven fields above and leaves user-added keys, extra providers,
plugins, Codex session history, agy oauth, and auth_tokens.json alone.
A matching hash skips those files. FORCE still replaces the whole file
(except omnigent config.yaml, which still merges only providers.cliproxy).

~/.omnigent/config.yaml carries the Omnigent-side `providers:` gateway entry
(api_key_ref env:OMNIGENT_GATEWAY_API_KEY). It is what Omnigent's readiness
and the native adapters consume: they inject it per session (Claude
--settings, Codex -c model_provider, Pi managed models.json) and forward the
referenced env var to the runner. The harness-native files are the fallback
for CLIs started directly in a terminal. Every file references the key as
OMNIGENT_GATEWAY_API_KEY (Codex env_key, OpenCode {env:...}, Pi $VAR, Claude
apiKeyHelper). That name is not on the host->runner allowlist
(omnigent/host/connect.py HARNESS_CREDENTIAL_ENV_VARS), so it is appended to
OMNIGENT_RUNNER_ENV_PASSTHROUGH. Only the non-secret OPENAI_BASE_URL /
ANTHROPIC_BASE_URL are exported; OPENAI_API_KEY is never set.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import tomllib
import yaml

TEMPLATE_DIR = Path("/opt/omnigent-kainotomic/harness-templates")
CATALOG_PATH = TEMPLATE_DIR / "gateway-models.jsonl"
MANAGED_SETTINGS_PATH = Path("/etc/claude-code/managed-settings.json")
CATALOG_HASH_NAME = ".kainotomic-catalog-hash"
KEY_HELPER = "/usr/local/bin/omnigent-gateway-key"
DEFAULT_BASE_URL = "https://openai.kainotomic.com"
KEY_VAR = "OMNIGENT_GATEWAY_API_KEY"
PROVIDER_ID = "cliproxy"
FAMILIES = ("anthropic", "openai")
# Claude Code alias pins (code.claude.com/docs/en/model-config, "Environment
# variables"). ANTHROPIC_SMALL_FAST_MODEL is deprecated in favour of the haiku
# pin. Omnigent's picker reads the same keys (claude_native/main.py).
CLAUDE_TIER_ENV = {
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "subagent": "CLAUDE_CODE_SUBAGENT_MODEL",
}
# codex-rs/protocol/src/config_types.rs ProfileV2Name: plain name, no dots.
_CODEX_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def log(msg: str) -> None:
    print(f"[omnigent-workspace] {msg}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class GatewayModel:
    id: str
    family: str
    default: bool = False
    opencode_default: bool = False
    claude_tiers: tuple[str, ...] = ()
    codex_profile: str | None = None
    reasoning: bool = True
    input: tuple[str, ...] = ("text",)
    context_window: int | None = None
    max_tokens: int | None = None
    thinking_level_map: dict[str, str | None] | None = None

    @property
    def short_id(self) -> str:
        return self.id.rsplit("/", 1)[-1]


def _positive_int(raw: object, what: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise SystemExit(f"catalog: {what} must be a positive integer, got {raw!r}")
    return raw


def _parse_model(raw: dict[str, object], line_no: int) -> GatewayModel:
    where = f"{CATALOG_PATH.name}:{line_no}"
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
        raise SystemExit(f"catalog: {where}: invalid id {model_id!r}")
    family = raw.get("family")
    if family not in FAMILIES:
        raise SystemExit(f"catalog: {where}: family must be one of {FAMILIES}, got {family!r}")
    tiers_raw = raw.get("claudeCode", [])
    if not isinstance(tiers_raw, list) or any(t not in CLAUDE_TIER_ENV for t in tiers_raw):
        raise SystemExit(f"catalog: {where}: claudeCode must list {sorted(CLAUDE_TIER_ENV)}")
    if tiers_raw and family != "anthropic":
        raise SystemExit(f"catalog: {where}: claudeCode tiers require family anthropic")
    profile = raw.get("codexProfile")
    if profile is not None and (
        not isinstance(profile, str) or not _CODEX_PROFILE_RE.match(profile)
    ):
        raise SystemExit(f"catalog: {where}: codexProfile must match {_CODEX_PROFILE_RE.pattern}")
    if profile is not None and family != "openai":
        raise SystemExit(f"catalog: {where}: codexProfile requires family openai")
    inputs = raw.get("input", ["text"])
    if (
        not isinstance(inputs, list)
        or not inputs
        or any(i not in ("text", "image") for i in inputs)
    ):
        raise SystemExit(f"catalog: {where}: input must be a non-empty list of text/image")
    tlm = raw.get("thinkingLevelMap")
    if tlm is not None and not isinstance(tlm, dict):
        raise SystemExit(f"catalog: {where}: thinkingLevelMap must be a mapping")
    return GatewayModel(
        id=model_id,
        family=str(family),
        default=bool(raw.get("default", False)),
        opencode_default=bool(raw.get("opencodeDefault", False)),
        claude_tiers=tuple(str(t) for t in tiers_raw),
        codex_profile=profile,
        reasoning=bool(raw.get("reasoning", True)),
        input=tuple(str(i) for i in inputs),
        context_window=_positive_int(raw.get("contextWindow"), f"{where} contextWindow"),
        max_tokens=_positive_int(raw.get("maxTokens"), f"{where} maxTokens"),
        thinking_level_map=dict(tlm) if isinstance(tlm, dict) else None,
    )


def load_catalog(path: Path | None = None) -> list[GatewayModel]:
    path = path or CATALOG_PATH
    models: list[GatewayModel] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise SystemExit(f"catalog: {path.name}:{line_no}: expected an object")
        models.append(_parse_model(raw, line_no))
    ids = [m.id for m in models]
    if len(set(ids)) != len(ids):
        raise SystemExit("catalog: duplicate model ids")
    profiles = [m.codex_profile for m in models if m.codex_profile]
    if len(set(profiles)) != len(profiles):
        raise SystemExit("catalog: duplicate codexProfile names")
    seen_tiers: set[str] = set()
    for m in models:
        dup = seen_tiers.intersection(m.claude_tiers)
        if dup:
            raise SystemExit(f"catalog: Claude tier(s) {sorted(dup)} assigned twice")
        seen_tiers.update(m.claude_tiers)
    for family in FAMILIES:
        defaults = [m for m in models if m.family == family and m.default]
        if len(defaults) != 1:
            raise SystemExit(
                f"catalog: family {family} needs exactly one default, got {len(defaults)}"
            )
    oc_defaults = [m for m in models if m.opencode_default]
    if len(oc_defaults) != 1:
        raise SystemExit(f"catalog: needs exactly one opencodeDefault, got {len(oc_defaults)}")
    return models


@dataclass(frozen=True)
class Allocation:
    """Catalog plus the per-family defaults after env overrides."""

    models: tuple[GatewayModel, ...]

    def family(self, name: str) -> list[GatewayModel]:
        return [m for m in self.models if m.family == name]

    def default(self, family: str) -> GatewayModel:
        return next(m for m in self.models if m.family == family and m.default)

    def opencode_default(self) -> GatewayModel:
        return next(m for m in self.models if m.opencode_default)

    def by_id(self, model_id: str) -> GatewayModel | None:
        return next((m for m in self.models if m.id == model_id), None)


def apply_overrides(models: list[GatewayModel], env: dict[str, str]) -> Allocation:
    """Point a family's default at OMNIGENT_GATEWAY_<FAMILY>_MODEL when set.

    An override outside the catalog is honoured with a bare entry (no limits,
    no tiers) so every harness file still names the same default."""
    result = list(models)
    for family in FAMILIES:
        override = env.get(f"OMNIGENT_GATEWAY_{family.upper()}_MODEL", "").strip()
        if not override:
            continue
        if not _MODEL_ID_RE.match(override):
            raise SystemExit(f"refusing to render: unsafe model id {override!r}")
        result = [replace(m, default=False) if m.family == family else m for m in result]
        existing = next((m for m in result if m.id == override), None)
        if existing is None:
            log(f"warning: {override} is not in {CATALOG_PATH.name}; rendering it without limits")
            result.append(GatewayModel(id=override, family=family, default=True))
        elif existing.family != family:
            raise SystemExit(
                f"{override} is a {existing.family} model; cannot default {family} to it"
            )
        else:
            result = [replace(m, default=True) if m.id == override else m for m in result]
    return Allocation(models=tuple(result))


def strip_comments(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def _dump_json(obj: object) -> str:
    return json.dumps(strip_comments(obj), indent=2) + "\n"


def _substitute(name: str, values: dict[str, str]) -> str:
    text = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
    return string.Template(text).substitute(values)


def _yaml_header(text: str) -> str:
    """Leading comment block of a YAML template, kept above the generated body."""
    lines: list[str] = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""


# ── per-harness model blocks ─────────────────────────────────────────


def claude_settings(skeleton: dict, alloc: Allocation) -> dict:
    """Claude Code anthropic family: default, alias pins, and /model allowlist.

    Alias env vars only remap opus/sonnet/haiku/fable. availableModels plus a
    replaceBuiltInOptions modelPicker hide the built-in Anthropic catalog;
    family aliases are omitted so they cannot wildcard official opus 4.x IDs.
    """
    env = skeleton.setdefault("env", {})
    default_id = alloc.default("anthropic").id
    env["ANTHROPIC_MODEL"] = default_id
    env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
    for model in alloc.family("anthropic"):
        for tier in model.claude_tiers:
            env[CLAUDE_TIER_ENV[tier]] = model.id
    anthropic_ids = [model.id for model in alloc.family("anthropic")]
    skeleton["model"] = default_id
    skeleton["availableModels"] = anthropic_ids
    skeleton["enforceAvailableModels"] = True
    skeleton["modelPicker"] = {
        "replaceBuiltInOptions": True,
        "options": [{"model": model_id, "label": model_id} for model_id in anthropic_ids],
    }
    return skeleton


def codex_limits(alloc: Allocation) -> dict[str, str]:
    """Codex root-level limits for the OpenAI default model."""
    default = alloc.default("openai")
    limits = ""
    if default.context_window is not None:
        limits = f"model_context_window = {default.context_window}"
    return {"CODEX_MODEL_LIMITS": limits}


def codex_profile_files(alloc: Allocation) -> dict[str, str]:
    """One ~/.codex/<name>.config.toml per codexProfile model.

    Codex layers `$CODEX_HOME/<name>.config.toml` over config.toml when run
    with `--profile <name>` (name: [A-Za-z0-9_-]+); the provider is inherited."""
    files: dict[str, str] = {}
    for model in alloc.family("openai"):
        if model.codex_profile:
            files[f"{model.codex_profile}.config.toml"] = (
                f"# codex --profile {model.codex_profile}: layered over config.toml.\n"
                f'model = "{model.id}"\n'
            )
    return files


def codex_picker_models(alloc: Allocation) -> list[GatewayModel]:
    """OpenAI default plus every codexProfile model, catalog order.

    This is the New Chat / model-options allowlist. Other openai-family
    rows stay in Pi/OpenCode; they must not appear in Codex's picker.
    """
    seen: set[str] = set()
    out: list[GatewayModel] = []
    default = alloc.default("openai")
    out.append(default)
    seen.add(default.id)
    for model in alloc.family("openai"):
        if model.codex_profile and model.id not in seen:
            out.append(model)
            seen.add(model.id)
    return out


def _bundled_codex_catalog() -> dict | None:
    """Compiled-in Codex catalog, probed from an empty CODEX_HOME.

    An empty home has no cliproxy provider, so GET /v1/models is not
    consulted. Used only as clone-source metadata; picker slugs come
    from gateway-models.jsonl.
    """
    codex = shutil.which("codex")
    if not codex:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="kt-codex-bundled-") as tmp:
            completed = subprocess.run(
                [codex, "debug", "models"],
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "CODEX_HOME": tmp},
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"warning: could not probe Codex bundled catalog ({exc})")
        return None
    if completed.returncode != 0:
        log("warning: codex debug models failed; writing a fallback catalog")
        return None
    try:
        catalog = json.loads(completed.stdout)
    except ValueError:
        log("warning: codex debug models returned non-JSON; writing a fallback catalog")
        return None
    models = catalog.get("models") if isinstance(catalog, dict) else None
    if not isinstance(models, list) or not models:
        return None
    return catalog


def _fallback_catalog_entry(model: GatewayModel, *, priority: int) -> dict:
    """Minimal ModelInfo Codex 0.154.0 accepts when a bundled clone is unavailable."""
    entry: dict[str, object] = {
        "slug": model.id,
        "display_name": model.id,
        "description": None,
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Low"},
            {"effort": "medium", "description": "Medium"},
            {"effort": "high", "description": "High"},
            {"effort": "xhigh", "description": "Extra high"},
        ],
        "default_reasoning_level": "medium",
        "shell_type": "unified_exec",
        "upgrade": None,
        "availability_nux": None,
        "model_messages": None,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": False,
        "experimental_supported_tools": [],
        "base_instructions": "You are Codex, a coding agent.",
    }
    if model.context_window is not None:
        entry["context_window"] = model.context_window
        entry["max_context_window"] = model.context_window
    return entry


def _clone_source_for(
    models: list[dict], model: GatewayModel, default: GatewayModel
) -> dict | None:
    """Closest bundled row to clone: matching short id, else the default's twin."""
    wanted = (model.short_id, default.short_id)
    for needle in wanted:
        for entry in models:
            slug = entry.get("slug")
            if isinstance(slug, str) and slug.rsplit("/", 1)[-1] == needle:
                return entry
    return next((entry for entry in models if isinstance(entry.get("slug"), str)), None)


def _clone_catalog_entry(source: dict, model: GatewayModel, *, priority: int) -> dict:
    entry = copy.deepcopy(source)
    entry["slug"] = model.id
    entry["display_name"] = model.id
    entry["visibility"] = "list"
    entry["priority"] = priority
    entry["upgrade"] = None
    entry["availability_nux"] = None
    if model.context_window is not None:
        entry["context_window"] = model.context_window
        entry["max_context_window"] = model.context_window
    return entry


def codex_model_catalog(alloc: Allocation) -> dict:
    """``{"models": [...]}`` for model_catalog_json: picker ids only."""
    picker = codex_picker_models(alloc)
    bundled = _bundled_codex_catalog()
    bundled_models = bundled.get("models") if isinstance(bundled, dict) else None
    sources = bundled_models if isinstance(bundled_models, list) else []
    default = alloc.default("openai")
    models: list[dict] = []
    for priority, model in enumerate(picker):
        source = _clone_source_for(sources, model, default) if sources else None
        if source is None:
            models.append(_fallback_catalog_entry(model, priority=priority))
        else:
            models.append(_clone_catalog_entry(source, model, priority=priority))
    return {"models": models}


def invalidate_codex_model_caches(config_home: Path) -> None:
    """Drop stored Codex/Claude probes so the next host start re-lists."""
    catalogs = config_home / "cache" / "model-catalogs"
    if catalogs.is_dir():
        for path in (
            *catalogs.glob("codex-native-*.json"),
            *catalogs.glob("claude-native-*.json"),
        ):
            with contextlib.suppress(OSError):
                path.unlink()
    probe = config_home / "cache" / "codex-model-probe"
    if probe.is_dir():
        shutil.rmtree(probe, ignore_errors=True)


def claude_catalog_has_1m_aliases(config_home: Path) -> bool:
    """True when a cached Claude New Chat catalog still lists ``[1m]`` twins."""
    catalogs = config_home / "cache" / "model-catalogs"
    if not catalogs.is_dir():
        return False
    for path in catalogs.glob("claude-native-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = payload.get("models") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and str(row.get("id", "")).endswith("[1m]"):
                return True
    return False


def opencode_models(alloc: Allocation) -> dict[str, dict]:
    """OpenCode provider.<id>.models (opencode.ai/config.json Model schema)."""
    out: dict[str, dict] = {}
    for model in alloc.models:
        entry: dict[str, object] = {
            "name": model.id,
            "reasoning": model.reasoning,
            "tool_call": True,
            "attachment": "image" in model.input,
            "modalities": {"input": list(model.input), "output": ["text"]},
        }
        if model.context_window is not None and model.max_tokens is not None:
            entry["limit"] = {"context": model.context_window, "output": model.max_tokens}
        out[model.id] = entry
    return out


def pi_models(alloc: Allocation) -> list[dict]:
    """Pi models.json entries (pi-mono docs/models.md Model Configuration)."""
    out: list[dict] = []
    for model in alloc.models:
        entry: dict[str, object] = {
            "id": model.id,
            "reasoning": model.reasoning,
            "input": list(model.input),
        }
        if model.thinking_level_map is not None:
            entry["thinkingLevelMap"] = model.thinking_level_map
        if model.context_window is not None:
            entry["contextWindow"] = model.context_window
        if model.max_tokens is not None:
            entry["maxTokens"] = model.max_tokens
        out.append(entry)
    return out


def omnigent_family_models(alloc: Allocation, family: str) -> dict[str, object]:
    """Omnigent FamilyConfig fields: models (role -> id) + default-model limits."""
    default = alloc.default(family)
    models: dict[str, str] = {"default": default.id}
    for model in alloc.family(family):
        for tier in model.claude_tiers:
            if tier != "subagent":
                models[tier] = model.id
        if not model.default and not model.claude_tiers:
            models[model.short_id] = model.id
    fields: dict[str, object] = {"models": models}
    if default.context_window is not None:
        fields["context_window"] = default.context_window
    if default.max_tokens is not None:
        fields["max_output_tokens"] = default.max_tokens
    return fields


# ── file writers ─────────────────────────────────────────────────────


def write_atomic(target: Path, content: str, mode: int = 0o644) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, target)


def catalog_digest(path: Path | None = None) -> str:
    """SHA-256 of the catalog file bytes (stable across processes)."""
    return hashlib.sha256((path or CATALOG_PATH).read_bytes()).hexdigest()


def catalog_hash_path(config_home: Path) -> Path:
    return config_home / CATALOG_HASH_NAME


def read_catalog_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_catalog_hash(path: Path, digest: str) -> None:
    write_atomic(path, digest + "\n", 0o644)
    log(f"catalog hash {digest[:12]} stored at {path}")


def _load_json_object(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_if_allowed(
    target: Path,
    content: str,
    *,
    force: bool,
    always: bool,
    catalog_stale: bool = False,
    merge: object | None = None,
    mode: int = 0o644,
) -> None:
    if always or force or not target.exists():
        write_atomic(target, content, mode)
        log(f"render  {target}")
        return
    if not catalog_stale:
        log(f"keep    {target} (catalog unchanged)")
        return
    if callable(merge):
        merge(target, content)
        return
    write_atomic(target, content, mode)
    log(f"update  {target} (catalog changed)")


def merge_opencode(target: Path, rendered: str) -> None:
    """Replace catalog default + cliproxy model map; keep plugins and extra providers."""
    existing = _load_json_object(target)
    new = json.loads(rendered)
    if existing is None or not isinstance(new, dict):
        write_atomic(target, rendered)
        log(f"update  {target} (catalog changed; replaced unreadable file)")
        return
    existing["model"] = new["model"]
    providers = existing.get("provider")
    if not isinstance(providers, dict):
        existing["provider"] = new["provider"]
    else:
        new_clip = new.get("provider", {}).get(PROVIDER_ID)
        clip = providers.get(PROVIDER_ID)
        if not isinstance(new_clip, dict):
            pass
        elif not isinstance(clip, dict):
            providers[PROVIDER_ID] = new_clip
        else:
            clip["models"] = new_clip["models"]
    write_atomic(target, _dump_json(existing))
    log(f"update  {target} (catalog changed; model + {PROVIDER_ID} models)")


def merge_pi_models(target: Path, rendered: str) -> None:
    """Replace cliproxy.models; keep other providers and cliproxy transport keys."""
    existing = _load_json_object(target)
    new = json.loads(rendered)
    if existing is None or not isinstance(new, dict):
        write_atomic(target, rendered)
        log(f"update  {target} (catalog changed; replaced unreadable file)")
        return
    providers = existing.get("providers")
    new_clip = new.get("providers", {}).get(PROVIDER_ID)
    if not isinstance(providers, dict) or not isinstance(new_clip, dict):
        existing["providers"] = new.get("providers", {})
    else:
        clip = providers.get(PROVIDER_ID)
        if not isinstance(clip, dict):
            providers[PROVIDER_ID] = new_clip
        else:
            clip["models"] = new_clip["models"]
    write_atomic(target, _dump_json(existing))
    log(f"update  {target} (catalog changed; {PROVIDER_ID} models)")


def merge_pi_settings(target: Path, rendered: str) -> None:
    """Replace catalog defaults; keep user settings (theme, extra keys)."""
    existing = _load_json_object(target)
    new = json.loads(rendered)
    if existing is None or not isinstance(new, dict):
        write_atomic(target, rendered)
        log(f"update  {target} (catalog changed; replaced unreadable file)")
        return
    existing["defaultProvider"] = new["defaultProvider"]
    existing["defaultModel"] = new["defaultModel"]
    write_atomic(target, _dump_json(existing))
    log(f"update  {target} (catalog changed; defaults)")


def merge_codex_config(target: Path, rendered: str) -> None:
    """Replace root model, limits, and catalog pin; keep other TOML tables."""
    try:
        new = tomllib.loads(rendered)
        existing = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        write_atomic(target, rendered)
        log(f"update  {target} (catalog changed; replaced unreadable file)")
        return
    new_model = new.get("model")
    if not isinstance(new_model, str) or not _MODEL_ID_RE.match(new_model):
        raise SystemExit(f"refusing to merge: unsafe Codex model {new_model!r}")
    text = existing
    if re.search(r"(?m)^model\s*=", text):
        text = re.sub(r'(?m)^model\s*=\s*"[^"]*"', f'model = "{new_model}"', text, count=1)
    else:
        text = f'model = "{new_model}"\n' + text
    new_window = new.get("model_context_window")
    if isinstance(new_window, int) and new_window > 0:
        if re.search(r"(?m)^model_context_window\s*=", text):
            text = re.sub(
                r"(?m)^model_context_window\s*=\s*\d+",
                f"model_context_window = {new_window}",
                text,
                count=1,
            )
        else:
            text = re.sub(
                r'(?m)^(model\s*=\s*"[^"]*"\s*)$',
                rf"\1\nmodel_context_window = {new_window}",
                text,
                count=1,
            )
    else:
        text = re.sub(r"(?m)^model_context_window\s*=\s*\d+\n?", "", text)
    catalog_path = new.get("model_catalog_json")
    if isinstance(catalog_path, str) and catalog_path:
        assignment = f"model_catalog_json = {json.dumps(catalog_path)}"
        if re.search(r"(?m)^model_catalog_json\s*=", text):
            text = re.sub(
                r'(?m)^model_catalog_json\s*=\s*"[^"]*"',
                assignment,
                text,
                count=1,
            )
        else:
            lines = text.splitlines(keepends=True)
            insert_at = next(
                (i for i, line in enumerate(lines) if line.lstrip().startswith("[")),
                len(lines),
            )
            insert = assignment + "\n"
            if insert_at > 0 and lines and not lines[insert_at - 1].endswith("\n"):
                insert = "\n" + insert
            lines.insert(insert_at, insert)
            text = "".join(lines)
    write_atomic(target, text)
    log(f"update  {target} (catalog changed; model + limits + catalog pin)")


def _apply_omnigent_catalog_fields(
    existing: dict, rendered: str, *, replace_provider: bool
) -> dict:
    new_provider = yaml.safe_load(rendered)["providers"][PROVIDER_ID]
    providers = existing.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        existing["providers"] = providers
    if replace_provider or not isinstance(providers.get(PROVIDER_ID), dict):
        providers[PROVIDER_ID] = new_provider
        return existing
    clip = providers[PROVIDER_ID]
    for family in FAMILIES:
        new_fam = new_provider.get(family)
        if not isinstance(new_fam, dict):
            continue
        fam = clip.get(family)
        if not isinstance(fam, dict):
            clip[family] = new_fam
            continue
        for key in ("models", "context_window", "max_output_tokens"):
            if key in new_fam:
                fam[key] = new_fam[key]
            else:
                fam.pop(key, None)
    return existing


def write_omnigent_config(
    target: Path, rendered: str, *, force: bool, catalog_stale: bool
) -> None:
    """`omnigent host` stores its host identity in this file, so a FORCE
    re-render replaces only providers.cliproxy. A catalog-hash miss updates
    family models/limits and leaves other cliproxy keys alone."""
    if not target.exists():
        write_atomic(target, rendered, 0o600)
        log(f"render  {target}")
        return
    if not force and not catalog_stale:
        log(f"keep    {target} (catalog unchanged)")
        return
    existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(existing, dict):
        raise SystemExit(f"refusing to merge: {target} is not a mapping")
    _apply_omnigent_catalog_fields(existing, rendered, replace_provider=force)
    write_atomic(target, yaml.safe_dump(existing, default_flow_style=False, sort_keys=True), 0o600)
    if force:
        log(f"merge   {target} (providers.{PROVIDER_ID} replaced, other keys kept)")
    else:
        log(f"update  {target} (catalog changed; {PROVIDER_ID} family models/limits)")


def render_files(values: dict[str, str], alloc: Allocation) -> dict[str, str]:
    """Return {template name: rendered content} for every harness file."""
    anthropic_default = alloc.default("anthropic").id
    openai_default = alloc.default("openai").id
    opencode_default = alloc.opencode_default().id

    claude = json.loads(_substitute("claude-managed-settings.json", values))
    claude_out = _dump_json(claude_settings(claude, alloc))

    codex_out = _substitute("codex-config.toml", {**values, **codex_limits(alloc)})
    parsed = tomllib.loads(codex_out)
    assert parsed["model"] == openai_default, "codex root model mismatch"
    picker_ids = [model.id for model in codex_picker_models(alloc)]
    catalog_obj = codex_model_catalog(alloc)
    catalog_slugs = [entry["slug"] for entry in catalog_obj["models"]]
    assert catalog_slugs == picker_ids, "codex catalog slugs mismatch picker"
    catalog_out = _dump_json(catalog_obj)
    codex_profiles = codex_profile_files(alloc)
    for content in codex_profiles.values():
        tomllib.loads(content)

    opencode = json.loads(_substitute("opencode.json", values))
    opencode["model"] = f"{PROVIDER_ID}/{opencode_default}"
    opencode["provider"][PROVIDER_ID]["models"] = opencode_models(alloc)
    opencode_out = _dump_json(opencode)

    pi = json.loads(_substitute("pi-models.json", values))
    pi["providers"][PROVIDER_ID]["models"] = pi_models(alloc)
    pi_out = _dump_json(pi)

    pi_settings = json.loads(_substitute("pi-settings.json", values))
    pi_settings["defaultProvider"] = PROVIDER_ID
    pi_settings["defaultModel"] = anthropic_default
    pi_settings_out = _dump_json(pi_settings)

    auth_out = _dump_json(json.loads(_substitute("opencode-auth.json", values)))

    omni_text = _substitute("omnigent-config.yaml", values)
    omni = yaml.safe_load(omni_text)
    provider = omni["providers"][PROVIDER_ID]
    for family in FAMILIES:
        provider[family].update(omnigent_family_models(alloc, family))
    omni_out = _yaml_header(omni_text) + yaml.safe_dump(
        omni, default_flow_style=False, sort_keys=False
    )
    assert yaml.safe_load(omni_out)["providers"][PROVIDER_ID]["openai"]["models"]["default"] == (
        openai_default
    )

    return {
        "claude-managed-settings.json": claude_out,
        "codex-config.toml": codex_out,
        "codex-model-catalog.json": catalog_out,
        **{f"codex-profile/{name}": content for name, content in codex_profiles.items()},
        "opencode.json": opencode_out,
        "pi-models.json": pi_out,
        "pi-settings.json": pi_settings_out,
        "opencode-auth.json": auth_out,
        "omnigent-config.yaml": omni_out,
    }


def render_all(env: dict[str, str]) -> None:
    base_url = env.get("OMNIGENT_GATEWAY_BASE_URL", "").strip().rstrip("/") or DEFAULT_BASE_URL
    values = {
        "GATEWAY_BASE_URL": base_url,
        "GATEWAY_OPENAI_BASE_URL": f"{base_url}/v1",
        "KEY_HELPER": KEY_HELPER,
    }
    for value in values.values():
        if any(ch in value for ch in '"\\\n'):
            raise SystemExit(f"refusing to render: unsafe characters in {value!r}")
    alloc = apply_overrides(load_catalog(), env)
    values["ANTHROPIC_MODEL"] = alloc.default("anthropic").id
    values["OPENAI_MODEL"] = alloc.default("openai").id
    values["OPENCODE_MODEL"] = alloc.opencode_default().id
    force = env.get("OMNIGENT_GATEWAY_FORCE", "").strip().lower() in ("1", "true", "yes")
    home = Path(env.get("HOME") or "/root")
    config_home = Path(env.get("OMNIGENT_CONFIG_HOME", "").strip() or home / ".omnigent")
    digest = catalog_digest()
    stored = read_catalog_hash(catalog_hash_path(config_home))
    catalog_stale = stored != digest
    catalog_json_path = str((home / ".codex" / "model_catalog.json").resolve())
    if any(ch in catalog_json_path for ch in '"\\\n'):
        raise SystemExit(f"refusing to render: unsafe Codex catalog path {catalog_json_path!r}")
    values["CODEX_MODEL_CATALOG_JSON"] = catalog_json_path

    log(
        f"gateway={base_url} anthropic_default={values['ANTHROPIC_MODEL']} "
        f"openai_default={values['OPENAI_MODEL']} "
        f"opencode_default={values['OPENCODE_MODEL']} models={len(alloc.models)} "
        f"catalog={'stale' if catalog_stale else 'unchanged'} "
        f"key={'set' if env.get(KEY_VAR) else 'MISSING'}"
    )
    if not env.get(KEY_VAR):
        log(f"warning: {KEY_VAR} is not set; harnesses will fail to authenticate")

    rendered = render_files(values, alloc)
    write_if_allowed(
        MANAGED_SETTINGS_PATH,
        rendered["claude-managed-settings.json"],
        force=force,
        always=True,
    )
    codex_config = home / ".codex" / "config.toml"
    codex_catalog = home / ".codex" / "model_catalog.json"
    catalog_missing = not codex_catalog.is_file()
    write_if_allowed(
        codex_config,
        rendered["codex-config.toml"],
        force=force,
        always=False,
        catalog_stale=catalog_stale,
        merge=merge_codex_config,
    )
    write_if_allowed(
        codex_catalog,
        rendered["codex-model-catalog.json"],
        force=force,
        always=False,
        catalog_stale=catalog_stale or catalog_missing,
    )
    if not force and codex_config.is_file():
        try:
            existing_toml = codex_config.read_text(encoding="utf-8")
        except OSError:
            existing_toml = ""
        if "model_catalog_json" not in existing_toml:
            merge_codex_config(codex_config, rendered["codex-config.toml"])
    if catalog_stale or catalog_missing or force or claude_catalog_has_1m_aliases(config_home):
        invalidate_codex_model_caches(config_home)
    write_if_allowed(
        home / ".config" / "opencode" / "opencode.json",
        rendered["opencode.json"],
        force=force,
        always=False,
        catalog_stale=catalog_stale,
        merge=merge_opencode,
    )
    write_if_allowed(
        home / ".pi" / "agent" / "models.json",
        rendered["pi-models.json"],
        force=force,
        always=False,
        catalog_stale=catalog_stale,
        merge=merge_pi_models,
    )
    write_if_allowed(
        home / ".pi" / "agent" / "settings.json",
        rendered["pi-settings.json"],
        force=force,
        always=False,
        catalog_stale=catalog_stale,
        merge=merge_pi_settings,
    )
    for key, content in rendered.items():
        if key.startswith("codex-profile/"):
            name = key.removeprefix("codex-profile/")
            write_if_allowed(
                home / ".codex" / name,
                content,
                force=force,
                always=False,
                catalog_stale=catalog_stale,
            )
    # OpenCode credential store (0600 like OpenCode writes it). Placeholder key;
    # the real one is opencode.json's {env:...}, which OpenCode merges on top.
    # Not catalog-driven — rewriting would clobber extra auth.json entries.
    xdg_data = Path(env.get("XDG_DATA_HOME", "").strip() or home / ".local" / "share")
    write_if_allowed(
        xdg_data / "opencode" / "auth.json",
        rendered["opencode-auth.json"],
        force=force,
        always=False,
        mode=0o600,
    )
    write_omnigent_config(
        config_home / "config.yaml",
        rendered["omnigent-config.yaml"],
        force=force,
        catalog_stale=catalog_stale,
    )
    if catalog_stale or not catalog_hash_path(config_home).is_file():
        write_catalog_hash(catalog_hash_path(config_home), digest)

    # Runner-visible env. The key travels ONLY as OMNIGENT_GATEWAY_API_KEY via
    # passthrough — never mirrored into OPENAI_API_KEY, which would make Pi and
    # OpenCode auto-register a built-in `openai` provider that sends the
    # CLIProxy key to api.openai.com.
    env.setdefault("OPENAI_BASE_URL", values["GATEWAY_OPENAI_BASE_URL"])
    env.setdefault("ANTHROPIC_BASE_URL", base_url)
    passthrough = [
        n.strip() for n in env.get("OMNIGENT_RUNNER_ENV_PASSTHROUGH", "").split(",") if n.strip()
    ]
    if KEY_VAR not in passthrough:
        passthrough.append(KEY_VAR)
    env["OMNIGENT_RUNNER_ENV_PASSTHROUGH"] = ",".join(passthrough)


# Same path as omnigent/cli_auth.py::_token_file_path (OMNIGENT_DATA_DIR, else
# $HOME/.omnigent/auth_tokens.json). Duplicated here so the overlay does not
# import the application package at PID-1 startup.
_TOKEN_FILE_NAME = "auth_tokens.json"
_WAIT_LOG_INTERVAL_S = 30.0
_WAIT_POLL_INTERVAL_S = 1.0


def token_file_path(env: dict[str, str]) -> Path:
    data_dir = env.get("OMNIGENT_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser() / _TOKEN_FILE_NAME
    home = Path(env.get("HOME") or "/root")
    return home / ".omnigent" / _TOKEN_FILE_NAME


def has_usable_login(path: Path, server_url: str) -> bool:
    """True when the tokens file has a session JWT for this server URL."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    entry = data.get(server_url.rstrip("/"))
    if not isinstance(entry, dict):
        return False
    token = entry.get("token")
    return isinstance(token, str) and bool(token.strip())


def wait_for_login(env: dict[str, str], server_url: str) -> None:
    """Block until a usable login exists, or exit 0 on SIGTERM/SIGINT.

    Polls (does not busy-spin). Logs every 30s so operators know
    ``docker exec … omnigent login`` is safe. tini forwards SIGTERM here.
    """
    path = token_file_path(env)
    if has_usable_login(path, server_url):
        return

    def _handle_term(_signum: int, _frame: object) -> None:
        log("received stop signal while waiting for login; exiting")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
    msg = f"waiting for omnigent login ({path}); docker exec is safe — container will not restart"
    log(msg)
    last_log = time.monotonic()
    while True:
        if has_usable_login(path, server_url):
            log(f"login found at {path}; starting host")
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            return
        now = time.monotonic()
        if now - last_log >= _WAIT_LOG_INTERVAL_S:
            log(msg)
            last_log = now
        time.sleep(_WAIT_POLL_INTERVAL_S)


def main(argv: list[str]) -> None:
    env = dict(os.environ)
    env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] = "1"
    render_all(env)
    if argv:
        cmd = argv
    else:
        server = env.get("OMNIGENT_SERVER_URL", "").strip()
        if not server:
            raise SystemExit("OMNIGENT_SERVER_URL is required (or pass a command to run)")
        wait_for_login(env, server)
        cmd = ["omnigent", "host", "--server", server, "--non-interactive"]
    log("exec " + " ".join(cmd))
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main(sys.argv[1:])
