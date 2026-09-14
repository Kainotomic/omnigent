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
                                   (Claude Code, Pi, OpenCode, Omnigent anthropic)
  OMNIGENT_GATEWAY_OPENAI_MODEL    overrides the catalog's openai default
                                   (Codex, Omnigent openai)
  OMNIGENT_GATEWAY_FORCE=1         overwrite files in the persistent home
  OMNIGENT_SERVER_URL              Omnigent server the host daemon dials

Model catalog: /opt/omnigent-kainotomic/harness-templates/gateway-models.jsonl,
one model per line with the ids CLIProxy serves, their limits and the
per-harness allocation (family, default, claudeCode tiers, codexProfile). Every
per-harness model block below is generated from it, so a model is added or
re-tiered in exactly one place.

Rendered from /opt/omnigent-kainotomic/harness-templates (string.Template for
the gateway scalars, then the model blocks are injected structurally):
  /etc/claude-code/managed-settings.json   always (image-owned, not in the home)
  ~/.omnigent/config.yaml                  only if absent; FORCE replaces just providers.cliproxy
  ~/.codex/config.toml                     only if absent (or FORCE)
  ~/.codex/<codexProfile>.config.toml      only if absent (or FORCE); `codex --profile <name>`
  ~/.config/opencode/opencode.json         only if absent (or FORCE)
  ~/.pi/agent/models.json                  only if absent (or FORCE)
  ~/.pi/agent/settings.json                only if absent (or FORCE)
  ~/.local/share/opencode/auth.json        only if absent (or FORCE); placeholder key

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

import json
import os
import re
import signal
import string
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import tomllib
import yaml

TEMPLATE_DIR = Path("/opt/omnigent-kainotomic/harness-templates")
CATALOG_PATH = TEMPLATE_DIR / "gateway-models.jsonl"
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
        claude_tiers=tuple(str(t) for t in tiers_raw),
        codex_profile=profile,
        reasoning=bool(raw.get("reasoning", True)),
        input=tuple(str(i) for i in inputs),
        context_window=_positive_int(raw.get("contextWindow"), f"{where} contextWindow"),
        max_tokens=_positive_int(raw.get("maxTokens"), f"{where} maxTokens"),
        thinking_level_map=dict(tlm) if isinstance(tlm, dict) else None,
    )


def load_catalog(path: Path = CATALOG_PATH) -> list[GatewayModel]:
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
    return models


@dataclass(frozen=True)
class Allocation:
    """Catalog plus the per-family defaults after env overrides."""

    models: tuple[GatewayModel, ...]

    def family(self, name: str) -> list[GatewayModel]:
        return [m for m in self.models if m.family == name]

    def default(self, family: str) -> GatewayModel:
        return next(m for m in self.models if m.family == family and m.default)

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
    """Claude Code (Anthropic family only): ANTHROPIC_MODEL + alias pins."""
    env = skeleton.setdefault("env", {})
    env["ANTHROPIC_MODEL"] = alloc.default("anthropic").id
    for model in alloc.family("anthropic"):
        for tier in model.claude_tiers:
            env[CLAUDE_TIER_ENV[tier]] = model.id
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


def write_if_allowed(
    target: Path, content: str, *, force: bool, always: bool, mode: int = 0o644
) -> None:
    if target.exists() and not (force or always):
        log(f"keep    {target} (exists; set OMNIGENT_GATEWAY_FORCE=1 to overwrite)")
        return
    write_atomic(target, content, mode)
    log(f"render  {target}")


def write_omnigent_config(target: Path, rendered: str, *, force: bool) -> None:
    """`omnigent host` stores its host identity in this file, so a FORCE
    re-render merges only providers.cliproxy instead of replacing the file."""
    if not target.exists():
        write_atomic(target, rendered, 0o600)
        log(f"render  {target}")
        return
    if not force:
        log(f"keep    {target} (exists; set OMNIGENT_GATEWAY_FORCE=1 to overwrite)")
        return
    existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(existing, dict):
        raise SystemExit(f"refusing to merge: {target} is not a mapping")
    providers = existing.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    providers[PROVIDER_ID] = yaml.safe_load(rendered)["providers"][PROVIDER_ID]
    existing["providers"] = providers
    write_atomic(target, yaml.safe_dump(existing, default_flow_style=False, sort_keys=True), 0o600)
    log(f"merge   {target} (providers.{PROVIDER_ID} replaced, other keys kept)")


def render_files(values: dict[str, str], alloc: Allocation) -> dict[str, str]:
    """Return {template name: rendered content} for every harness file."""
    anthropic_default = alloc.default("anthropic").id
    openai_default = alloc.default("openai").id

    claude = json.loads(_substitute("claude-managed-settings.json", values))
    claude_out = _dump_json(claude_settings(claude, alloc))

    codex_out = _substitute("codex-config.toml", {**values, **codex_limits(alloc)})
    parsed = tomllib.loads(codex_out)
    assert parsed["model"] == openai_default, "codex root model mismatch"
    codex_profiles = codex_profile_files(alloc)
    for content in codex_profiles.values():
        tomllib.loads(content)

    opencode = json.loads(_substitute("opencode.json", values))
    opencode["model"] = f"{PROVIDER_ID}/{anthropic_default}"
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
    force = env.get("OMNIGENT_GATEWAY_FORCE", "").strip().lower() in ("1", "true", "yes")
    home = Path(env.get("HOME") or "/root")

    log(
        f"gateway={base_url} anthropic_default={values['ANTHROPIC_MODEL']} "
        f"openai_default={values['OPENAI_MODEL']} models={len(alloc.models)} "
        f"key={'set' if env.get(KEY_VAR) else 'MISSING'}"
    )
    if not env.get(KEY_VAR):
        log(f"warning: {KEY_VAR} is not set; harnesses will fail to authenticate")

    rendered = render_files(values, alloc)
    targets = (
        ("claude-managed-settings.json", Path("/etc/claude-code/managed-settings.json"), True),
        ("codex-config.toml", home / ".codex" / "config.toml", False),
        ("opencode.json", home / ".config" / "opencode" / "opencode.json", False),
        ("pi-models.json", home / ".pi" / "agent" / "models.json", False),
        ("pi-settings.json", home / ".pi" / "agent" / "settings.json", False),
    )
    for template, target, always in targets:
        write_if_allowed(target, rendered[template], force=force, always=always)
    for key, content in rendered.items():
        if key.startswith("codex-profile/"):
            name = key.removeprefix("codex-profile/")
            write_if_allowed(home / ".codex" / name, content, force=force, always=False)
    # OpenCode credential store (0600 like OpenCode writes it). Placeholder key;
    # the real one is opencode.json's {env:...}, which OpenCode merges on top.
    xdg_data = Path(env.get("XDG_DATA_HOME", "").strip() or home / ".local" / "share")
    write_if_allowed(
        xdg_data / "opencode" / "auth.json",
        rendered["opencode-auth.json"],
        force=force,
        always=False,
        mode=0o600,
    )
    config_home = Path(env.get("OMNIGENT_CONFIG_HOME", "").strip() or home / ".omnigent")
    write_omnigent_config(
        config_home / "config.yaml", rendered["omnigent-config.yaml"], force=force
    )

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
