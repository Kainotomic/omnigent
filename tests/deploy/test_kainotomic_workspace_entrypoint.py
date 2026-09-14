"""Catalog-hash skip vs re-render for the Kainotomic workspace entrypoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINT = _ROOT / "deploy/kainotomic/workspace-entrypoint.py"
_TEMPLATES = _ROOT / "deploy/kainotomic/harness-templates"

_CATALOG_V1 = """\
{"id":"acme/alpha","family":"anthropic","default":true,"claudeCode":["opus"],"reasoning":true,"input":["text"],"contextWindow":1000,"maxTokens":100}
{"id":"acme/beta","family":"openai","default":true,"opencodeDefault":true,"reasoning":true,"input":["text"],"contextWindow":2000,"maxTokens":200}
"""

_CATALOG_V2 = """\
{"id":"acme/alpha","family":"anthropic","default":true,"claudeCode":["opus"],"reasoning":true,"input":["text"],"contextWindow":1000,"maxTokens":100}
{"id":"acme/gamma","family":"openai","default":true,"opencodeDefault":true,"reasoning":true,"input":["text"],"contextWindow":3000,"maxTokens":300}
"""


def _load_entrypoint():
    name = "kainotomic_workspace_entrypoint"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bind(module, tmp_path: Path, catalog: str):
    catalog_path = tmp_path / "gateway-models.jsonl"
    catalog_path.write_text(catalog, encoding="utf-8")
    module.TEMPLATE_DIR = _TEMPLATES
    module.CATALOG_PATH = catalog_path
    module.MANAGED_SETTINGS_PATH = tmp_path / "managed-settings.json"
    home = tmp_path / "home"
    home.mkdir()
    return home, catalog_path


def _env(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "OMNIGENT_GATEWAY_API_KEY": "dummy",
        "OMNIGENT_GATEWAY_BASE_URL": "https://openai.example.test",
    }


def test_catalog_hash_skip_then_rerender_preserves_user_keys(tmp_path: Path) -> None:
    """A matching hash leaves home files alone; a new catalog updates model bits.

    Existing workspace homes used to skip every restart, so a catalog change
    never reached OpenCode/Pi/Codex/Omnigent. The overlay-owned hash file is
    what decides skip vs surgical update.
    """
    module = _load_entrypoint()
    home, catalog_path = _bind(module, tmp_path, _CATALOG_V1)
    env = _env(home)

    module.render_all(env)

    opencode_path = home / ".config" / "opencode" / "opencode.json"
    pi_settings_path = home / ".pi" / "agent" / "settings.json"
    auth_path = home / ".local" / "share" / "opencode" / "auth.json"
    omni_path = home / ".omnigent" / "config.yaml"
    hash_path = home / ".omnigent" / ".kainotomic-catalog-hash"

    opencode = json.loads(opencode_path.read_text(encoding="utf-8"))
    assert opencode["model"] == "cliproxy/acme/beta"
    assert "acme/beta" in opencode["provider"]["cliproxy"]["models"]
    opencode["plugin"] = ["user-plugin"]
    opencode["provider"]["extra"] = {"name": "keep-me"}
    opencode_path.write_text(json.dumps(opencode, indent=2) + "\n", encoding="utf-8")

    pi_settings = json.loads(pi_settings_path.read_text(encoding="utf-8"))
    pi_settings["theme"] = "dark"
    pi_settings_path.write_text(json.dumps(pi_settings, indent=2) + "\n", encoding="utf-8")

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["user-added"] = {"type": "api", "key": "keep"}
    auth_path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")

    omni = yaml.safe_load(omni_path.read_text(encoding="utf-8"))
    omni["host_id"] = "stable-identity"
    omni_path.write_text(yaml.safe_dump(omni, default_flow_style=False), encoding="utf-8")

    first_hash = hash_path.read_text(encoding="utf-8").strip()
    assert first_hash == module.catalog_digest(catalog_path)

    opencode["model"] = "cliproxy/user-picked"
    opencode_path.write_text(json.dumps(opencode, indent=2) + "\n", encoding="utf-8")

    module.render_all(env)

    skipped = json.loads(opencode_path.read_text(encoding="utf-8"))
    assert skipped["model"] == "cliproxy/user-picked"
    assert skipped["plugin"] == ["user-plugin"]
    assert hash_path.read_text(encoding="utf-8").strip() == first_hash

    catalog_path.write_text(_CATALOG_V2, encoding="utf-8")
    module.render_all(env)

    updated = json.loads(opencode_path.read_text(encoding="utf-8"))
    assert updated["model"] == "cliproxy/acme/gamma"
    assert "acme/gamma" in updated["provider"]["cliproxy"]["models"]
    assert "acme/beta" not in updated["provider"]["cliproxy"]["models"]
    assert updated["plugin"] == ["user-plugin"]
    assert updated["provider"]["extra"] == {"name": "keep-me"}

    pi_updated = json.loads(pi_settings_path.read_text(encoding="utf-8"))
    assert pi_updated["defaultModel"] == "acme/alpha"
    assert pi_updated["theme"] == "dark"

    pi_models = json.loads((home / ".pi" / "agent" / "models.json").read_text(encoding="utf-8"))
    ids = {m["id"] for m in pi_models["providers"]["cliproxy"]["models"]}
    assert ids == {"acme/alpha", "acme/gamma"}

    auth_kept = json.loads(auth_path.read_text(encoding="utf-8"))
    assert auth_kept["user-added"] == {"type": "api", "key": "keep"}

    omni_updated = yaml.safe_load(omni_path.read_text(encoding="utf-8"))
    assert omni_updated["host_id"] == "stable-identity"
    assert omni_updated["providers"]["cliproxy"]["openai"]["models"]["default"] == "acme/gamma"

    assert hash_path.read_text(encoding="utf-8").strip() == module.catalog_digest(catalog_path)
    assert hash_path.read_text(encoding="utf-8").strip() != first_hash


_CATALOG_TWO_ANTHROPIC = """\
{"id":"acme/alpha","family":"anthropic","default":true,"claudeCode":["opus"],"reasoning":true,"input":["text"],"contextWindow":1000,"maxTokens":100}
{"id":"acme/delta","family":"anthropic","claudeCode":["haiku","subagent"],"reasoning":true,"input":["text"],"contextWindow":1000,"maxTokens":100}
{"id":"acme/beta","family":"openai","default":true,"opencodeDefault":true,"reasoning":true,"input":["text"],"contextWindow":2000,"maxTokens":200}
"""


def test_claude_managed_settings_allowlists_anthropic_catalog_ids(tmp_path: Path) -> None:
    """Claude Code's /model picker is independent of the alias env pins.

    Without availableModels it keeps its built-in Anthropic catalog (opus 4.x,
    bare claude-opus-5, [1m] twins). The overlay must allowlist the anthropic
    factory IDs — not family aliases, which wildcard every official version.
    """
    module = _load_entrypoint()
    home, _catalog_path = _bind(module, tmp_path, _CATALOG_TWO_ANTHROPIC)
    module.render_all(_env(home))

    managed = json.loads(module.MANAGED_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert managed["model"] == "acme/alpha"
    assert managed["availableModels"] == ["acme/alpha", "acme/delta"]
    assert managed["enforceAvailableModels"] is True
    assert managed["modelPicker"] == {
        "replaceBuiltInOptions": True,
        "options": [
            {"model": "acme/alpha", "label": "acme/alpha"},
            {"model": "acme/delta", "label": "acme/delta"},
        ],
    }
    env = managed["env"]
    assert env["ANTHROPIC_MODEL"] == "acme/alpha"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "acme/alpha"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "acme/delta"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "acme/delta"
    assert env["CLAUDE_CODE_DISABLE_1M_CONTEXT"] == "1"
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL" not in env
    assert "acme/beta" not in json.dumps(managed)
