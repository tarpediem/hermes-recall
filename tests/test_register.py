"""register(ctx), plugin.yaml, config_schema.py, and the skill file."""

import ast
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def config_schema():
    """The real ``CONFIG_SCHEMA``, loaded the way Hermes loads it.

    ``plugins.memory.config_schema.get_provider_config_schema`` executes the
    file by path under ``_hermes_memory_config_schema.<name>`` — never as a
    package import — so this mirrors that exactly.
    """
    spec = importlib.util.spec_from_file_location(
        "_hermes_memory_config_schema.recall", str(REPO_ROOT / "config_schema.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG_SCHEMA


class FakeCtx:
    def __init__(self, skill_raiser=None):
        self.providers = []
        self.skills = []
        self.order = []
        self.skill_raiser = skill_raiser

    def register_memory_provider(self, provider):
        self.providers.append(provider)
        self.order.append("provider")

    def register_skill(self, name, path, description):
        self.order.append("skill")
        if self.skill_raiser is not None:
            raise self.skill_raiser
        self.skills.append((name, path, description))


def test_register_registers_the_provider_then_the_skill(recall_package):
    ctx = FakeCtx()

    recall_package.register(ctx)

    assert len(ctx.providers) == 1
    assert ctx.providers[0].name == "recall"
    assert ctx.order == ["provider", "skill"]
    name, path, description = ctx.skills[0]
    assert name == "memory"
    assert Path(path).is_file()
    assert "Recall" in description


def test_register_registers_the_provider_before_a_failing_skill(recall_package):
    ctx = FakeCtx(skill_raiser=RuntimeError("registry unavailable"))

    with pytest.raises(RuntimeError):
        recall_package.register(ctx)

    assert len(ctx.providers) == 1, "a skill failure must never cost the provider"


def test_registered_provider_is_the_real_provider_class(recall_package):
    ctx = FakeCtx()

    recall_package.register(ctx)

    assert isinstance(ctx.providers[0], recall_package.RecallMemoryProvider)


def test_package_reexports_the_public_names(recall_package):
    for name in ("RecallClient", "RecallError", "RecallAuthError", "RecallMemoryProvider",
                 "register"):
        assert hasattr(recall_package, name), name


def test_plugin_yaml_declares_no_python_dependencies():
    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text())

    assert manifest["name"] == "recall"
    assert "python_dependencies" not in manifest
    assert "external_dependencies" not in manifest
    assert manifest["version"]
    assert manifest["description"]


def test_plugin_yaml_declares_no_pip_dependencies():
    """``pip_dependencies`` is the key Hermes actually reads (web_server /
    memory_setup). Declaring it would make the dashboard offer an install step
    for a plugin that needs nothing beyond what Hermes already pins."""
    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text())

    assert "pip_dependencies" not in manifest


def test_plugin_yaml_lists_the_hooks_we_implement():
    manifest = yaml.safe_load((REPO_ROOT / "plugin.yaml").read_text())

    for hook in ("on_session_end", "on_session_switch", "on_pre_compress",
                 "on_memory_write", "on_delegation"):
        assert hook in manifest["hooks"], hook


def test_config_schema_module_only_imports_the_declarative_base():
    """config_schema.py is loaded by the web server, which must NOT import the
    agent runtime — so it may only import from plugins.memory.config_schema."""
    tree = ast.parse((REPO_ROOT / "config_schema.py").read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module == "plugins.memory.config_schema", node.module
        elif isinstance(node, ast.Import):
            raise AssertionError("config_schema.py must not use plain imports")


def test_config_schema_declares_the_api_key_secret(config_schema):
    assert config_schema.name == "recall"
    assert config_schema.label == "Recall"

    secrets = [f for f in config_schema.fields if f.is_secret]
    assert [f.key for f in secrets] == ["api_key"]
    assert secrets[0].env_key == "RECALL_API_KEY"
    # Inline = shown in the compact panel, so the one field without which the
    # provider cannot work is never hidden behind the full-config modal.
    assert secrets[0].inline is True


def test_config_schema_name_matches_the_provider_name(config_schema, recall_package):
    """``_flat_json_path`` keys the config file on ``schema.name``; the loader
    keys everything else on ``provider.name``. A mismatch would silently split
    them into two config surfaces."""
    assert config_schema.name == recall_package.RecallMemoryProvider().name


def test_config_schema_fields_match_the_provider_config_keys(config_schema):
    """Every non-secret panel field must be a key the provider actually reads.

    ``load_recall_config`` drops anything not in ``DEFAULTS``, so an extra
    field would be saved and ignored, and a missing one would be unreachable
    from the dashboard.
    """
    from recall._provider import DEFAULTS

    keys = {f.key for f in config_schema.fields}
    assert keys - {"api_key"} == set(DEFAULTS)


def test_dashboard_and_provider_agree_on_the_config_file_path(tmp_path):
    """The Critical this test exists for: ``config_schema.py`` leaves
    ``storage`` at ``STORAGE_FLAT_JSON``, whose contract is
    ``get_hermes_home() / <name> / "config.json"``
    (``hermes_cli/web_server.py::_flat_json_path``). Both the provider's write
    path and its read path must be that exact file, or every non-secret field
    saved from the panel lands where nothing loads it.
    """
    from recall._provider import RecallMemoryProvider, load_recall_config, recall_config_path

    expected = tmp_path / "recall" / "config.json"
    assert recall_config_path(str(tmp_path)) == expected

    RecallMemoryProvider().save_config({"limit": 7}, str(tmp_path))
    assert expected.is_file(), "save_config must write the flat-json path"

    assert load_recall_config(str(tmp_path))["limit"] == 7


def test_skill_file_has_yaml_frontmatter_with_name_and_description():
    text = (REPO_ROOT / "skills" / "memory" / "SKILL.md").read_text()

    assert text.startswith("---\n")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "memory"
    assert frontmatter["description"]


def test_skill_file_documents_both_tools_and_the_injected_block():
    text = (REPO_ROOT / "skills" / "memory" / "SKILL.md").read_text()

    assert "recall_search" in text
    assert "recall_store" in text
    assert "Relevant memories (Recall)" in text
