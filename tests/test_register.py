"""register(ctx), plugin.yaml, config_schema.py, and the skill file."""

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_config_schema_declares_the_api_key_secret():
    source = (REPO_ROOT / "config_schema.py").read_text()

    assert 'name="recall"' in source
    assert "KIND_SECRET" in source
    assert 'env_key="RECALL_API_KEY"' in source


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
