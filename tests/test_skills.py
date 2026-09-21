"""Tests for the skills layer (SkillLibrary, skill_manage, load_skill, index)."""

import pytest
from pydantic import ValidationError

from nm_memory_layer import SkillLibrary, create_load_skill_tool, create_skill_manage_tool

BODY = """# Backfill FX series

1. Set the date range.
2. Run jetta backfill for each pair.
3. Verify row counts against trading days.
"""


@pytest.fixture()
def lib(tmp_path):
    return SkillLibrary(skills_dir=str(tmp_path / "skills"))


@pytest.fixture()
def manage(lib):
    return create_skill_manage_tool(lib)


@pytest.fixture()
def load(lib):
    return create_load_skill_tool(lib)


@pytest.fixture()
def populated(lib):
    lib.create_skill("fx-backfill", "Backfill FX series from Dukascopy.", BODY)
    return lib


class TestCreateAndList:
    def test_create_writes_agentskills_style_file(self, populated, tmp_path):
        path = tmp_path / "skills" / "fx-backfill" / "SKILL.md"
        assert path.exists()
        text = path.read_text()
        assert text.startswith("---\nname: fx-backfill")
        assert "Backfill FX series from Dukascopy." in text
        assert "jetta backfill" in text

    def test_list_skills_metadata_only(self, populated):
        skills = populated.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "fx-backfill"
        assert "Dukascopy" in skills[0]["description"]
        assert "jetta" not in skills[0]["description"]  # body never leaks into metadata

    def test_index_renders_names_and_descriptions_only(self, populated):
        index = populated.render_index()
        assert index.startswith("<skills_index>")
        assert "- fx-backfill: Backfill FX series from Dukascopy." in index
        assert "jetta backfill" not in index  # no body in the index

    def test_empty_library_renders_empty_index(self, lib):
        assert lib.render_index() == ""

    def test_duplicate_name_rejected(self, populated):
        result = populated.create_skill("fx-backfill", "dup", "body")
        assert result.startswith("Rejected:")

    def test_invalid_name_rejected(self, lib):
        assert lib.create_skill("Bad Name!", "d", "b").startswith("Rejected:")

    def test_invalid_category_rejected(self, lib):
        assert lib.create_skill("ok-name", "d", "b", category="../evil").startswith("Rejected:")

    def test_empty_body_rejected(self, lib):
        assert lib.create_skill("ok-name", "d", "   ").startswith("Rejected:")

    def test_category_nesting(self, lib):
        lib.create_skill("deploy-k8s", "Deploy services.", "steps here", category="devops")
        assert (lib.skills_dir / "devops" / "deploy-k8s" / "SKILL.md").exists()
        assert [s["name"] for s in lib.list_skills()] == ["deploy-k8s"]


class TestLoad:
    def test_load_returns_full_file(self, populated):
        loaded = populated.load_skill("fx-backfill")
        assert loaded.startswith("---")
        assert "jetta backfill" in loaded

    def test_load_resolves_by_dir_name_when_frontmatter_missing(self, lib, tmp_path):
        skill_dir = tmp_path / "skills" / "legacy-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("no frontmatter here")
        assert "no frontmatter here" in lib.load_skill("legacy-skill")

    def test_load_unknown_rejected(self, lib):
        assert lib.load_skill("nope").startswith("Rejected:")


class TestPatchAndEdit:
    def test_patch_is_targeted(self, populated):
        result = populated.patch_skill("fx-backfill", "2. Run jetta backfill for each pair.", "2. Run jetta backfill --full.")
        assert result.startswith("OK:")
        text = (populated.skills_dir / "fx-backfill" / "SKILL.md").read_text()
        assert "--full" in text and "each pair" not in text
        assert text.count("jetta backfill") == 1  # only the patched line mentions it

    def test_patch_missing_text_rejected(self, populated):
        assert populated.patch_skill("fx-backfill", "not present", "x").startswith("Rejected:")

    def test_edit_rewrites_body_keeps_frontmatter(self, populated):
        result = populated.edit_skill("fx-backfill", "# Rewritten\n\nnew steps")
        assert result.startswith("OK:")
        text = (populated.skills_dir / "fx-backfill" / "SKILL.md").read_text()
        assert "name: fx-backfill" in text  # frontmatter preserved
        assert "Dukascopy" in text  # description preserved
        assert "new steps" in text and "jetta" not in text  # body replaced

    def test_patch_unknown_skill_rejected(self, lib):
        assert lib.patch_skill("ghost", "a", "b").startswith("Rejected:")


class TestFilesAndDelete:
    def test_write_and_remove_reference_file(self, populated):
        assert populated.write_skill_file("fx-backfill", "references/notes.md", "extra notes").startswith("OK:")
        assert (populated.skills_dir / "fx-backfill" / "references" / "notes.md").exists()
        assert populated.remove_skill_file("fx-backfill", "references/notes.md").startswith("OK:")
        assert not (populated.skills_dir / "fx-backfill" / "references" / "notes.md").exists()

    def test_path_traversal_blocked(self, populated):
        assert populated.write_skill_file("fx-backfill", "../outside.md", "evil").startswith("Rejected:")
        assert populated.write_skill_file("fx-backfill", "/etc/passwd", "evil").startswith("Rejected:")
        assert populated.write_skill_file("fx-backfill", "a/../../b.md", "evil").startswith("Rejected:")
        assert not (populated.skills_dir.parent / "outside.md").exists()

    def test_delete_removes_directory(self, populated):
        populated.write_skill_file("fx-backfill", "references/x.md", "x")
        assert populated.delete_skill("fx-backfill").startswith("OK:")
        assert not (populated.skills_dir / "fx-backfill").exists()

    def test_delete_unknown_rejected(self, lib):
        assert lib.delete_skill("ghost").startswith("Rejected:")


class TestTools:
    def test_manage_dispatch_all_actions(self, lib, manage):
        assert manage.invoke({"action": "create", "name": "tool-made", "description": "d", "content": "b"}).startswith("OK:")
        assert manage.invoke({"action": "patch", "name": "tool-made", "old_content": "b", "content": "b2"}).startswith("OK:")
        assert manage.invoke({"action": "write_file", "name": "tool-made", "relative_path": "references/r.md", "content": "ref"}).startswith("OK:")
        assert manage.invoke({"action": "edit", "name": "tool-made", "content": "rewritten"}).startswith("OK:")
        assert manage.invoke({"action": "remove_file", "name": "tool-made", "relative_path": "references/r.md"}).startswith("OK:")
        assert manage.invoke({"action": "delete", "name": "tool-made"}).startswith("OK:")

    def test_invalid_action_blocked_at_schema_level(self, manage):
        with pytest.raises(ValidationError):
            manage.invoke({"action": "nuke", "name": "x"})

    def test_load_tool_dispatch(self, populated, load):
        assert "jetta backfill" in load.invoke({"name": "fx-backfill"})
        assert load.invoke({"name": "ghost"}).startswith("Rejected:")

    def test_docstrings_teach_patch_preference(self, manage):
        assert "PREFER patch" in manage.description
