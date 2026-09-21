"""Tests for the always-on prompt memory (PromptMemory + memory_manage tool)."""

import pytest
from pydantic import ValidationError

from nm_memory_layer import MEMORY_CHAR_LIMIT, PromptMemory, create_memory_manage_tool


@pytest.fixture()
def memory(tmp_path):
    return PromptMemory(memory_dir=str(tmp_path / "memories"))


@pytest.fixture()
def tool(memory):
    return create_memory_manage_tool(memory)


class TestPromptMemoryOps:
    def test_empty_load_returns_empty_string(self, memory):
        assert memory.load() == ""

    def test_add_bullet_prefixing_and_load_format(self, memory):
        memory.add("memory", "Quarterly reports live in reports/qN.md")
        memory.add("memory", "- already a bullet")
        memory.add("user", "Gordon prefers concise answers")
        block = memory.load()
        assert block.startswith("<agent_memory>")
        assert "<memory>\n- Quarterly reports live in reports/qN.md\n- already a bullet\n</memory>" in block
        assert "<user>\n- Gordon prefers concise answers\n</user>" in block

    def test_replace_patches_first_occurrence(self, memory):
        memory.add("memory", "reports live in reports/qN.md")
        result = memory.replace("memory", "reports/qN.md", "reports/YYYY/qN.md")
        assert result.startswith("OK:")
        assert "reports/YYYY/qN.md" in memory._read("memory")

    def test_remove_deletes_and_collapses_blank_lines(self, memory):
        memory.add("memory", "line one")
        memory.add("memory", "line two")
        memory.add("memory", "line three")
        assert memory.remove("memory", "line two").startswith("OK:")
        content = memory._read("memory")
        assert "line two" not in content
        assert "\n\n" not in content
        assert content.count("\n") == 3  # two entries + trailing newline

    def test_empty_add_rejected(self, memory):
        assert memory.add("memory", "   ").startswith("Rejected:")

    def test_missing_text_rejected_for_replace_and_remove(self, memory):
        assert memory.replace("memory", "not present", "x").startswith("Rejected:")
        assert memory.remove("memory", "not present").startswith("Rejected:")


class TestBudget:
    def test_combined_budget_enforced_across_both_files(self, memory):
        memory.add("memory", "y" * (MEMORY_CHAR_LIMIT - 100))  # file: LIMIT - 98 chars
        # "- " prefix makes the user entry 152 chars -> combined LIMIT + 54.
        result = memory.add("user", "x" * 150)
        assert result.startswith("Rejected:")
        assert memory._read("user") == ""

    def test_write_up_to_exactly_the_limit_is_allowed(self, memory):
        memory.add("memory", "y" * (MEMORY_CHAR_LIMIT - 60))  # file: "- "+3515y +\n = LIMIT-57 chars
        result = memory.add("user", "x" * 54)  # "- "+54x +\n = 57 chars -> combined = LIMIT
        assert result.startswith("OK:")
        assert memory.total_chars() == MEMORY_CHAR_LIMIT

    def test_rejected_write_leaves_files_untouched(self, memory):
        before = memory._read("memory")
        memory.add("memory", "y" * (MEMORY_CHAR_LIMIT + 1))
        assert memory._read("memory") == before


class TestTool:
    def test_dispatch_all_operations(self, tool):
        assert tool.invoke({"operation": "add", "target": "memory", "content": "tool line"}).startswith("OK:")
        assert tool.invoke(
            {"operation": "replace", "target": "memory", "old_content": "tool line", "content": "tool line v2"}
        ).startswith("OK:")
        assert tool.invoke({"operation": "remove", "target": "memory", "old_content": "tool line v2"}).startswith("OK:")

    def test_invalid_operation_blocked_at_schema_level(self, tool):
        with pytest.raises(ValidationError):
            tool.invoke({"operation": "bogus", "target": "memory"})

    def test_invalid_target_blocked_at_schema_level(self, tool):
        with pytest.raises(ValidationError):
            tool.invoke({"operation": "add", "target": "everywhere", "content": "x"})

    def test_docstring_teaches_layer_boundary(self, tool):
        description = tool.description
        assert "NEXT session" in description
        assert "session_search" in description
