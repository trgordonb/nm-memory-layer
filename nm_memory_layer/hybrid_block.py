def _heading_sections(wiki_dir: str):
    """Split every wiki page into sections exactly like the skill's
    wiki_search.py: line-level ATX headings outside code fences (fence-aware),
    flush the preamble before the first heading, heading-stack tracking, and a
    locator "<rel_path>\\x1f<section_index>" per section (the \\x1f separator
    matches semantic_sections rows already stored in the vector index)."""
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def _split_page(rel_path: str, body: str) -> list[dict]:
        sections = []
        heading_stack: list[tuple[int, str]] = []
        current_path: list[str] = []
        current_level = 0
        lines: list[str] = []
        in_fence = False

        def append_section() -> None:
            sections.append({
                "text": "\n".join(lines),
                "section_index": len(sections),
                "heading_path": list(current_path),
            })

        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                lines.append(line)
                continue
            heading = None if in_fence else HEADING_RE.match(line)
            if not heading:
                lines.append(line)
                continue
            append_section()
            current_level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()
            heading_stack.append((current_level, heading_text))
            current_path = [t for _, t in heading_stack]
            lines = []

        append_section()
        return sections

    pages = WikiStore(wiki_dir)._pages()
    out = []
    for rel_path, text in pages:
        body = text
        if body.startswith("---"):
            body = "---".join(body.split("---")[2:])
        for chunk in _split_page(rel_path, body):
            out.append({
                "locator": f"{rel_path}\x1f{chunk['section_index']}",
                "rel_path": rel_path,
                "text": chunk["text"],
                "heading_path": chunk["heading_path"],
            })
    return out

