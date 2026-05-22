from pathlib import Path

REQUIRED_SKILLS = {
    "cinematic-prompting",
    "image-to-video-routing",
    "video-quality-review",
    "gpu-cost-optimization",
    "safety-and-policy",
}


def test_required_deepagent_skills_exist() -> None:
    skills_root = Path("skills")
    discovered = {path.parent.name for path in skills_root.glob("*/SKILL.md")}

    assert REQUIRED_SKILLS.issubset(discovered)


def test_skill_files_have_frontmatter_description() -> None:
    for skill_file in Path("skills").glob("*/SKILL.md"):
        content = skill_file.read_text()
        assert content.startswith("---\n")
        assert "description:" in content
        assert "name:" in content
