from pathlib import Path


def test_docker_and_runbook_files_exist() -> None:
    assert Path("docker-compose.yml").exists()
    assert Path("docker/api/Dockerfile").exists()
    assert Path("docker/web/Dockerfile").exists()
    assert Path("docs/runbook.md").exists()
