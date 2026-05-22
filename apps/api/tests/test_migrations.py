from pathlib import Path


def test_alembic_migration_files_exist() -> None:
    assert Path("alembic.ini").exists()
    assert Path("migrations/env.py").exists()
    assert Path("migrations/versions/0001_create_generation_jobs.py").exists()
