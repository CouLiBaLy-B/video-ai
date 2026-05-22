"""create generation jobs table

Revision ID: 0001_create_generation_jobs
Revises:
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_create_generation_jobs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create generation_jobs table."""
    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.String(length=64), nullable=False),
    )
    op.create_index("idx_generation_jobs_status", "generation_jobs", ["status"])


def downgrade() -> None:
    """Drop generation_jobs table."""
    op.drop_index("idx_generation_jobs_status", table_name="generation_jobs")
    op.drop_table("generation_jobs")
