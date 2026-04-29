"""add rag tables

Revision ID: a4f8c2d1e9b3
Revises: c9fb57636433
Create Date: 2026-04-29 16:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "a4f8c2d1e9b3"
down_revision: str | Sequence[str] | None = "c9fb57636433"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "destination_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("destination_name", sa.Text(), nullable=False),
        sa.Column("source_title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_destination_documents_destination_name",
        "destination_documents",
        ["destination_name"],
    )

    op.create_table(
        "destination_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("destination_name", sa.Text(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["destination_documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_destination_chunks_destination_name",
        "destination_chunks",
        ["destination_name"],
    )
    op.create_index(
        "ix_destination_chunks_document_id",
        "destination_chunks",
        ["document_id"],
    )
    op.create_index(
        "ix_destination_chunks_chunk_index",
        "destination_chunks",
        ["chunk_index"],
    )
    # Vector index intentionally omitted — requires data present first.
    # Add after ingestion:
    # CREATE INDEX ON destination_chunks USING hnsw (embedding vector_cosine_ops)
    # WITH (m = 16, ef_construction = 64);  -- requires pgvector >= 0.5.0


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("destination_chunks")
    op.drop_table("destination_documents")
    # Extension left in place intentionally; other objects may depend on the vector type.
