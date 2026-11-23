"""create alerts

Revision ID: c3e8a1f5b7d2
Revises: b7e2c94d1f06
Create Date: 2026-11-23 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3e8a1f5b7d2'
down_revision: Union[str, Sequence[str], None] = 'b7e2c94d1f06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SEVERITY = ('low', 'medium', 'high')
STATUS = ('new', 'confirmed', 'dismissed')
SOURCE = ('rule', 'model')


def upgrade() -> None:
    """Upgrade schema."""
    # OMNI-907: the alerts inbox, the plan's data model plus video_id and assigned_to (D-51).
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=True),
        sa.Column('video_id', sa.Integer(), nullable=True),
        sa.Column('segment_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('expected_in_s', sa.Float(), nullable=True),
        sa.Column('severity', sa.Enum(*SEVERITY, name='alert_severity'), nullable=False),
        sa.Column('status', sa.Enum(*STATUS, name='alert_status'), server_default='new', nullable=False),
        sa.Column('source', sa.Enum(*SOURCE, name='alert_source'), nullable=False),
        sa.Column('reasoning', sa.String(), nullable=True),
        sa.Column('tier_reached', sa.Integer(), nullable=True),
        sa.Column('raised_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('reviewed_by', sa.Integer(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('assigned_to', sa.Integer(), nullable=True),
        sa.CheckConstraint('(camera_id IS NULL) <> (video_id IS NULL)', name='ck_alerts_camera_or_video'),
        sa.CheckConstraint("(status = 'new') = (reviewed_at IS NULL)", name='ck_alerts_review_recorded'),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id']),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['segment_id'], ['segments.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['assigned_to'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_alerts_camera_id', 'alerts', ['camera_id'])
    op.create_index('ix_alerts_video_id', 'alerts', ['video_id'])
    op.create_index('ix_alerts_status_raised_at', 'alerts', ['status', 'raised_at'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_alerts_status_raised_at', table_name='alerts')
    op.drop_index('ix_alerts_video_id', table_name='alerts')
    op.drop_index('ix_alerts_camera_id', table_name='alerts')
    op.drop_table('alerts')
    for name in ('alert_source', 'alert_status', 'alert_severity'):
        sa.Enum(name=name).drop(op.get_bind(), checkfirst=True)
