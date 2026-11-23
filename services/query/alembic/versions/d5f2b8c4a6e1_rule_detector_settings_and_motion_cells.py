"""rule detector settings and motion cells

Revision ID: d5f2b8c4a6e1
Revises: c3e8a1f5b7d2
Create Date: 2026-11-23 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f2b8c4a6e1'
down_revision: Union[str, Sequence[str], None] = 'c3e8a1f5b7d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # OMNI-908, D-52: which of Algorithm 1's cells moved, and the grid; NULL on older rows.
    op.add_column('segments', sa.Column('motion_cells', sa.BigInteger(), nullable=True))
    op.add_column('segments', sa.Column('motion_grid', sa.SmallInteger(), nullable=True))
    # The rule detector's camera settings: the zone (whole frame), N (10 s), the hours (none).
    op.add_column('cameras', sa.Column('zone_left', sa.Float(), server_default='0', nullable=False))
    op.add_column('cameras', sa.Column('zone_top', sa.Float(), server_default='0', nullable=False))
    op.add_column('cameras', sa.Column('zone_right', sa.Float(), server_default='1', nullable=False))
    op.add_column('cameras', sa.Column('zone_bottom', sa.Float(), server_default='1', nullable=False))
    op.add_column('cameras', sa.Column('sustained_motion_s', sa.Float(), server_default='10', nullable=False))
    op.add_column('cameras', sa.Column('active_from', sa.Time(), nullable=True))
    op.add_column('cameras', sa.Column('active_until', sa.Time(), nullable=True))
    op.create_check_constraint(
        'ck_cameras_zone_inside_frame', 'cameras',
        '0 <= zone_left AND zone_left < zone_right AND zone_right <= 1 '
        'AND 0 <= zone_top AND zone_top < zone_bottom AND zone_bottom <= 1',
    )
    op.create_check_constraint('ck_cameras_sustained_motion_positive', 'cameras', 'sustained_motion_s > 0')
    op.create_check_constraint(
        'ck_cameras_hours_both_or_neither', 'cameras', '(active_from IS NULL) = (active_until IS NULL)'
    )


def downgrade() -> None:
    """Downgrade schema."""
    for name in ('ck_cameras_hours_both_or_neither', 'ck_cameras_sustained_motion_positive', 'ck_cameras_zone_inside_frame'):
        op.drop_constraint(name, 'cameras', type_='check')
    for column in ('active_until', 'active_from', 'sustained_motion_s', 'zone_bottom', 'zone_right', 'zone_top', 'zone_left'):
        op.drop_column('cameras', column)
    op.drop_column('segments', 'motion_grid')
    op.drop_column('segments', 'motion_cells')
