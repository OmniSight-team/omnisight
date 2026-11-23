"""An alert's status, where it came from, and how serious it is (OMNI-907, D-51).

Shared, because `query` serves alerts and `pfm` (the rule detector, then the model) raises them.
"""

from enum import StrEnum


class AlertStatus(StrEnum):
    NEW = "new"  # raised, not reviewed yet: what the unread count counts
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class AlertSource(StrEnum):
    RULE = "rule"  # slice 9's rule detector (OMNI-908)
    MODEL = "model"  # the PFM (OMNI-911)


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
