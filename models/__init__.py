"""
Models module for Seawaste Detection
"""

from .seawaste_model import (
    SeawasteDetectionModel,
    create_seawaste_model,
    CBAM,
    MarineColorCorrection,
    SmallObjectEnhancer
)

__all__ = [
    'SeawasteDetectionModel',
    'create_seawaste_model',
    'CBAM',
    'MarineColorCorrection',
    'SmallObjectEnhancer'
]
