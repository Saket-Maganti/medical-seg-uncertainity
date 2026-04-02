"""
Legacy compatibility shim.

The canonical implementation now lives in models/unet_mc.py.
"""

from models.unet_mc import DeepEnsemble, MCDropoutUNet

__all__ = ["MCDropoutUNet", "DeepEnsemble"]
