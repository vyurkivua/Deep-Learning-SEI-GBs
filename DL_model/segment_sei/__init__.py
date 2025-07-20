"""
Code adapted from the `segmentation_models.pytorch` library by Pavel Yakubovskiy.
"""
import warnings, torch
from typing import Optional
from . import encoders, decoders, losses, metrics
from .decoders.fpn import FPN
from .base.hub_mixin import from_pretrained
from .__version__ import __version__

warnings.filterwarnings("ignore", message="is with a literal", category=SyntaxWarning)

_ARCH = {"fpn": FPN}

def create_model(
    arch: str,
    encoder_name: str = "mit_b5",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs
) -> torch.nn.Module:
    try:
        cls = _ARCH[arch.lower()]
    except KeyError:
        raise KeyError(f"`{arch}` not supported; choose from {list(_ARCH)}")
    return cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs
    )

__all__ = ["encoders", "decoders", "losses", "metrics", "FPN", "from_pretrained", "create_model", "__version__"]
