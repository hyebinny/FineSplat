"""Point Transformer V3, vendored from SplatFormer's Pointcept fork (MIT;
github.com/ChenYutongTHU/Pointcept @ c4aa232). Import paths made local, registry
decorators dropped, ``torch_scatter.segment_csr`` replaced by a ``torch.segment_reduce``
shim. Stage 2 uses only ``Block``/``SerializedAttention`` and ``Point``.
"""

from .ptv3 import PointTransformerV3
from .structure import Point

__all__ = ["PointTransformerV3", "Point"]
