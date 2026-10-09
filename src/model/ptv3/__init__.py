"""Point Transformer V3, vendored from SplatFormer's Pointcept fork (MIT).

Copied verbatim from github.com/ChenYutongTHU/Pointcept @ c4aa232 (branch w_setup):
ptv3.py, structure.py, modules.py, misc.py, pdnorm.py, serialization/. Import paths were
made local, `MODELS`/`MODULES`/`gin` decorators dropped, and `torch_scatter.segment_csr`
swapped for a `torch.segment_reduce` shim (torch_scatter_compat.py) - installing upstream
Pointcept instead would pull in pointops/ocnn/torch_geometric for backbones we never run.

The refiners use only `Block`/`SerializedAttention` (from .ptv3) and `Point`
(from .structure); the full `PointTransformerV3` U-Net is kept as reference.
"""

from .ptv3 import PointTransformerV3
from .structure import Point

__all__ = ["PointTransformerV3", "Point"]
