"""The one `torch_scatter` entry point PTv3 needs, built on core torch.

`segment_csr` is the only torch_scatter call in all of Pointcept (PTv3 uses it
in `SerializedPooling`). `torch.segment_reduce` matches it exactly - same CSR
`offsets` convention, same reductions, autograd included - so this shim avoids
building torch-scatter from source against a torch/CUDA pair it never shipped
wheels for (torch 2.13+cu130 here).
"""

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, out=None, reduce: str = "sum"):
    """Reduce `src` over the row segments delimited by CSR index pointers `indptr`."""
    assert out is None, "segment_csr(out=...) is unused by PTv3"
    if indptr.dim() > 1:
        # PTv3 always passes a 1-D pointer; a batched pointer would need
        # per-row segmentation, which torch.segment_reduce does not do.
        raise NotImplementedError("only 1-D indptr is supported")
    return torch.segment_reduce(src, reduce, offsets=indptr, axis=0, unsafe=True)
