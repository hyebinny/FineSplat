"""Reading FineSplat training checkpoints."""


def checkpoint_state(ckpt: dict, use_ema: bool = False) -> dict:
    """Checkpoint weights under the current module names (EMA weights if ``use_ema``).
    Legacy ``refiner.*`` keys (and a top-level ``"refiner"`` dict) are mapped onto ``stage2.*``."""
    state = ckpt.get("ema") if use_ema else None
    if state is None:
        state = ckpt.get("state", ckpt.get("refiner", ckpt))
    return {("stage2." + k[len("refiner."):] if k.startswith("refiner.") else k): v
            for k, v in state.items()}
