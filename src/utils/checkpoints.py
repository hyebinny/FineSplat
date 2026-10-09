"""Reading FineSplat training checkpoints."""


def checkpoint_state(ckpt: dict, use_ema: bool = False) -> dict:
    """The weights inside a training checkpoint, under the current module names.

    ``use_ema`` reads the EMA weights an end-to-end checkpoint carries. Checkpoints written
    before the Stage-2 module was renamed keep their weights under ``refiner.*`` (and the
    dict under ``"refiner"``); they are mapped onto ``stage2.*`` here so they still load.
    """
    state = ckpt.get("ema") if use_ema else None
    if state is None:
        state = ckpt.get("state", ckpt.get("refiner", ckpt))
    return {("stage2." + k[len("refiner."):] if k.startswith("refiner.") else k): v
            for k, v in state.items()}
