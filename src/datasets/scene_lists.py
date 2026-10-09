"""Resolving a config's scene list to per-scene camera json paths.

Two forms are accepted under ``config.data``:

  root / split / scene_list   the dataset root, the split folder under it and a text file
                              of scene ids, one per line (``src/data/DL3DV/test_scenes.txt``);
                              scene i lives at <root>/<split>/<id>/<scene_file>
  data_path                   a text file listing the camera json of every scene directly,
                              absolute or relative to the list file (the original 2Xplat form)
"""
import os


def resolve_scene_paths(data_cfg, split: str | None = None) -> list[str]:
    """Absolute paths of the scenes' camera json files, in list order."""
    root = data_cfg.get("root", None)
    if root:
        split = split or data_cfg.get("split", None)
        scene_file = data_cfg.get("scene_file", "opencv_cameras.json")
        with open(data_cfg.scene_list) as f:
            ids = [line.strip() for line in f if line.strip()]
        base = os.path.join(root, split) if split else root
        return [os.path.join(base, i, scene_file) for i in ids]
    list_file = data_cfg.data_path
    folder = os.path.dirname(os.path.abspath(list_file))
    with open(list_file) as f:
        paths = [line.strip() for line in f if line.strip()]
    return [p if p.startswith("/") else os.path.join(folder, p) for p in paths]
