from contextlib import contextmanager
import numpy as np
import torch
from typing import List, Tuple

def pack_points(
    foreground_points: List[Tuple[int, int]] | None,
    background_points: List[Tuple[int, int]] | None,
):
    foreground_points = foreground_points or []
    background_points = background_points or []
    if not foreground_points and not background_points:
        return None, None
    points_array = np.array(foreground_points + background_points, dtype=np.int32)
    labels_array = np.array(
        [1] * len(foreground_points) + [0] * len(background_points), dtype=np.int32
    )
    return points_array, labels_array

@contextmanager
def autocast_context(enabled_flag: bool):
    if enabled_flag:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield
