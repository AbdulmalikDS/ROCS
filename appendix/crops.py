import numpy as np


def saliency_boxes(image, saliency, n=5, crop_ratio=0.6):
    """Five crops with the centre one moved onto the saliency peak."""
    w, h = image.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    rows, cols = saliency.shape
    peak = int(np.argmax(saliency))
    cy, cx = divmod(peak, cols)
    cx, cy = int(cx * w / cols), int(cy * h / rows)
    x1, y1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
    x1, y1 = min(x1, w - cw), min(y1, h - ch)
    return [
        (x1, y1, x1 + cw, y1 + ch),
        (0, 0, cw, ch), (w - cw, 0, w, ch),
        (0, h - ch, cw, h), (w - cw, h - ch, w, h),
    ][:n]
