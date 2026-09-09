import random


def fixed5_boxes(w, h, r=0.6, n=5):
    cw, ch = max(1, int(r * w)), max(1, int(r * h))
    cx, cy = w // 2, h // 2
    return [
        (cx - cw // 2, cy - ch // 2, cx - cw // 2 + cw, cy - ch // 2 + ch),
        (0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h),
    ][:n]


def grid_boxes(w, h, n=3):
    cw, ch = w // n, h // n
    if min(cw, ch) < 1:
        raise ValueError("The grid is larger than the image.")
    return [(x * cw, y * ch, (x + 1) * cw, (y + 1) * ch)
            for y in range(n) for x in range(n)]


def random_boxes(w, h, n=5, rng=None):
    rng = rng or random.Random(0)
    boxes = []
    for _ in range(n):
        ratio = rng.uniform(0.3, 0.7)
        cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        boxes.append((x, y, x + cw, y + ch))
    return boxes
