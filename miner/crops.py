import random


def fixed5_boxes(w, h, r=0.6, n=5):
    # Each side scales independently, so a crop keeps the image's aspect ratio
    # and covers r^2 of the area. Corners first, centre last.
    cw, ch = max(1, int(r * w)), max(1, int(r * h))
    return [(x, y, x + cw, y + ch) for x, y in (
        (0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch),
        ((w - cw) // 2, (h - ch) // 2),
    )][:n]


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
