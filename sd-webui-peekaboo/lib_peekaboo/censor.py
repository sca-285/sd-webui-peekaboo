"""Censor styles: mosaic, blur, black bar, a CENSORED label, or stickers."""

import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

STICKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "stickers")

STYLES = ["Mosaic", "Blur", "Black bar", "Label", "Sticker"]
SIZED = ("Mosaic", "Blur", "Sticker")  # the styles that use the block / blur size

AUTO = "Auto by part (🍒 🌸 🍆💦 🍑)"
# Twemoji file names (code points) for each sticker choice
EMOJI = {
    "🔞": ["1f51e"],
    "🍆💦": ["1f346", "1f4a6"],
    "🍑": ["1f351"],
    "🍒": ["1f352"],
    "🌸": ["1f338"],
    "💋": ["1f48b"],
    "🙈": ["1f648"],
    "⭐": ["2b50"],
    "❤️": ["2764"],
    "🔥": ["1f525"],
    "💦": ["1f4a6"],
    "👀": ["1f440"],
    "😳": ["1f633"],
    "😏": ["1f60f"],
    "💯": ["1f4af"],
    "❌": ["274c"],
    "🚫": ["1f6ab"],
}
STICKERS = [AUTO] + list(EMOJI)
BY_PART = {
    "FEMALE_BREAST_EXPOSED": "🍒",
    "FEMALE_GENITALIA_EXPOSED": "🌸",
    "MALE_GENITALIA_EXPOSED": "🍆💦",
    "BUTTOCKS_EXPOSED": "🍑",
    "ANUS_EXPOSED": "🍑",
    "MALE_BREAST_EXPOSED": "🔥",
}

_cache = {}


def _sticker(code):
    if code not in _cache:
        _cache[code] = Image.open(os.path.join(STICKER_DIR, f"{code}.png")).convert("RGBA")
    return _cache[code]


def _font(size):
    try:
        from modules.paths_internal import roboto_ttf_file
        return ImageFont.truetype(roboto_ttf_file, size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


SHAPES = ["Oval", "Box"]


def _union(boxes, size, margin=0):
    x0 = max(0, min(b[0] for b in boxes) - margin)
    y0 = max(0, min(b[1] for b in boxes) - margin)
    x1 = min(size[0], max(b[2] for b in boxes) + margin)
    y1 = min(size[1], max(b[3] for b in boxes) + margin)
    return int(x0), int(y0), int(x1), int(y1)


def shape_mask(size, boxes, shape="Oval", grow=0.0):
    """White where the boxes are: an oval (or the box) per detection, grown by grow px."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in boxes:
        rect = (x0 - grow, y0 - grow, x1 + grow, y1 + grow)
        if shape == "Box":
            draw.rectangle(rect, fill=255)
        else:
            draw.ellipse(rect, fill=255)
    return mask


def _block(img, boxes, strength):
    """Mosaic block size: strength % of the smallest area's short side, and never under 1 % of the
    image's long side (the usual rule for a mosaic that really hides), 4 px at least."""
    short = min(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
    return max(4, round(short * strength / 100), math.ceil(max(img.size) / 100))


def mosaic(img, boxes, strength, shape="Oval"):
    """Pixel blocks over the boxes. The blocks sit on one grid over the whole image, so areas
    next to each other line up; with Oval, every block the oval touches is covered (nothing of
    the part peeks out between the blocks and the oval's edge)."""
    block = _block(img, boxes, strength)
    x0, y0, x1, y1 = _union(boxes, img.size)
    # the region, snapped outwards to the block grid
    gx0, gy0 = (x0 // block) * block, (y0 // block) * block
    gx1, gy1 = min(img.width, -(-x1 // block) * block), min(img.height, -(-y1 // block) * block)
    region = img.crop((gx0, gy0, gx1, gy1))
    cols, rows = max(1, math.ceil((gx1 - gx0) / block)), max(1, math.ceil((gy1 - gy0) / block))
    pixels = region.resize((cols, rows), Image.BOX).resize((cols * block, rows * block), Image.NEAREST)
    pixels = pixels.crop((0, 0, gx1 - gx0, gy1 - gy0))

    shape_img = shape_mask(img.size, boxes, shape).crop((gx0, gy0, gx1, gy1))
    # a block counts if any of its pixels is inside the shape
    arr = np.asarray(shape_img, dtype=np.uint8)
    padded = np.zeros((rows * block, cols * block), dtype=np.uint8)
    padded[: arr.shape[0], : arr.shape[1]] = arr
    cells = padded.reshape(rows, block, cols, block).max(axis=(1, 3))
    cell_mask = Image.fromarray(np.repeat(np.repeat(cells, block, 0), block, 1)).crop((0, 0, gx1 - gx0, gy1 - gy0))
    img.paste(Image.composite(pixels, region, cell_mask), (gx0, gy0))


def blur(img, boxes, strength, shape="Oval"):
    """A heavy blur over the boxes, fading out softly past their edge (no hard frame)."""
    short = min(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
    radius = max(8, short * strength / 100 * 1.5)
    fade = max(4, short * 0.12)
    margin = int(radius * 2 + fade * 2)
    x0, y0, x1, y1 = _union(boxes, img.size, margin)
    region = img.crop((x0, y0, x1, y1))
    # shrink first: no detail survives a blur of a picture that small
    small = region.resize((max(1, region.width // 4), max(1, region.height // 4)), Image.BOX)
    blurred = small.filter(ImageFilter.GaussianBlur(radius / 4)).resize(region.size, Image.BICUBIC)
    mask = shape_mask(img.size, boxes, shape, grow=fade / 2).crop((x0, y0, x1, y1))
    mask = mask.filter(ImageFilter.GaussianBlur(fade / 3))
    img.paste(Image.composite(blurred, region, mask), (x0, y0))


def bar(img, box):
    ImageDraw.Draw(img).rectangle(box, fill=(0, 0, 0))


def label(img, box, text):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=max(2, min(w, h) // 10), fill=(0, 0, 0))
    text = (text or "CENSORED").strip() or "CENSORED"
    # the largest size that fits 86% of the width and 60% of the height
    size = max(8, int(h * 0.6))
    while size > 8:
        font = _font(size)
        l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=max(1, size // 18))
        if r - l <= w * 0.86 and b - t <= h * 0.6:
            break
        size = int(size * 0.9)
    font = _font(size)
    stroke = max(1, size // 18)
    l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    draw.text((x0 + (w - (r - l)) / 2 - l, y0 + (h - (b - t)) / 2 - t), text, font=font, fill=(255, 255, 255),
              stroke_width=stroke, stroke_fill=(255, 255, 255))


def sticker(img, box, choice, part_label, scale, seed):
    """One sticker (or a row, for 🍆💦) centred on the box, a little tilted."""
    name = BY_PART.get(part_label, "🔞") if choice == AUTO else choice
    codes = EMOJI.get(name, EMOJI["🔞"])
    x0, y0, x1, y1 = box
    side = max(8, int(max(x1 - x0, y1 - y0) * scale))
    # the first emoji covers the area; a second one (💦) sits over its upper right corner
    extra = int(side * 0.5) if len(codes) > 1 else 0
    row = Image.new("RGBA", (side + extra // 2, side + extra // 2), (0, 0, 0, 0))
    row.alpha_composite(_sticker(codes[0]).resize((side, side), Image.LANCZOS), (0, extra // 2))
    for code in codes[1:]:
        row.alpha_composite(_sticker(code).resize((extra, extra), Image.LANCZOS), (row.width - extra, 0))
    row = _outline(row, max(2, side // 28))
    angle = random.Random(seed + x0 * 7 + y0 * 13).uniform(-14, 14)
    row = row.rotate(angle, resample=Image.BICUBIC, expand=True)
    # centre the first emoji (not the whole row) on the area
    cx, cy = (x0 + x1) // 2 + extra // 4, (y0 + y1) // 2 - extra // 4
    img.paste(row, (cx - row.width // 2, cy - row.height // 2), row)


def _outline(rgba, width):
    """A white border around the sticker's shape, like a die-cut sticker."""
    pad = width * 2
    canvas = Image.new("RGBA", (rgba.width + pad * 2, rgba.height + pad * 2), (0, 0, 0, 0))
    canvas.alpha_composite(rgba, (pad, pad))
    alpha = (np.asarray(canvas.split()[-1]) > 24).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    border = Image.fromarray(cv2.dilate(alpha, kernel)).filter(ImageFilter.GaussianBlur(0.8))
    out = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    out.putalpha(border)
    out.alpha_composite(canvas)
    return out


def apply(image, detections, style, strength=10, text="CENSORED", sticker_choice=AUTO, scale=1.2,
          under=True, seed=0, shape="Oval"):
    """A censored copy of image; detections carry .box (already padded) and .label."""
    img = image.convert("RGB").copy()
    boxes = [d.box for d in detections]
    if not boxes:
        return img
    if style == "Mosaic":
        mosaic(img, boxes, strength, shape)
    elif style == "Blur":
        blur(img, boxes, strength, shape)
    elif style == "Sticker" and under:
        mosaic(img, boxes, strength, shape)
    if style in ("Black bar", "Label"):
        # overlapping areas become one bar / one label, not two printed over each other
        from lib_peekaboo.detector import merge

        for box in merge(boxes):
            if style == "Black bar":
                bar(img, box)
            else:
                label(img, box, text)
    elif style == "Sticker":
        for d in detections:
            sticker(img, d.box, sticker_choice, d.label, scale, seed)
    return img
