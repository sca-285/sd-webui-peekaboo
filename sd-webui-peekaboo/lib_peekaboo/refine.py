"""Refine: redraw detected areas in detail and blend them back without a seam.

The loop, in the spirit of Impact Pack's FaceDetailer:

1. crop a region around the area, a few times its size, so the model sees the
   body around it (context);
2. scale that crop to the model's own resolution;
3. inpaint it with a soft oval mask (not a box), the latent mask kept soft
   (not rounded to 0/1), the full number of steps (not steps x denoise) and,
   by default, a sampler without added noise;
4. scale it back, match its colours to the original under the mask, and blend
   it in with the same soft oval.

Areas whose crops overlap are redrawn together, in one pass.
"""

import dataclasses
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from modules import processing, scripts, shared
from modules.processing import StableDiffusionProcessingImg2Img

# p attributes copied into the inpaint pass when both sides have them
COPY = [
    "sd_model", "outpath_samples", "outpath_grids", "styles", "subseed_strength", "seed_resize_from_h",
    "seed_resize_from_w", "cfg_scale", "distilled_cfg_scale", "tiling", "clip_skip",
]

# samplers that add fresh noise at every step (grain in a small redrawn area) -> a calm counterpart
CALM = {
    "Euler a": "Euler",
    "DPM++ 2S a": "DPM++ 2M",
    "DPM++ SDE": "DPM++ 2M",
    "DPM++ 2M SDE": "DPM++ 2M",
    "DPM++ 2M SDE Heun": "DPM++ 2M",
    "DPM++ 3M SDE": "DPM++ 2M",
    "DPM2 a": "DPM2",
    "DPM adaptive": "DPM++ 2M",
    "Restart": "DPM++ 2M",
    "LMS": "DPM++ 2M",
}


def _round16(v):
    return max(64, int(round(v / 16)) * 16)


def _grow(box, factor, size):
    """box scaled by factor around its centre, kept inside the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * factor, (y1 - y0) * factor
    W, H = size
    # keep the crop inside the image by sliding it, not by squashing it
    w, h = min(w, W), min(h, H)
    left = min(max(0, cx - w / 2), W - w)
    top = min(max(0, cy - h / 2), H - h)
    return int(left), int(top), int(math.ceil(left + w)), int(math.ceil(top + h))


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def groups(boxes, crop_factor, size):
    """Boxes whose crops overlap, together: [(crop region, [boxes])]."""
    items = [(_grow(b, crop_factor, size), [b]) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if _overlaps(items[i][0], items[j][0]):
                    boxes_ij = items[i][1] + items[j][1]
                    union = (min(b[0] for b in boxes_ij), min(b[1] for b in boxes_ij),
                             max(b[2] for b in boxes_ij), max(b[3] for b in boxes_ij))
                    region = _grow(union, 1.0, size)
                    # the union's own crop, at least as big as both crops
                    a, b = items[i][0], items[j][0]
                    region = (min(region[0], a[0], b[0]), min(region[1], a[1], b[1]),
                              max(region[2], a[2], b[2]), max(region[3], a[3], b[3]))
                    items[i] = (region, boxes_ij)
                    del items[j]
                    merged = True
                    break
            if merged:
                break
    return items


def oval_mask(size, boxes, softness, offset=(0, 0), scale=1.0):
    """A mask with a filled oval per box, its edge faded over softness x the oval's short side.

    The fade straddles the oval's edge: the oval is drawn half a fade bigger, then blurred,
    so the middle of the area is fully redrawn and the blend reaches a little past it."""
    ovals = []
    for x0, y0, x1, y1 in boxes:
        ovals.append(((x0 - offset[0]) * scale, (y0 - offset[1]) * scale,
                      (x1 - offset[0]) * scale, (y1 - offset[1]) * scale))
    fade = max((min(x1 - x0, y1 - y0) * softness for x0, y0, x1, y1 in ovals), default=0.0)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in ovals:
        g = fade / 2
        draw.ellipse((x0 - g, y0 - g, x1 + g, y1 + g), fill=255)
    if fade >= 2:
        mask = mask.filter(ImageFilter.GaussianBlur(fade / 3))
    return mask


def match_colours(new, old, alpha, strength=0.85):
    """Shift new's colours (Lab mean, under alpha) towards old's: no tone jump at the edge."""
    a = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    total = float(a.sum())
    if total < 1:
        return new
    lab_new = cv2.cvtColor(np.asarray(new.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_old = cv2.cvtColor(np.asarray(old.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_new = (lab_new * a).sum((0, 1)) / total
    mean_old = (lab_old * a).sum((0, 1)) / total
    lab_new += (mean_old - mean_new) * strength
    out = cv2.cvtColor(np.clip(lab_new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def _sampler(p, calm):
    name = getattr(p, "sampler_name", None)
    return CALM.get(name, name) if calm else name


def _target_size(p, region):
    """The crop's size for the model: its long side at the model's native resolution."""
    native = math.sqrt(max(64, getattr(p, "width", 1024)) * max(64, getattr(p, "height", 1024)))
    native = min(max(native, 512), 1536)
    w, h = region[2] - region[0], region[3] - region[1]
    scale = native / max(w, h)
    return _round16(w * scale), _round16(h * scale)


def _inpaint(p, crop, mask, prompt, negative, seed, subseed, denoise, steps, calm):
    fields = {f.name for f in dataclasses.fields(StableDiffusionProcessingImg2Img)}
    kwargs = dict(
        init_images=[crop],
        resize_mode=0,
        denoising_strength=denoise,
        mask=mask,
        mask_blur=0,  # the mask is already soft
        mask_round=False,  # keep it soft in the latent too: this is what hides the edge
        inpainting_fill=1,  # start from the original pixels
        inpaint_full_res=False,  # the crop is already the area, at the model's size
        inpainting_mask_invert=0,
        prompt=prompt,
        negative_prompt=negative,
        seed=seed,
        subseed=subseed,
        batch_size=1,
        n_iter=1,
        steps=steps,
        width=crop.width,
        height=crop.height,
        sampler_name=_sampler(p, calm),
        scheduler=getattr(p, "scheduler", None),
        restore_faces=False,
        do_not_save_samples=True,
        do_not_save_grid=True,
    )
    if "mask_blur_x" in fields:
        kwargs.update(mask_blur_x=0, mask_blur_y=0)
    for name in COPY:
        if name in fields and hasattr(p, name):
            kwargs[name] = getattr(p, name)
    # the full number of steps, not steps x denoise: fewer steps leave grain
    overrides = dict(getattr(p, "override_settings", None) or {})
    overrides["img2img_fix_steps"] = True
    overrides.setdefault("img2img_color_correction", False)
    kwargs["override_settings"] = overrides
    kwargs["override_settings_restore_afterwards"] = True
    kwargs = {k: v for k, v in kwargs.items() if k in fields and v is not None}

    i2i = StableDiffusionProcessingImg2Img(**kwargs)
    # no scripts in this pass (this one included), but a runner: some WebUIs look into it
    i2i.scripts = scripts.ScriptRunner()
    i2i.script_args = []
    i2i._peekaboo_pass = True
    try:
        result = processing.process_images(i2i)
    finally:
        i2i.close()
    return result.images[0] if result.images else None


def run(p, image, boxes, prompt, negative, seed, subseed, denoise, steps, softness, crop_factor,
        calm=True, match=True):
    """Redraw the boxes (already padded) of image; returns the new image."""
    image = image.convert("RGB")
    out = image.copy()
    for index, (region, members) in enumerate(groups(boxes, crop_factor, image.size)):
        if shared.state.interrupted or shared.state.skipped:
            break
        rw, rh = region[2] - region[0], region[3] - region[1]
        if rw < 8 or rh < 8:
            continue
        tw, th = _target_size(p, region)
        scale = tw / rw
        crop = out.crop(region).resize((tw, th), Image.LANCZOS)
        mask = oval_mask((tw, th), members, softness, offset=region[:2], scale=scale)

        new = _inpaint(p, crop, mask, prompt, negative, seed + index, subseed + index, denoise, steps, calm)
        if new is None:
            continue
        new = new.convert("RGB")
        if new.size != (tw, th):
            new = new.resize((tw, th), Image.LANCZOS)
        new = new.resize((rw, rh), Image.LANCZOS)
        alpha = oval_mask((rw, rh), members, softness, offset=region[:2])
        old = out.crop(region)
        if match:
            new = match_colours(new, old, alpha)
        out.paste(Image.composite(new, old, alpha), region[:2])
    return out
