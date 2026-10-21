"""What the two Peekaboo scripts share: image style presets and small helpers."""

import json
import os

import gradio as gr

from lib_peekaboo import detector

TAG = "[Peekaboo]"

# Gradio 4 names it, Gradio 3 (reForge) takes a bool
NO_PROGRESS = "hidden" if int(gr.__version__.split(".")[0]) >= 4 else False

# Starting points by image style. Choosing one fills the controls; any of them can be changed after.
PRESETS = {
    "Photo": dict(
        model="640m (accurate)", score=0.35, pad=10, search=detector.SEARCH_NORMAL,
        denoise=0.32, soft=0.3, crop=2.5,
        add_prompt="detailed skin texture, natural skin, realistic anatomy",
        add_negative="plastic skin, airbrushed, blurry, deformed, bad anatomy",
    ),
    "Semi-real": dict(
        model="640m (accurate)", score=0.28, pad=15, search=detector.SEARCH_MIRROR,
        denoise=0.36, soft=0.25, crop=2.5,
        add_prompt="detailed skin, soft shading, correct anatomy",
        add_negative="blurry, deformed, bad anatomy",
    ),
    "Anime & cartoon": dict(
        model="640m (accurate)", score=0.2, pad=20, search=detector.SEARCH_TILES,
        denoise=0.42, soft=0.18, crop=2.2,
        add_prompt="clean lineart, detailed, correct anatomy",
        add_negative="blurry, sketch, deformed, bad anatomy",
    ),
}
DEFAULT_PRESET = "Semi-real"
DETECT_KEYS = ["model", "score", "pad", "search"]
REFINE_KEYS = ["denoise", "soft", "crop", "add_prompt", "add_negative"]


def saved_defaults(script_file, tab):
    """A reader of a script's defaults saved in ui-config.json (by label). The WebUI applies them
    only after the whole UI is built; the scripts need some earlier, to show the right controls."""
    values = {}
    try:
        from modules import shared

        path = shared.cmd_opts.ui_config_file
        if os.path.isfile(path):
            with open(path, encoding="utf8") as f:
                values = json.load(f)
    except Exception:
        pass
    prefix = f"customscript/{os.path.basename(script_file)}/{tab}/"

    def get(label, default):
        return values.get(f"{prefix}{label}/value", default)

    return get


def join_prompt(base, extra):
    base, extra = (base or "").strip(), (extra or "").strip()
    if not extra:
        return base
    return f"{base.rstrip(',')}, {extra}" if base else extra


def detection_controls(eid, d):
    """Detector, search, confidence and padding: the detection part of Fine-tuning."""
    model = gr.Radio(list(detector.MODELS), value=d["model"], label="Detector", elem_id=eid("model"))
    search = gr.Radio(detector.SEARCHES, value=d["search"], label="Search", elem_id=eid("search"),
                      info="More passes find more, a little slower.")
    score = gr.Slider(0.1, 0.9, value=d["score"], step=0.01, label="Detection confidence",
                      elem_id=eid("score"), info="Lower finds more, with more false alarms.")
    pad = gr.Slider(0, 60, value=d["pad"], step=1, label="Area padding (%)", elem_id=eid("pad"))
    return dict(model=model, search=search, score=score, pad=pad)


def bind_preset(preset, controls, keys):
    """Choosing an image style fills the given controls."""

    def apply(name):
        values = PRESETS.get(name)
        if not values:
            return [gr.update() for _ in keys]
        return [gr.update(value=values[k]) for k in keys]

    preset.change(apply, inputs=[preset], outputs=[controls[k] for k in keys], show_progress=NO_PROGRESS)


def labels_of(parts):
    return {detector.PARTS[x] for x in (parts or []) if x in detector.PARTS}


def flush_warnings(p=None):
    """Print the detector's warnings, and add them to the generation's comments."""
    if not detector.warnings:
        return
    for w in dict.fromkeys(detector.warnings):
        print(f"{TAG} {w}")
        if p is not None and isinstance(getattr(p, "comments", None), dict):
            p.comments[f"{TAG} {w}"] = 1
    detector.warnings.clear()


def find(image, index, model, search, score, pad, labels):
    """Detections with padded boxes, or [] (printed either way)."""
    found = detector.detect(image, model, float(score), labels, search)
    if not found:
        print(f"{TAG} image {index + 1}: nothing found.")
        return []
    print(f"{TAG} image {index + 1}: {', '.join(map(repr, found))}")
    for d in found:
        d.box = detector.expand(d.box, image.size, float(pad) / 100)
    return found


def parts_note(found):
    return ", ".join(sorted({d.part for d in found})).lower()
