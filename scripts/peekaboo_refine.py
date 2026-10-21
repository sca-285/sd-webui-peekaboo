"""Peekaboo Refine: redraw exposed body parts in detail, right after ADetailer.

An always-on block in txt2img and img2img, like ADetailer. For each image it
finds the parts with the NudeNet detector and inpaints each area again at full
resolution with the image's own prompt, seed and sampler.

It works in the postprocess_image step, which the WebUI runs for each image
before saving it, and puts itself right after ADetailer there: ADetailer fixes
faces and hands, then Peekaboo Refine, then everything else (other
post-processing, colour grading in postprocess_image_after_composite, the
Settings > Postprocessing upscalers, saving, the grid). Peekaboo Censor, in the
Script dropdown, runs after all of it.
"""

import os
import sys
import traceback

import gradio as gr
from modules import images, processing, scripts, shared
from modules.shared import opts
from modules.ui_components import InputAccordion

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from lib_peekaboo import common, detector  # noqa: E402

TAG = common.TAG
KEY = "Peekaboo refine"  # the PNG info entry
# callbacks of face / hand detailers, by (lower-case) name: ADetailer, uddetailer, DDSD...
DETAILERS = ("adetailer", "detailer", "ddsd")


def place_after_detailers(runner):
    """Move this script's postprocess_image hook right after the last detailer's.

    The WebUI keeps the ordered hooks in a cached list; reordering it holds until scripts are
    added or removed. Called at the start of every generation, so it stays in place."""
    try:
        hooks = runner.ordered_callbacks("postprocess_image")
    except Exception:
        return
    mine = [h for h in hooks if isinstance(h.callback, Script)]
    if not mine:
        return
    for h in mine:
        hooks.remove(h)
    detailers = [i for i, h in enumerate(hooks) if any(d in h.name.lower() for d in DETAILERS)]
    at = detailers[-1] + 1 if detailers else 0
    hooks[at:at] = mine


class Script(scripts.Script):
    sorting_priority = 95  # near ADetailer in the list of blocks

    def title(self):
        return "Peekaboo Refine"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        d = common.PRESETS[common.DEFAULT_PRESET]

        def eid(name):
            return f"peekaboo_refine_{name}_{tab}"

        with InputAccordion(False, label="Peekaboo Refine 🙈", elem_id=eid("enabled")) as enabled:
            preset = gr.Radio(list(common.PRESETS), value=common.DEFAULT_PRESET, label="Image style",
                              elem_id=eid("preset"),
                              info="Sets detection and refine for this kind of image; fine-tune below if needed.")
            parts = gr.Dropdown(list(detector.PARTS), value=detector.DEFAULT_PARTS, multiselect=True,
                                label="Parts", elem_id=eid("parts"))
            add_prompt = gr.Textbox(value=d["add_prompt"], label="Add to prompt", lines=1,
                                    elem_id=eid("add_prompt"), placeholder="Added to the image's own prompt")
            add_negative = gr.Textbox(value=d["add_negative"], label="Add to negative prompt", lines=1,
                                      elem_id=eid("add_negative"), placeholder="Added to the image's own negative")
            denoise = gr.Slider(0.05, 1.0, value=d["denoise"], step=0.01, label="Denoising strength",
                                elem_id=eid("denoise"), info="0.3-0.4 keeps the shape and adds detail.")
            keep = gr.Checkbox(value=False, label="Keep the unrefined image too", elem_id=eid("keep"))

            with gr.Accordion("Fine-tuning (set by the image style)", open=False, elem_id=eid("fine")):
                detect = common.detection_controls(eid, d)
                steps = gr.Slider(0, 150, value=0, step=1, label="Steps (0: as the image)", elem_id=eid("steps"),
                                  info="All of them are run, whatever the denoising strength.")
                soft = gr.Slider(0.0, 0.6, value=d["soft"], step=0.01, label="Edge softness", elem_id=eid("soft"),
                                 info="How wide the blend at the edge is, as a part of the area's size.")
                crop = gr.Slider(1.2, 4.0, value=d["crop"], step=0.1, label="Context (x area)", elem_id=eid("crop"),
                                 info="How much of the body around the area the model sees while redrawing it.")
                calm = gr.Checkbox(value=True, label="Smooth sampler", elem_id=eid("calm"),
                                   info="Euler a, SDE and other noise-adding samplers become their calm "
                                        "counterpart for this pass: less grain.")
                match = gr.Checkbox(value=True, label="Match colours", elem_id=eid("match"),
                                    info="Keeps the redrawn area's tone the same as the image around it.")

        controls = dict(detect, denoise=denoise, soft=soft, crop=crop, add_prompt=add_prompt,
                        add_negative=add_negative)
        common.bind_preset(preset, controls, common.DETECT_KEYS + common.REFINE_KEYS)

        self.infotext_fields = [(enabled, lambda params: KEY in params)]
        return [enabled, preset, parts, detect["model"], detect["search"], detect["score"], detect["pad"],
                add_prompt, add_negative, denoise, steps, soft, crop, keep, calm, match]

    # ---------------------------------------------------------------- run

    def process(self, p, enabled=False, *args):
        if not enabled or getattr(p, "_peekaboo_pass", False) or p.scripts is None:
            return
        place_after_detailers(p.scripts)
        p.extra_generation_params.pop(KEY, None)

    def postprocess_image(self, p, pp, enabled=False, preset=common.DEFAULT_PRESET, parts=(),
                          model=detector.DEFAULT_MODEL, search=detector.SEARCH_NORMAL, score=0.3, pad=15,
                          add_prompt="", add_negative="", denoise=0.36, steps=0, soft=0.25, crop=2.5, keep=False,
                          calm=True, match=True):
        if not enabled or getattr(p, "_peekaboo_pass", False) or shared.state.interrupted:
            return
        index = p.iteration * p.batch_size + getattr(p, "batch_index", 0)
        p.extra_generation_params.pop(KEY, None)
        labels = common.labels_of(parts)
        if not labels:
            return
        image = pp.image
        try:
            found = common.find(image, index, model, search, score, pad, labels)
            result = self.refine(p, index, image, found, add_prompt, add_negative, denoise, steps, soft, crop,
                                 calm, match) if found else None
        except Exception as exc:
            traceback.print_exc()
            print(f"{TAG} image {index + 1} left as it is: {exc}")
            detector.warnings.append(f"Refine: image {index + 1} left as it is: {exc}")
            result = None
        common.flush_warnings(p)
        if result is None:
            return

        style = f"{preset.lower()}: " if preset in common.PRESETS else ""
        note = f"{style}{float(denoise):g} ({common.parts_note(found)})"
        pp.image = result
        p.extra_generation_params[KEY] = note
        # Peekaboo Censor keeps its own unprocessed copy (the refined image); no second one here
        if keep and not getattr(p, "_peekaboo_censor", False):
            self.keep_original(p, index, image, note)

    @staticmethod
    def refine(p, index, image, found, add_prompt, add_negative, denoise, steps, soft, crop, calm, match):
        from lib_peekaboo import refine

        seed = p.all_seeds[index] if index < len(p.all_seeds) else p.seed
        subseed = p.all_subseeds[index] if index < len(p.all_subseeds) else p.subseed
        prompt = p.all_prompts[index] if index < len(p.all_prompts) else p.prompt
        negative = p.all_negative_prompts[index] if index < len(p.all_negative_prompts) else p.negative_prompt
        return refine.run(p, image, [d.box for d in found], common.join_prompt(prompt, add_prompt),
                          common.join_prompt(negative, add_negative), seed, subseed, float(denoise),
                          int(steps) or p.steps, float(soft), float(crop), bool(calm), bool(match))

    @staticmethod
    def keep_original(p, index, image, note):
        """Show the unrefined image after the results, and save it with -original."""
        p.extra_generation_params.pop(KEY, None)
        try:
            info = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds,
                                              iteration=p.iteration, position_in_batch=p.batch_index)
        except Exception:
            info = None
        p.extra_generation_params[KEY] = note
        if opts.samples_save and not p.do_not_save_samples:
            seed = p.all_seeds[index] if index < len(p.all_seeds) else p.seed
            prompt = p.all_prompts[index] if index < len(p.all_prompts) else p.prompt
            images.save_image(image, p.outpath_samples, "", seed, prompt, opts.samples_format, info=info, p=p,
                              suffix="-original")
        if info and opts.enable_pnginfo:
            image.info["parameters"] = info
        p.extra_result_images.append(image)
