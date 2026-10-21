"""Peekaboo Censor: cover exposed body parts in the finished images.

A script in the Script dropdown of txt2img and img2img. It lets the WebUI make
the images as usual (Hires fix, ADetailer, Peekaboo Refine, colour grading,
upscalers: everything), then finds the parts with the NudeNet detector in each
finished image and covers them: mosaic, blur, black bar, a label or stickers.
It saves the censored images, and the untouched ones too if asked.
"""

import json
import os
import sys
import traceback

import gradio as gr
from modules import images, processing, scripts, shared
from modules.shared import opts

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from lib_peekaboo import censor, common, detector  # noqa: E402

TAG = common.TAG
KEY = "Peekaboo censor"  # the PNG info entry


class Script(scripts.Script):
    def title(self):
        return "Peekaboo Censor"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        saved = common.saved_defaults(__file__, tab)
        d = common.PRESETS[common.DEFAULT_PRESET]

        def eid(name):
            return f"peekaboo_censor_{name}_{tab}"

        # which controls show depends on the censor style; start from the saved default
        style_now = saved("Censor style", "Mosaic")

        preset = gr.Radio(list(common.PRESETS), value=common.DEFAULT_PRESET, label="Image style",
                          elem_id=eid("preset"),
                          info="Sets detection for this kind of image; fine-tune below if needed.")
        parts = gr.Dropdown(list(detector.PARTS), value=detector.DEFAULT_PARTS, multiselect=True,
                            label="Parts", elem_id=eid("parts"))
        style = gr.Dropdown(censor.STYLES, value="Mosaic", label="Censor style", elem_id=eid("style"))
        shape = gr.Radio(censor.SHAPES, value="Oval", label="Shape", elem_id=eid("shape"),
                         visible=style_now in censor.SIZED,
                         info="Oval follows the body part; Box covers the whole detected box.")
        strength = gr.Slider(3, 30, value=10, step=1, label="Mosaic block / blur size (%)",
                             elem_id=eid("strength"), visible=style_now in censor.SIZED,
                             info="Of the area's size; a mosaic block is never under 1% of the image.")
        text = gr.Textbox(value="CENSORED", label="Label text", elem_id=eid("text"), max_lines=1,
                          visible=style_now == "Label")
        sticker = gr.Dropdown(censor.STICKERS, value=censor.AUTO, label="Sticker", elem_id=eid("sticker"),
                              visible=style_now == "Sticker")
        scale = gr.Slider(0.8, 2.0, value=1.3, step=0.05, label="Sticker size", elem_id=eid("scale"),
                          visible=style_now == "Sticker")
        under = gr.Checkbox(value=True, label="Mosaic under stickers", elem_id=eid("under"),
                            visible=style_now == "Sticker")
        keep = gr.Checkbox(value=True, label="Keep the uncensored image too", elem_id=eid("keep"))

        with gr.Accordion("Fine-tuning (set by the image style)", open=False, elem_id=eid("fine")):
            detect = common.detection_controls(eid, d)

        common.bind_preset(preset, detect, common.DETECT_KEYS)

        def show_style(st):
            return (gr.update(visible=st in censor.SIZED), gr.update(visible=st in censor.SIZED),
                    gr.update(visible=st == "Label"),
                    gr.update(visible=st == "Sticker"), gr.update(visible=st == "Sticker"),
                    gr.update(visible=st == "Sticker"))

        style.change(show_style, inputs=[style], outputs=[shape, strength, text, sticker, scale, under],
                     show_progress=common.NO_PROGRESS)

        return [preset, parts, detect["model"], detect["search"], detect["score"], detect["pad"], style, strength,
                text, sticker, scale, under, keep, shape]

    # ---------------------------------------------------------------- run

    def run(self, p, preset, parts, model, search, score, pad, style, strength, text, sticker, scale, under, keep,
            shape="Oval"):
        would_save = opts.samples_save and not p.do_not_save_samples
        would_save_grid = opts.grid_save and not p.do_not_save_grid
        p.do_not_save_samples = True  # saved here, after censoring
        p.do_not_save_grid = True
        p._peekaboo_censor = True  # tells Peekaboo Refine not to keep a copy of its own

        proc = processing.process_images(p)

        labels = common.labels_of(parts)
        first = proc.index_of_first_image
        count = len(proc.all_seeds) if proc.all_seeds else len(proc.images) - first
        generated = proc.images[first:first + count]
        rest = proc.images[first + count:]
        infos = list(proc.infotexts)

        out_images, out_infos, results = [], [], []
        for i, image in enumerate(generated):
            info = infos[first + i] if first + i < len(infos) else (infos[-1] if infos else "")
            seed = proc.all_seeds[i] if i < len(proc.all_seeds) else p.seed
            prompt = proc.all_prompts[i] if i < len(proc.all_prompts) else p.prompt
            censored, note = None, None
            if labels and not shared.state.interrupted:
                try:
                    found = common.find(image, i, model, search, score, pad, labels)
                    if found:
                        censored = censor.apply(image, found, style, float(strength), text, sticker, float(scale),
                                                bool(under), int(seed), shape)
                        if style == "Sticker":
                            what = "auto sticker" if sticker == censor.AUTO else f"sticker {sticker}"
                        else:
                            what = style.lower()
                        style_note = f"{preset.lower()}: " if preset in common.PRESETS else ""
                        note = f"{style_note}{what} ({common.parts_note(found)})"
                except Exception as exc:
                    traceback.print_exc()
                    print(f"{TAG} image {i + 1} left as it is: {exc}")
                    detector.warnings.append(f"Censor: image {i + 1} left as it is: {exc}")

            entries = [(image, info, "")]
            if censored is not None:
                censored_info = f"{info}, {KEY}: {json.dumps(note, ensure_ascii=False)}" if info else \
                    f"{KEY}: {json.dumps(note, ensure_ascii=False)}"
                entries = [(censored, censored_info, "-censored")] + ([(image, info, "")] if keep else [])
            results.append(entries[0][0])
            for img, text_info, suffix in entries:
                if text_info and opts.enable_pnginfo:
                    img.info["parameters"] = text_info
                out_images.append(img)
                out_infos.append(text_info)
                if would_save:
                    images.save_image(img, p.outpath_samples, "", seed, prompt, opts.samples_format,
                                      info=text_info, p=p, suffix=suffix)

        # the grid is made again from the results, so a censored batch has a censored grid
        grid = []
        if first > 0 and len(results) > 1:
            g = images.image_grid(results, p.batch_size)
            grid = [g]
            if would_save_grid:
                images.save_image(g, p.outpath_grids, "grid", proc.all_seeds[0] if proc.all_seeds else p.seed,
                                  proc.all_prompts[0] if proc.all_prompts else p.prompt, opts.grid_format,
                                  info=infos[0] if infos else "", short_filename=not opts.grid_extended_filename,
                                  p=p, grid=True)

        if detector.warnings:
            proc.comments = (proc.comments or "") + "".join(f"{TAG} {w}\n" for w in dict.fromkeys(detector.warnings))
        common.flush_warnings()
        proc.images = grid + out_images + rest
        proc.infotexts = (infos[:1] if grid else []) + out_infos + infos[first + count:]
        proc.index_of_first_image = len(grid)
        return proc
