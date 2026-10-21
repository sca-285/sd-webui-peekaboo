# Peekaboo: Detailed Settings and Implementation Notes

This document complements the [README](../README.md). The defaults below describe the built-in starting presets; they are not guaranteed to be optimal for every checkpoint or image.

## When each part runs

For each image, the usual sequence is:

1. The image is generated, including Hires fix.
2. ADetailer and other face/hand detailers run.
3. **Peekaboo Refine** redraws the detected areas.
4. Other post-processing scripts, colour-grading extensions and the upscalers under **Settings → Postprocessing** run.
5. **Peekaboo Censor** covers detected areas in the finished image, then saves the censored outputs and builds the grid.

Refine places itself immediately after ADetailer in the WebUI's `postprocess_image` sequence at the start of a generation. A script ordered before ADetailer remains before it. Censor runs later, so its output includes post-processing and upscaling. Images saved independently by another extension before Censor runs are not censored.

## How Refine redraws areas

Nearby areas are grouped into one pass. For each group, Refine:

1. Crops a surrounding area using **Context**, so the model sees the surrounding body.
2. Scales the crop to the model's generation resolution.
3. Inpaints with a soft oval mask, preserving a soft edge through the inpainting pass.
4. Scales the result back, matches its colours to the surrounding image when enabled, and blends it into the original.

Refine uses the loaded checkpoint for inpainting. Its realism and anatomy therefore depend on that model. The inpainting pass does not reuse other generation scripts such as ControlNet.

## How Censor covers areas

- **Mosaic:** blocks use a shared image grid, so adjacent areas line up. With the Oval shape, every block touched by the oval is covered.
- **Blur:** the image is reduced before blurring to remove fine detail; the blur fades past the area edge.
- **Black bar** and **Label:** overlapping detections are combined so bars or labels are not repeatedly stamped over each other.
- Overlapping areas of different types are all covered; duplicate detections of the same area are covered once.
- **Sticker:** the area can be mosaicked underneath a sticker to prevent gaps from exposing the original image.

## Image-style presets

Choosing a style fills the detection settings and, for Refine, the inpainting settings. Any value can then be adjusted under **Fine-tuning**. Censor uses the detection settings only.

| Setting | Photo | Semi-real | Anime & cartoon |
|---|---:|---:|---:|
| Intended for | Photographs and photoreal checkpoints | 2.5D, realistic renders and painterly work | Anime, cartoons, flat colours and line art |
| Detection confidence | 0.35 | 0.28 | 0.20 |
| Search | Normal | Also mirrored | Mirrored + close-up tiles |
| Area padding | 10% | 15% | 20% |
| Refine denoising | 0.32 | 0.36 | 0.42 |
| Refine edge softness | 0.30 | 0.25 | 0.18 |
| Refine context | 2.5× | 2.5× | 2.2× |
| Added to refine prompt | `detailed skin texture, natural skin, realistic anatomy` | `detailed skin, soft shading, correct anatomy` | `clean lineart, detailed, correct anatomy` |
| Added to refine negative prompt | `plastic skin, airbrushed, blurry, deformed, bad anatomy` | `blurry, deformed, bad anatomy` | `blurry, sketch, deformed, bad anatomy` |

NudeNet is mainly trained on photographs, so stylized images generally benefit from a lower confidence threshold, additional search passes and wider padding. The built-in presets use the 640m detector.

## Settings reference

### Shared detection settings

| Setting | Description |
|---|---|
| **Image style** | Photo, Semi-real or Anime & cartoon. Selects starting values. |
| **Parts** | Breasts (female), genitals (female), genitals (male), anus, buttocks, chest (male), belly, armpits and feet. Only detected exposed parts are processed. |
| **Detector** | **640m (accurate)** is generally more thorough, especially on drawn or anime images. **320n (fast)** is quicker. |
| **Search** | **Normal** uses one pass; **Also mirrored** adds a pass on a mirrored image; **Mirrored + close-up tiles** also checks four overlapping tiles at detector resolution. Findings are merged. |
| **Detection confidence** | Lower values can detect more areas but may increase false detections. |
| **Area padding** | Expands each detected box on every side, as a percentage of its size. |

### Peekaboo Refine

| Setting | Description |
|---|---|
| **Add to prompt / negative prompt** | Text appended to the image's original prompts, including its existing LoRA terms. The image-style preset supplies starting text. |
| **Denoising strength** | Around 0.3–0.4 is a reasonable starting range for adding detail while retaining shape. Higher values redraw more. |
| **Keep the unrefined image too** | Displays and saves the image before Refine with the `-original` suffix, unless Censor is handling the pair. |
| **Steps** | `0` follows the source image's step count. Refine runs the full configured step count even at lower denoising strength to reduce unfinished grain. |
| **Edge softness** | Width of the blend near the boundary, relative to the detected area's size. |
| **Context** | How much surrounding image is included, expressed as a multiple of the area's size. |
| **Smooth sampler** | On by default. Noise-adding samplers such as Euler a and SDE samplers are replaced by calmer counterparts for the refine pass. |
| **Match colours** | On by default. Adjusts the redrawn area to better match the tone of the surrounding image. |

### Peekaboo Censor

| Setting | Description |
|---|---|
| **Censor style** | Mosaic, Blur, Black bar, Label or Sticker. |
| **Shape** | For mosaic, blur and the mosaic under stickers: Oval follows the detected part; Box covers the full detection box. |
| **Mosaic block / blur size** | Size relative to the detected area. Mosaic blocks have a minimum size relative to the image to keep small detections covered. |
| **Label text** | `CENSORED` by default; can be changed to another short label. |
| **Sticker** | Choose a sticker or **Auto by part**, which picks a sticker based on the detected part. |
| **Sticker size** | Sticker size relative to the detection area. |
| **Mosaic under stickers** | Applies mosaic before placing a sticker and adds a white die-cut border. |
| **Keep the uncensored image too** | Saves the processed but uncensored image next to the censored result. When Refine is enabled, this is the refined version. |

## Saving and generation information

- Refine saves the refined image as the main result; if kept separately, the original receives `-original`.
- Censor saves the covered image with `-censored`. If **Keep the uncensored image too** is enabled, the uncensored version is saved as well. The batch grid is rebuilt from the censored images.
- PNG generation information records the applied Refine and Censor settings.
- To save UI settings as defaults, set the controls, then open **Settings → Defaults**, choose **View changes**, review the changes, and press **Apply**. Defaults apply after restarting the UI and are maintained separately for txt2img and img2img.
- Images for which no parts are detected are left unchanged.

## API

- **Refine** uses `alwayson_scripts: {"peekaboo refine": {"args": [...]}}`. Arguments cover enablement, image style, parts, detector, search, confidence, padding, prompt additions, denoising, steps, edge softness, context, keeping the original, sampler smoothing and colour matching.
- **Censor** uses `script_name: "peekaboo censor"` and `script_args: [...]`. Arguments cover image style, parts, detector, search, confidence, padding, censor style, mosaic/blur size, label text, sticker, sticker size, mosaic under stickers, keeping the uncensored image and shape.

Check your WebUI's API conventions if automating these scripts; script argument order can change when controls are added.
