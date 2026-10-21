"""Body-part detection with the NudeNet v3 detector models (YOLOv8, ONNX).

The models are downloaded on first use into models/nudenet/ from NudeNet's
GitHub release. They are not shipped with the extension. A model file placed
there by hand (320n.onnx or 640m.onnx) is used as is.
"""

import os
import threading
import zipfile

import cv2
import numpy as np

# The order of the model's output classes.
CLASSES = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

# What the UI offers: exposed parts only, by a readable name.
PARTS = {
    "Breasts (female)": "FEMALE_BREAST_EXPOSED",
    "Genitals (female)": "FEMALE_GENITALIA_EXPOSED",
    "Genitals (male)": "MALE_GENITALIA_EXPOSED",
    "Anus": "ANUS_EXPOSED",
    "Buttocks": "BUTTOCKS_EXPOSED",
    "Chest (male)": "MALE_BREAST_EXPOSED",
    "Belly": "BELLY_EXPOSED",
    "Armpits": "ARMPITS_EXPOSED",
    "Feet": "FEET_EXPOSED",
}
DEFAULT_PARTS = ["Breasts (female)", "Genitals (female)", "Genitals (male)", "Anus"]

MODELS = {
    "320n (fast)": ("320n.onnx", 320),
    "640m (accurate)": ("640m.onnx", 640),
}
DEFAULT_MODEL = "640m (accurate)"
RELEASE = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/"
# 320n also ships inside NudeNet's Python package: a second place to get it.
PYPI_WHEEL = "https://pypi.org/pypi/nudenet/3.4.2/json"


class Detection:
    __slots__ = ("label", "part", "score", "box")

    def __init__(self, label, score, box):
        self.label = label
        self.part = next((k for k, v in PARTS.items() if v == label), label)
        self.score = score
        self.box = box  # (x0, y0, x1, y1) in image pixels

    def __repr__(self):
        return f"{self.part} {self.score:.2f} {self.box}"


def model_dir():
    try:
        from modules import paths
        base = paths.models_path
    except Exception:
        base = os.path.join(os.getcwd(), "models")
    return os.path.join(base, "nudenet")


def _fetch(url, target):
    """Download url to target through a .part file; returns target."""
    import urllib.request

    part = target + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "sd-webui-peekaboo"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        expected = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    size = os.path.getsize(part)
    if expected and size != expected:
        os.remove(part)
        raise RuntimeError(f"the download stopped at {size} of {expected} bytes")
    os.replace(part, target)
    return target


def _from_release(file_name, target):
    return _fetch(RELEASE + file_name, target)


def _from_wheel(file_name, target):
    """Take 320n.onnx out of the nudenet wheel on PyPI."""
    import json
    import urllib.request

    info = json.load(urllib.request.urlopen(PYPI_WHEEL, timeout=30))
    url = next(u["url"] for u in info["urls"] if u["filename"].endswith(".whl"))
    wheel = _fetch(url, target + ".whl")
    try:
        with zipfile.ZipFile(wheel) as whl, whl.open("nudenet/320n.onnx") as src, open(target + ".part", "wb") as dst:
            dst.write(src.read())
        os.replace(target + ".part", target)
    finally:
        os.remove(wheel)
    return target


SOURCES = {"320n.onnx": [_from_release, _from_wheel], "640m.onnx": [_from_release]}


def _check(path):
    """Why the file is not a usable model, or None."""
    import onnxruntime as ort

    size = os.path.getsize(path)
    if size < 1 << 20:  # the models are 12 MB and more: this is an error page or a broken download
        with open(path, "rb") as f:
            head = f.read(64)
        if head.lstrip()[:1] == b"<":
            return f"it is a web page ({size} bytes), not a model"
        return f"it is only {size} bytes"
    try:
        ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        return f"ONNX Runtime cannot read it ({str(exc).split(':')[-1].strip()})"
    return None


def _set_aside(path):
    """Keep a broken model file as .bad (for a look) and out of the way."""
    bad = path + ".bad"
    try:
        os.replace(path, bad)
    except OSError:
        os.remove(path)
    return bad


def ensure_model(name):
    """The local path of a working model, downloading it (again) when needed."""
    file_name, _ = MODELS[name]
    folder = model_dir()
    path = os.path.join(folder, file_name)
    if os.path.isfile(path):
        problem = _check(path)
        if not problem:
            return path
        print(f"[Peekaboo] {path} is broken: {problem}. Moved to {os.path.basename(_set_aside(path))}; "
              f"downloading it again.")
    os.makedirs(folder, exist_ok=True)
    errors = []
    for source in SOURCES[file_name]:
        where = "GitHub" if source is _from_release else "PyPI"
        print(f"[Peekaboo] downloading the {file_name} detector from {where} to {folder}")
        try:
            source(file_name, path)
        except Exception as exc:
            errors.append(f"{where}: {exc}")
            continue
        problem = _check(path)
        if not problem:
            print(f"[Peekaboo] {file_name} ready.")
            return path
        _set_aside(path)
        errors.append(f"{where}: the download is broken ({problem})")
    raise RuntimeError(
        f"Could not get a working {file_name} ({'; '.join(errors)}). Download it from "
        f"{RELEASE + file_name} with a browser and put it in {folder}."
    )


_sessions = {}
_lock = threading.Lock()
FAST = "320n (fast)"
# problems worth showing under the image (the script reads and clears this)
warnings = []


def _load(name):
    import onnxruntime as ort

    # CPU: small models, and it leaves the GPU to the diffusion model
    return ort.InferenceSession(ensure_model(name), providers=["CPUExecutionProvider"])


def _session(name):
    """(session, model name actually used). 640m falls back to 320n when it cannot be had;
    the WebUI is asked again after a restart."""
    with _lock:
        if name not in _sessions:
            try:
                _sessions[name] = (_load(name), name, None)
            except Exception as exc:
                if name == FAST:
                    raise
                print(f"[Peekaboo] {exc} Using 320n instead.")
                if FAST not in _sessions:
                    _sessions[FAST] = (_load(FAST), FAST, None)
                _sessions[name] = (_sessions[FAST][0], FAST, f"{exc} The 320n detector was used instead.")
        session, used, problem = _sessions[name]
        if problem:
            warnings.append(problem)
        return session, used


SEARCH_NORMAL = "Normal"
SEARCH_MIRROR = "Also mirrored"
SEARCH_TILES = "Mirrored + close-up tiles"
SEARCHES = [SEARCH_NORMAL, SEARCH_MIRROR, SEARCH_TILES]


def _raw(session, size, rgb, min_score, wanted):
    """One pass of the model over an RGB array: (boxes x0 y0 x1 y1, scores, classes) in its pixels."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    # square canvas, image at the top left: the model's training layout
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[:h, :w] = rgb
    blob = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    blob = blob.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    pred = np.squeeze(out, 0).T  # (candidates, 4 + classes)
    scores = pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf >= min_score
    if wanted is not None:
        keep &= wanted[cls]
    pred, cls, conf = pred[keep], cls[keep], conf[keep]
    scale = side / size
    cx, cy, bw, bh = (pred[:, i] * scale for i in range(4))
    boxes = np.stack([np.clip(cx - bw / 2, 0, w), np.clip(cy - bh / 2, 0, h),
                      np.clip(cx + bw / 2, 0, w), np.clip(cy + bh / 2, 0, h)], 1) if len(pred) else np.zeros((0, 4))
    return boxes, conf, cls


def _tiles(w, h):
    """Four overlapping tiles (two thirds of each side), for small parts in big images."""
    tw, th = int(w * 2 / 3), int(h * 2 / 3)
    return [(x, y, x + tw, y + th) for y in (0, h - th) for x in (0, w - tw)]


def detect(image, model=DEFAULT_MODEL, min_score=0.35, labels=None, search=SEARCH_NORMAL):
    """Detections in a PIL image, best first. labels: the class names to keep.

    search adds passes: the image mirrored (the detector is not symmetric, a
    mirrored pass finds parts it missed), and four overlapping close-up tiles
    (small parts in large images, drawn styles). Overlapping finds are merged,
    keeping the best score."""
    session, used = _session(model)
    size = MODELS[used][1]
    wanted = None if labels is None else np.array([CLASSES[c] in labels for c in range(len(CLASSES))])

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    all_boxes, all_conf, all_cls = [], [], []

    def add(result, dx=0, dy=0, mirrored_width=None):
        boxes, conf, cls = result
        if not len(boxes):
            return
        boxes = boxes.copy()
        if mirrored_width is not None:
            boxes[:, [0, 2]] = mirrored_width - boxes[:, [2, 0]]
        boxes[:, [0, 2]] += dx
        boxes[:, [1, 3]] += dy
        all_boxes.append(boxes)
        all_conf.append(conf)
        all_cls.append(cls)

    add(_raw(session, size, rgb, min_score, wanted))
    if search in (SEARCH_MIRROR, SEARCH_TILES):
        add(_raw(session, size, np.ascontiguousarray(rgb[:, ::-1]), min_score, wanted), mirrored_width=w)
    if search == SEARCH_TILES and min(w, h) >= 384:
        for x0, y0, x1, y1 in _tiles(w, h):
            add(_raw(session, size, np.ascontiguousarray(rgb[y0:y1, x0:x1]), min_score, wanted), x0, y0)

    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes)
    conf = np.concatenate(all_conf)
    cls = np.concatenate(all_cls)

    xywh = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes]
    # one part found twice (overlapping boxes of the same class) is kept once; different parts
    # that overlap (two bodies close together) are all kept, so none is left uncovered
    picked = []
    for c in np.unique(cls):
        idx = np.nonzero(cls == c)[0]
        keep = np.array(cv2.dnn.NMSBoxes([xywh[i] for i in idx], conf[idx].astype(float).tolist(),
                                         float(min_score), 0.45)).reshape(-1)
        picked.extend(int(idx[k]) for k in keep)

    found = []
    for i in picked:
        x0, y0, x1, y1 = boxes[i]
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        # a find cut by a tile edge: grow it to cover the same part found in other passes
        same = [j for j in range(len(boxes)) if cls[j] == cls[i] and _iou(boxes[i], boxes[j]) > 0.3]
        if len(same) > 1:
            x0, y0 = boxes[same, 0].min(), boxes[same, 1].min()
            x1, y1 = boxes[same, 2].max(), boxes[same, 3].max()
        found.append(Detection(CLASSES[cls[i]], float(conf[i]),
                               (int(x0), int(y0), int(round(x1)), int(round(y1)))))
    found.sort(key=lambda d: -d.score)
    return found


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def expand(box, size, pad):
    """A box grown by pad (a fraction of its own size) on every side, inside the image."""
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * pad
    dy = (y1 - y0) * pad
    w, h = size
    return (max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(x1 + dx)), min(h, int(y1 + dy)))


def merge(boxes, gap=0):
    """Boxes that overlap (or come within gap pixels) joined into one."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]
