# Notice

The code in this extension is new and MIT-licensed (see LICENSE).

## NudeNet detector models

Detection uses the NudeNet v3 detector models (`320n.onnx`, `640m.onnx`) by
[notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet), licensed under
the GNU AGPL-3.0. The models are not included in this extension: they are
downloaded on first use from NudeNet's GitHub release (320n can also come from
NudeNet's Python package on PyPI) into `models/nudenet`. The detection code in
`lib_peekaboo/detector.py` is written for this extension and does not use
NudeNet's Python package.

## Twemoji stickers

The sticker images in `assets/stickers` are from
[Twemoji](https://github.com/twitter/twemoji), Copyright 2019 Twitter, Inc and
other contributors, licensed under
[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). They were rendered
from the SVG sources to 512 × 512 PNG; nothing else was changed.
