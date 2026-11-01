"""Compare ship-detection settings on both Huntington Beach images.

A good setting finds many radar targets AND most of them match a real AIS ship (high match rate).
Too strict: almost no targets. Too loose: many targets but a low match rate (sea clutter counted as ships).

    python scripts/tune_ship_detection.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.ship_detection import detect_ships, match_dark_vessels  # noqa: E402
from ml.preprocessing.ais_clean import load_ais  # noqa: E402
from ml.preprocessing.land_mask import land_mask  # noqa: E402
from ml.preprocessing.sar_preprocess import load_scene  # noqa: E402

BBOX = (-118.25, 33.45, -117.85, 33.75)
SCENES = sorted((ROOT / "Datasets" / "raw" / "sentinel1").glob("S1*_20211003T*.zip"))
AIS = ROOT / "Datasets" / "raw" / "ais" / "AIS_huntington_2021-10-02_03.csv"
SETTINGS = [
    ("old: global, core z 10", dict(method="global")),
    ("local, core 5 dB", dict(method="local", local_min_db=3, local_core_db=5)),
    ("local, core 6 dB", dict(method="local", local_min_db=3, local_core_db=6)),
    ("local, core 8 dB", dict(method="local", local_min_db=4, local_core_db=8)),
    ("local, core 10 dB", dict(method="local", local_min_db=5, local_core_db=10)),
]

for path in SCENES:
    s = load_scene(path, bbox=BBOX)
    t = s["acquisition_time"]
    land = land_mask(s["image_u8"].shape, s["transform"])
    ais = load_ais(str(AIS), t_start=t - pd.Timedelta(hours=1), t_end=t + pd.Timedelta(hours=1))
    print(f"\n{path.name[17:32]} UTC")
    for name, kw in SETTINGS:
        m = match_dark_vessels(detect_ships(s["db"], land, s["transform"], **kw), ais, t)
        n, ok = len(m), sum(not x["dark"] for x in m)
        print(f"  {name:24s} targets {n:4d} | matched to AIS {ok:4d} | no AIS {n - ok:4d} | "
              f"match rate {ok / n * 100 if n else 0:5.1f}%")
