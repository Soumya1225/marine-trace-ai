"""Train the look-alike model (B4) on the Yang & Singha 2025 dataset downloaded by download_lookalike_med.py.

    python scripts/train_lookalike_med.py

  * oil samples: every labelled oil slick (box from the .xml) in the ow / oc patches;
  * look-alike samples: the dark patches in the nw / nc patches (no oil there, experts checked);
  * test: 5-fold cross-validation grouped by satellite pass, so images of the same pass are never in both
    training and test;
  * compares the existing rule-based check, logistic regression and gradient boosting, keeps the best,
    saves models/lookalike_learned.pkl and models/metrics/lookalike_med.json (+ a figure).
"""
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.config import METRICS_DIR, MODELS_DIR  # noqa: E402
from ml.lookalike import learned  # noqa: E402
from ml.lookalike.second_stage import check  # noqa: E402

DIR = ROOT / "Datasets" / "lookalike_med"
TAB = DIR / "PANGAEA_980773.tab"
D = learned.DOWNSAMPLE
MAX_LOOKALIKES_PER_PATCH = 3


def pass_of_image(tab):
    """image name -> satellite pass (Sentinel product name), used to group the cross-validation."""
    lines = tab.read_text(encoding="utf-8").splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if l.strip() == "*/"), 0)
    head = lines[start].split("\t")
    col = next((i for i, h in enumerate(head) if "Sentinel_ID" in h), 7)
    out = {}
    for line in lines[start + 1:]:
        c = line.split("\t")
        if len(c) > col and c[1].strip():
            sid = c[col].strip()
            out[c[1].strip()] = f"{sid[:3]}_{sid[17:25]}" if len(sid) > 25 else c[1].strip()  # satellite + day
    return out


def load_img(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    im = im.resize((max(w // D, 1), max(h // D, 1)), Image.BILINEAR)
    return np.asarray(im, np.float32)


def boxes(xml_path):
    out = []
    for ob in ET.parse(xml_path).getroot().iter("object"):
        b = ob.find("bndbox")
        if b is not None:
            out.append([int(float(b.find(k).text)) // D for k in ("xmin", "ymin", "xmax", "ymax")])
    return out


def crop(img, mask, pad=learned.RING_PX + 4):
    rr, cc = np.nonzero(mask)
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, img.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def oil_samples(img, xml_path):
    """Largest dark area inside each labelled box (the box itself is not a mask)."""
    out = []
    for x0, y0, x1, y1 in boxes(xml_path):
        box = np.zeros(img.shape, bool)
        box[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        m = learned.dark_mask(img, k=0.7, min_px=6) & box
        lab, n = learned.ndimage.label(m)
        if n == 0:
            continue
        sizes = learned.ndimage.sum(m, lab, range(1, n + 1))
        out.append(lab == 1 + int(np.argmax(sizes)))
    return out


def lookalike_samples(img):
    m = learned.dark_mask(img, k=1.0, min_px=20)
    lab, n = learned.ndimage.label(m)
    if n == 0:
        return []
    sizes = learned.ndimage.sum(m, lab, range(1, n + 1))
    order = np.argsort(-sizes)[:MAX_LOOKALIKES_PER_PATCH]
    return [lab == 1 + int(i) for i in order]


def build(dir_=None):
    dir_ = dir_ or DIR
    passes = pass_of_image(TAB) if TAB.exists() else {}
    X, y, groups, rule = [], [], [], []
    for subset, label in (("ow", 1), ("oc", 1), ("nw", 0), ("nc", 0)):
        files = sorted((dir_ / subset).glob("*.jpg"))
        print(f"  {subset}: {len(files)} images", flush=True)
        for p in files:
            img = load_img(p)
            if label:
                xml = p.with_suffix(".xml")
                masks = oil_samples(img, xml) if xml.exists() else []
            else:
                masks = lookalike_samples(img)
            for m in masks:
                ci, cm = crop(img, m)
                f = learned.measure(ci, cm)
                if f is None:
                    continue
                X.append(learned.vector(f)[0])
                y.append(label)
                groups.append(passes.get(p.name, p.stem))
                rule.append(check(ci, cm, {"elongation": round(f["elongation"], 2)}, use_learned=False)["oil_score"])
    return np.array(X), np.array(y), np.array(groups), np.array(rule)


def evaluate(X, y, groups, rule):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {"logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                          max_leaf_nodes=31, random_state=0)}
    folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups))
    res = {"rule_based": {"auc": roc_auc_score(y, rule), "accuracy": accuracy_score(y, rule >= 0.5),
                          "f1_oil": f1_score(y, rule >= 0.5)}}
    probs = {}
    for name, make in models.items():
        p = np.zeros(len(y))
        for tr, te in folds:
            m = make().fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        probs[name] = p
        res[name] = {"auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5), "f1_oil": f1_score(y, p >= 0.5)}
    for r in res.values():
        for k in r:
            r[k] = round(float(r[k]), 3)
    best = max(models, key=lambda k: res[k]["auc"])
    return res, best, models[best]().fit(X, y), probs, folds


def importance(model, X, y):
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=0)
    return {k: round(float(v), 3) for k, v in sorted(zip(learned.FEATURES, imp.importances_mean), key=lambda t: -t[1])}


def figure(res, imp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(res)
    ax[0].bar([n.replace("_", "\n") for n in names], [res[n]["auc"] for n in names], color=["#999", "#5b8ff9", "#2a9d8f"])
    ax[0].set_ylim(0.5, 1)
    ax[0].set_title("Oil vs look-alike: ROC-AUC (5-fold, by satellite pass)")
    for i, n in enumerate(names):
        ax[0].text(i, res[n]["auc"] + 0.01, f"{res[n]['auc']:.2f}", ha="center")
    k = list(imp)[::-1]
    ax[1].barh(k, [imp[x] for x in k], color="#2a9d8f")
    ax[1].set_title("What the model uses (permutation importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    if not (DIR / "ow").exists():
        sys.exit(f"Dataset not found in {DIR}. Run: python scripts/download_lookalike_med.py")
    print("Measuring dark patches ...", flush=True)
    X, y, groups, rule = build()
    print(f"{len(y)} samples: {int(y.sum())} oil, {int((1 - y).sum())} look-alike, {len(set(groups))} satellite passes")
    res, best, model, _, _ = evaluate(X, y, groups, rule)
    imp = importance(model, X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(learned.MODEL_PATH, "wb") as fh:
        ranges = {k: [float(np.percentile(X[:, i], 1)), float(np.percentile(X[:, i], 99))]
                  for i, k in enumerate(learned.FEATURES)}
        pickle.dump({"model": model, "features": learned.FEATURES, "kind": best, "ranges": ranges,
                     "dataset": "Yang & Singha 2025, PANGAEA.980773 (CC-BY-4.0)"}, fh)
    out = {"dataset": "Yang & Singha 2025, Eastern Mediterranean 2019, PANGAEA.980773 (CC-BY-4.0)",
           "samples": {"oil": int(y.sum()), "lookalike": int((1 - y).sum()), "passes": len(set(groups))},
           "validation": "5-fold cross-validation grouped by satellite pass", "results": res, "chosen": best,
           "feature_importance": imp}
    (METRICS_DIR / "lookalike_med.json").write_text(json.dumps(out, indent=2))
    figure(res, imp, METRICS_DIR / "lookalike_med.png")
    print("\nResults (oil vs look-alike, unseen satellite passes):")
    for n, r in res.items():
        print(f"  {n:20s} AUC {r['auc']:.3f}  accuracy {r['accuracy']:.3f}  F1(oil) {r['f1_oil']:.3f}")
    print(f"Chosen: {best}. Saved {learned.MODEL_PATH.name} and metrics/lookalike_med.json + .png")


if __name__ == "__main__":
    main()
