"""
src/models/baseline.py  (Objective 2 -- the floor the sequence model must beat)

Isolated-character classifiers on UJIpenchars2:
  * each character -> fixed-length vector: 48 points resampled along the whole pen path
    (pen-up jumps included, flagged), scaled into a unit box keeping aspect ratio,
    plus the aspect ratio and stroke count
  * k-NN (k chosen on the validation writers) and a small MLP (scikit-learn)
  * evaluated on the 20 held-out test writers

    python -m src.models.baseline [--config configs/base.yaml]

Writes results/baseline.json.
"""
import argparse
import json
import time
from collections import Counter

import numpy as np

from src.utils import load_config, resolve, save_json, set_seed
from src.preprocessing.clean import to_ink, remove_duplicates

N_POINTS = 48


def char_vector(strokes, flip_y=False, n=N_POINTS):
    ink = remove_duplicates(to_ink(strokes, flip_y=flip_y), 0.0)
    P = np.concatenate(ink)
    pen = np.concatenate([np.r_[1.0 if k > 0 else 0.0, np.zeros(len(s) - 1)] for k, s in enumerate(ink)])
    x0, y0 = P.min(0)
    w, h = np.ptp(P[:, 0]), np.ptp(P[:, 1])
    size = max(w, h, 1e-9)
    Q = (P - [x0 + w / 2, y0 + h / 2]) / size                    # unit box, centred, aspect kept
    if len(Q) == 1:
        R = np.repeat(Q, n, 0); pu = np.zeros(n)
    else:
        seg = np.hypot(*np.diff(Q, axis=0).T)
        cum = np.r_[0, np.cumsum(seg)]
        t = np.linspace(0, cum[-1], n) if cum[-1] > 0 else np.zeros(n)
        R = np.stack([np.interp(t, cum, Q[:, 0]), np.interp(t, cum, Q[:, 1])], 1)
        idx = np.clip(np.searchsorted(cum, t, side="right"), 1, len(Q) - 1)
        pu = pen[idx]                                            # 1 while travelling in the air
    return np.concatenate([R.ravel(), pu, [np.log((w + 1e-9) / (h + 1e-9)) / 3, len(ink) / 5]]).astype(np.float32)


def featurize(samples, flip_y=False):
    X = np.stack([char_vector(s.strokes, flip_y) for s in samples])
    y = np.array([s.label for s in samples])
    return X, y


def top_confusions(y_true, y_pred, k=15):
    c = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    return [{"true": t, "pred": p, "count": n} for (t, p), n in c.most_common(k)]


def main(argv=None):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from src.preprocessing.parse_uji import parse_uji_file, make_splits

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    d = cfg["data"]
    flip = d.get("uji_flip_y", False)
    samples = parse_uji_file(resolve(d["uji_file"]))
    train, val, test = make_splits(samples, tuple(d["uji_val_writers"]))
    Xtr, ytr = featurize(train, flip)
    Xva, yva = featurize(val, flip)
    Xte, yte = featurize(test, flip)
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    results = {}
    best_k, best_acc = 1, -1
    for k in (1, 3, 5, 7):
        acc = KNeighborsClassifier(k, weights="distance").fit(Xtr, ytr).score(Xva, yva)
        print(f"  kNN k={k}: val acc {acc:.4f}")
        if acc > best_acc:
            best_k, best_acc = k, acc
    knn = KNeighborsClassifier(best_k, weights="distance").fit(np.r_[Xtr, Xva], np.r_[ytr, yva])
    t0 = time.time(); pk = knn.predict(Xte); ms = (time.time() - t0) / len(Xte) * 1000
    results["knn"] = {"k": best_k, "val_acc": best_acc, "test_acc": float((pk == yte).mean()),
                      "test_cer": float((pk != yte).mean()), "ms_per_char": ms,
                      "top_confusions": top_confusions(yte, pk)}

        # Newer scikit-learn calls np.isnan on predictions during early stopping, which fails
    # for string labels -> train the MLP on integer ids and map predictions back.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(np.r_[ytr, yva, yte])
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256), alpha=1e-4, max_iter=300,
                        early_stopping=True, random_state=cfg["seed"])
    mlp.fit(Xtr, le.transform(ytr))
    va = mlp.score(Xva, le.transform(yva))
    t0 = time.time(); pm_ids = mlp.predict(Xte); ms = (time.time() - t0) / len(Xte) * 1000
    pm = le.inverse_transform(pm_ids)
    results["mlp"] = {"val_acc": va, "test_acc": float((pm == yte).mean()),
                      "test_cer": float((pm != yte).mean()), "ms_per_char": ms,
                      "top_confusions": top_confusions(yte, pm)}

    for name, r in results.items():
        print(f"{name}: test accuracy {r['test_acc']:.4f}  (isolated-char CER {r['test_cer']:.4f})")
        print("   most common confusions:", ", ".join(f"{c['true']}->{c['pred']}({c['count']})"
                                                     for c in r["top_confusions"][:8]))
    out = resolve("results/baseline.json")
    save_json(results, out)
    print("wrote", out)
    return results


if __name__ == "__main__":
    main()
