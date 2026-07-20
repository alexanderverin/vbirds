"""Этап 3 — baseline: классические признаки (цвет + текстура) → SVM.

Точка отсчёта для CNN (Этап 4). Признаки считаем НА СОДЕРЖИМОМ, а не на артефактах кадра:
  * цветовые гистограммы R/G/B — НОРМИРОВАННЫЕ (density) → инвариантны к размеру фото и без
    чёрного letterbox-паддинга (иначе паддинг, коррелирующий с классом, дал бы утечку — см. Этап 2);
  * GLCM-текстура — «зернистость»/структура (видны ли вены/масса эмбриона).

Оценка: готовый train/test сплит датасета (Этап 1), фиксированный seed, StandardScaler + SVC(RBF),
лёгкий grid-search по C. Метрики: accuracy, confusion matrix, per-class recall/precision.
Плюс абляция (цвет-только / текстура-только / вместе) — честно показать, откуда сигнал.

Сверка: случайный уровень = 0.333; уровень «чистой утечки» по метаданным (Этап 1) = 0.648.
Запуск: python ml/src/baseline.py    (признаки кэшируются в ml/reports/baseline_features.npz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "chicken"
REPORTS = ROOT / "ml" / "reports"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CACHE = REPORTS / "baseline_features.npz"

FEAT_SIZE = 128          # рабочий размер для признаков (скорость)
HIST_BINS = 32           # бинов на цветовой канал → 96 цветовых признаков
GLCM_LEVELS = 32         # квантование серого для GLCM (256→32)
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
N_COLOR = 3 * HIST_BINS  # 96
N_TEX = len(GLCM_PROPS)  # 6


def infer_class(name: str) -> str:
    n = name.lower()
    if "infertile" in n:
        return "infertile"
    if "fertile" in n:
        return "fertile"
    if "dead" in n:
        return "dead"
    return "unknown"


def infer_split(path: Path) -> str:
    p = [x.lower() for x in path.parts]
    return "train" if "training" in p else "test" if "testing" in p else "unknown"


def extract_features(pil: Image.Image) -> np.ndarray:
    """[96 цветовых нормированных бинов] + [6 GLCM-текстурных] = 102 признака."""
    rgb = ImageOps.exif_transpose(pil).convert("RGB").resize((FEAT_SIZE, FEAT_SIZE))
    arr = np.asarray(rgb)
    color = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=HIST_BINS, range=(0, 255), density=True)
        color.append(h)
    color = np.concatenate(color).astype(np.float32)

    gray = np.asarray(rgb.convert("L"))
    gq = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    glcm = graycomatrix(gq, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    tex = np.array([graycoprops(glcm, p).mean() for p in GLCM_PROPS], dtype=np.float32)
    return np.concatenate([color, tex])


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"Признаки из кэша: {CACHE.name} ({d['X'].shape[0]} фото)")
        return d["X"], d["y"], d["split"]
    files = sorted(p for p in DATA.rglob("*") if p.suffix.lower() in IMG_EXT)
    X, y, sp = [], [], []
    for i, p in enumerate(files):
        try:
            with Image.open(p) as im:
                X.append(extract_features(im))
            y.append(infer_class(p.name)); sp.append(infer_split(p))
        except Exception as e:
            print(f"skip {p.name}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  признаки: {i + 1}/{len(files)}")
    X = np.vstack(X); y = np.array(y); sp = np.array(sp)
    REPORTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, split=sp)
    print(f"Признаки посчитаны и сохранены в {CACHE.name}: {X.shape}")
    return X, y, sp


def evaluate(X, y, split, feat_slice, name) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    tr, te = split == "train", split == "test"
    Xtr, Xte = X[tr][:, feat_slice], X[te][:, feat_slice]
    ytr, yte = y[tr], y[te]
    pipe = make_pipeline(StandardScaler(),
                         GridSearchCV(SVC(kernel="rbf"), {"C": [1, 10, 100]}, cv=3, n_jobs=-1))
    pipe.fit(Xtr, ytr)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(yte, pipe.predict(Xte))
    print(f"  {name:22s}: test accuracy = {acc:.3f}")
    return acc


def main() -> int:
    if not DATA.exists() or not any(DATA.rglob("*")):
        print(f"Нет данных в {DATA}."); return 1
    REPORTS.mkdir(parents=True, exist_ok=True)
    X, y, split = build_dataset()

    labels = ["fertile", "infertile", "dead"]
    print("\n=== Абляция признаков (сколько сигнала откуда) ===")
    print("  случайный уровень = 0.333 | уровень утечки по метаданным (Этап 1) = 0.648")
    evaluate(X, y, split, slice(0, N_COLOR), "цвет-только (96)")
    evaluate(X, y, split, slice(N_COLOR, N_COLOR + N_TEX), "текстура-только (6)")

    # --- основная модель: все признаки, полный отчёт
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    tr, te = split == "train", split == "test"
    grid = GridSearchCV(SVC(kernel="rbf"), {"C": [1, 10, 100], "gamma": ["scale"]}, cv=3, n_jobs=-1)
    model = make_pipeline(StandardScaler(), grid)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    acc = accuracy_score(y[te], pred)
    best_c = model.named_steps["gridsearchcv"].best_params_
    print(f"\n=== Основная модель (цвет+текстура, {X.shape[1]} признаков), best={best_c} ===")
    print(f"  test accuracy = {acc:.3f}")
    print("\nПолный отчёт (precision / recall / f1 по классам):")
    print(classification_report(y[te], pred, labels=labels, digits=3))

    cm = confusion_matrix(y[te], pred, labels=labels)
    plot_cm(cm, labels, acc)
    print("Confusion matrix сохранена: ml/reports/baseline_confusion_matrix.png")
    return 0


def plot_cm(cm, labels, acc) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("предсказано"); ax.set_ylabel("истина")
    ax.set_title(f"Baseline SVM (цвет+текстура)\ntest accuracy = {acc:.3f}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(REPORTS / "baseline_confusion_matrix.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
