"""Этап 1 — EDA куриного датасета овоскопирования.

Что делает:
  * рекурсивно находит изображения в data/chicken/;
  * выводит класс из имени родительской папки (fertile / infertile / dead);
  * считает распределение классов, размеры изображений, разброс яркости;
  * ищет аномалии: битые файлы, grayscale, экстремальные пропорции, точные дубликаты (по md5);
  * сохраняет графики и таблицы в ml/reports/ и печатает текстовое резюме.

Запуск:
    python ml/src/eda.py
Никакого обучения — только чтение и статистика (CPU).
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI, только сохранение в файлы
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "chicken"
REPORTS = ROOT / "ml" / "reports"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_KEYWORDS = {"fertile": "fertile", "infertile": "infertile", "dead": "dead"}


def infer_class(path: Path) -> str:
    """Класс = ключевое слово из пути/имени. infertile проверяем раньше fertile (подстрока).

    Датасет — экспорт Edge Impulse: класс закодирован в префиксе имени файла
    (`dead.dead-702...jpg`, `fertile.fertile1099...jpg`, `infertile.infertile-104...jpg`).
    """
    joined = "/".join(p.lower() for p in path.parts)
    if "infertile" in joined:
        return "infertile"
    if "fertile" in joined:
        return "fertile"
    if "dead" in joined:
        return "dead"
    return "unknown"


def infer_split(path: Path) -> str:
    """train/test-сплит задан папками training/ и testing/ в исходном экспорте."""
    parts = [p.lower() for p in path.parts]
    if "training" in parts:
        return "train"
    if "testing" in parts:
        return "test"
    return "unknown"


def scan() -> pd.DataFrame:
    rows = []
    hashes: dict[str, str] = {}
    files = [p for p in DATA.rglob("*") if p.suffix.lower() in IMG_EXT]
    for p in files:
        rec = {"path": str(p.relative_to(DATA)), "cls": infer_class(p), "split": infer_split(p),
               "ok": True, "w": None, "h": None, "mode": None, "mean_lum": None, "dup_of": None}
        try:
            raw = p.read_bytes()
            h = hashlib.md5(raw).hexdigest()
            if h in hashes:
                rec["dup_of"] = hashes[h]
            else:
                hashes[h] = rec["path"]
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                rec["w"], rec["h"], rec["mode"] = im.width, im.height, im.mode
                # средняя яркость по уменьшенной копии — дёшево и достаточно для EDA
                g = np.asarray(im.convert("L").resize((64, 64)))
                rec["mean_lum"] = float(g.mean())
        except (UnidentifiedImageError, OSError) as e:
            rec["ok"] = False
            rec["mode"] = f"ERROR:{type(e).__name__}"
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not DATA.exists() or not any(DATA.rglob("*")):
        print(f"Нет данных в {DATA}. Сначала: python ml/src/download_data.py")
        return 1

    df = scan()
    total = len(df)
    print(f"\n=== EDA: куриный датасет ({total} файлов) ===\n")

    # --- распределение классов
    dist = df["cls"].value_counts()
    print("Распределение классов:")
    for cls, n in dist.items():
        print(f"  {cls:10s}: {n:5d}  ({100*n/total:4.1f}%)")
    imbalance = dist.max() / max(dist.min(), 1)
    print(f"Дисбаланс (max/min): {imbalance:.2f}x")

    # --- готовый train/test сплит (из папок training/ и testing/)
    if df["split"].nunique() > 1:
        print("\nГотовый train/test сплит (класс × сплит):")
        print(pd.crosstab(df["cls"], df["split"]).to_string())

    # --- размеры
    valid = df[df["ok"]]
    print("\nРазмеры изображений (px):")
    print(f"  ширина:  min={valid.w.min():.0f}  med={valid.w.median():.0f}  max={valid.w.max():.0f}")
    print(f"  высота:  min={valid.h.min():.0f}  med={valid.h.median():.0f}  max={valid.h.max():.0f}")
    print(f"  уникальных (w,h): {valid.groupby(['w','h']).ngroups}")

    # --- аномалии
    broken = df[~df["ok"]]
    gray = valid[valid["mode"].isin(["L", "LA", "1"])]
    valid = valid.assign(ar=valid.w / valid.h)
    extreme_ar = valid[(valid.ar < 0.5) | (valid.ar > 2.0)]
    dups = df[df["dup_of"].notna()]
    unknown = df[df["cls"] == "unknown"]
    print("\nАномалии:")
    print(f"  битых/нечитаемых файлов : {len(broken)}")
    print(f"  grayscale изображений   : {len(gray)}")
    print(f"  экстремальные пропорции : {len(extreme_ar)} (aspect <0.5 или >2.0)")
    print(f"  точные дубликаты (md5)  : {len(dups)}")
    print(f"  без класса (unknown)    : {len(unknown)}")

    # --- график 1: распределение классов
    fig, ax = plt.subplots(figsize=(6, 4))
    dist.sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Распределение классов (chicken)")
    ax.set_ylabel("кол-во фото")
    for i, v in enumerate(dist.sort_index()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(REPORTS / "eda_class_distribution.png", dpi=110); plt.close(fig)

    # --- график 2: яркость по классам
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in sorted(valid["cls"].unique()):
        s = valid[valid["cls"] == cls]["mean_lum"]
        ax.hist(s, bins=30, alpha=0.5, label=f"{cls} (n={len(s)})")
    ax.set_title("Распределение средней яркости по классам")
    ax.set_xlabel("mean luminance (0-255)"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS / "eda_brightness_by_class.png", dpi=110); plt.close(fig)

    # --- сетка примеров по классам
    save_examples(df)

    # --- сырой отчёт
    df.to_csv(REPORTS / "eda_scan.csv", index=False)
    print(f"\nОтчёты сохранены в {REPORTS.relative_to(ROOT)}/:")
    print("  eda_class_distribution.png, eda_brightness_by_class.png, eda_examples.png, eda_scan.csv")
    return 0


def save_examples(df: pd.DataFrame, per_class: int = 4) -> None:
    classes = [c for c in ["fertile", "infertile", "dead"] if c in df["cls"].unique()]
    if not classes:
        return
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 2.2, len(classes) * 2.2))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(classes):
        sample = df[(df["cls"] == cls) & (df["ok"])].head(per_class)
        for c in range(per_class):
            ax = axes[r, c]; ax.axis("off")
            if c < len(sample):
                p = DATA / sample.iloc[c]["path"]
                try:
                    with Image.open(p) as im:
                        ax.imshow(ImageOps.exif_transpose(im))
                except Exception:
                    pass
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=11)
    fig.suptitle("Примеры по классам (chicken)")
    fig.tight_layout(); fig.savefig(REPORTS / "eda_examples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
