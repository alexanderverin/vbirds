"""Загрузка публичного куриного датасета овоскопирования с Kaggle.

Датасет: hlnkmb/chicken-egg-analysis-dataset (~4275 фото, 3 класса: fertile/infertile/dead).
Источник подтверждён: https://www.kaggle.com/datasets/hlnkmb/chicken-egg-analysis-dataset

Требуется Kaggle API token (~/.kaggle/kaggle.json или переменные KAGGLE_USERNAME/KAGGLE_KEY).
Локально ставим ТОЛЬКО данные — обучение всё равно в Colab.

Использование:
    python ml/src/download_data.py            # скачать в data/chicken/
    python ml/src/download_data.py --force    # перекачать, даже если папка не пуста
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_SLUG = "hlnkmb/chicken-egg-analysis-dataset"
DEST = Path(__file__).resolve().parents[2] / "data" / "chicken"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download chicken egg dataset from Kaggle")
    ap.add_argument("--force", action="store_true", help="перекачать поверх существующего")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    existing = list(DEST.rglob("*.jpg")) + list(DEST.rglob("*.png"))
    if existing and not args.force:
        print(f"В {DEST} уже есть {len(existing)} изображений. Пропускаю (--force для перекачки).")
        return 0

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        # kaggle падает при импорте, если нет kaggle.json — ловим и объясняем по-человечески.
        print("ОШИБКА: не найдены Kaggle-креды.", file=sys.stderr)
        print("Положи kaggle.json в ~/.kaggle/ (chmod 600) или задай KAGGLE_USERNAME/KAGGLE_KEY.", file=sys.stderr)
        print(f"Детали: {e}", file=sys.stderr)
        return 2

    api = KaggleApi()
    api.authenticate()
    print(f"Скачиваю {DATASET_SLUG} -> {DEST} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(DEST), unzip=True, quiet=False)

    imgs = list(DEST.rglob("*.jpg")) + list(DEST.rglob("*.png")) + list(DEST.rglob("*.jpeg"))
    print(f"Готово. Изображений на диске: {len(imgs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
