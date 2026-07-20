"""Инференс-обёртка для API (Этап 8). Переиспользует извлечение признаков из /ml
(backend оборачивает отрефакторенный код из /ml, не дублирует его). Модель — лёгкий baseline (без torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))
from baseline import extract_features  # noqa: E402  (переиспользуем код из /ml)

MODEL_PATH = ROOT / "ml" / "models" / "baseline_svm.joblib"

# Честная пометка отдаётся В КАЖДОМ ответе API — ограничение не прячем (часть честного продукта).
DISCLAIMER = ("Модель обучена на ПУБЛИЧНОМ ДАТАСЕТЕ КУРИНЫХ яиц. Точность на перепелиных яйцах "
              "НЕ валидирована (пигментированная скорлупа хуже просвечивается; нет данных). "
              "Это MVP — используйте как подсказку, не как окончательный вердикт. См. README.")

_bundle = None

# Сообщение при отклонении неверного фото.
BAD_PHOTO_MSG = ("Похоже, это не фото яйца на просвете. Пришли снимок овоскопирования: "
                 "яйцо на просвете в тёмном помещении (тёмный фон + светящееся яйцо).")


def looks_like_candling(pil: Image.Image) -> tuple[bool, str]:
    """Лёгкая эвристика «это фото овоскопирования?». Пороги откалиброваны на реальном датасете
    (реальных яиц отвергается <1%). Ловит частые ошибки: селфи, скриншоты, документы, дневной свет,
    однотонные картинки. НЕ ловит любое текстурное не-яйцо — для этого нужен обученный детектор (roadmap)."""
    import numpy as np
    a = np.asarray(pil.convert("RGB").resize((96, 96))).astype(np.float32)
    R, B = a[..., 0], a[..., 2]
    lum = 0.299 * R + 0.587 * a[..., 1] + 0.114 * B
    mean_lum, std = float(lum.mean()), float(lum.std())
    warm = float(((R > 90) & (R - B > 20)).mean())  # доля тёплых ярких пикселей (свечение скорлупы)
    if std < 8:
        return False, "плоское/однотонное изображение"
    if mean_lum > 170:
        return False, "слишком светлое (овоскопирование — в темноте)"
    if mean_lum > 125 and warm < 0.05:
        return False, "светлое и без тёплого свечения"
    return True, "ok"


def load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Нет модели {MODEL_PATH}. Запусти: python ml/src/save_baseline.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_pil(pil: Image.Image) -> tuple[str, dict[str, float]]:
    b = load()
    model = b["model"]
    feats = extract_features(pil).reshape(1, -1)
    proba = model.predict_proba(feats)[0]
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    top = max(probs, key=probs.get)
    return top, probs
