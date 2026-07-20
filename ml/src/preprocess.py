"""Этап 2 — препроцессинг фото овоскопирования.

Единый вход для baseline (Этап 3) и CNN (Этап 4), плюс тот же пайплайн потом
применяется к перепелиным фото (Этап 5) и в API (Этап 8) — один код на всё.

Ключевые решения (обоснованы находками EDA (Этап 1)):
  * РЕСАЙЗ К ЕДИНОМУ КВАДРАТУ 224×224. Убирает признаки «размер/пропорции» как
    подсказку класса (на Этапе 1 они давали утечку: класс угадывался по метаданным
    съёмки с accuracy 0.65). 224 — стандартный вход ResNet18/EfficientNet-B0.
  * LETTERBOX (вписать с сохранением пропорций + добить чёрным), НЕ грубый resize.
    До половины фото dead/infertile имеют экстремальные пропорции; грубый resize
    сплющил бы яйцо и исказил форму/текстуру. Фон при просвечивании тёмный, поэтому
    чёрный паддинг сливается с фоном естественно и не вносит ложных краёв.
  * Нормализация яркости — НЕ в детерминированном препроцессинге, а как аугментация
    (см. ниже). Причина: при просвечивании яркость/опрозрачность — потенциально
    РЕАЛЬНЫЙ биологический сигнал, глушить его глобально рискованно.

Аугментации (применяются ТОЛЬКО на train, в Colab на Этапе 4; здесь — спека + превью).
Обоснование под овоскопирование, а не универсальный список:
  * hflip (горизонтальное отражение): ориентация яйца влево/вправо произвольна при
    съёмке → безопасно удваивает данные.
  * rotate ±15°: телефон в руке, угол съёмки гуляет → реалистичная вариативность.
  * brightness/contrast jitter (умеренный, ±20%): ДВОЙНАЯ польза — (1) устойчивость к
    разному освещению (важно для переноса на перепелов с другим светом), (2) ломает
    «яркость как ярлык класса» из Этапа 1, заставляя смотреть на содержимое.
  * mild zoom (±10%): яйцо занимает разную долю кадра → масштабная инвариантность.
НЕ используем и почему:
  * сильный hue/color jitter — цвет свечения сквозь скорлупу (краснота) несёт сигнал,
    на Этапе 7 мы его отдельно исследуем; агрессивный сдвиг оттенка стёр бы его.
  * shear/сильные геометрические искажения — форма яйца стабильна, искажать нереалистично.
  * vertical flip — спорно: воздушная камера у тупого конца задаёт «верх»; не включаем,
    чтобы не ломать этот естественный приор (rotate ±15° и так покрывает наклон).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "chicken"
REPORTS = ROOT / "ml" / "reports"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMG_SIZE = 224
FILL = (0, 0, 0)  # чёрный паддинг — сливается с тёмным фоном просвечивания

# Нормализация ImageNet — для CNN на Этапе 4 (transfer learning). Здесь как константы,
# чтобы один и тот же контракт использовался в Colab и в API.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_resize(im: Image.Image, size: int = IMG_SIZE, fill=FILL) -> Image.Image:
    """Вписать изображение в квадрат size×size с сохранением пропорций, добив чёрным."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_path(path: str | Path, size: int = IMG_SIZE) -> Image.Image:
    """Детерминированный препроцессинг одного файла (для baseline/инференса/API)."""
    with Image.open(path) as im:
        return letterbox_resize(im, size)


def to_normalized_array(im: Image.Image) -> np.ndarray:
    """RGB PIL -> float32 CHW, нормализация ImageNet (контракт входа CNN)."""
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr.transpose(2, 0, 1)


# --- аугментации (детерминированные версии для превью; в Colab — рандомизированные) ---
def aug_hflip(im: Image.Image) -> Image.Image:
    return ImageOps.mirror(im)


def aug_rotate(im: Image.Image, deg: float = 15) -> Image.Image:
    return im.rotate(deg, resample=Image.BILINEAR, fillcolor=FILL)


def aug_brightness(im: Image.Image, factor: float = 1.2) -> Image.Image:
    return ImageEnhance.Brightness(im).enhance(factor)


def aug_contrast(im: Image.Image, factor: float = 1.2) -> Image.Image:
    return ImageEnhance.Contrast(im).enhance(factor)


def aug_zoom(im: Image.Image, factor: float = 1.1) -> Image.Image:
    w, h = im.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def _sample_paths(per_class: int = 1) -> list[Path]:
    out: list[Path] = []
    for cls in ["fertile", "infertile", "dead"]:
        hits = [p for p in DATA.rglob("*") if p.suffix.lower() in IMG_EXT
                and cls in p.name.lower()
                and not (cls == "fertile" and "infertile" in p.name.lower())]
        out.extend(sorted(hits)[:per_class])
    return out


def demo_letterbox() -> None:
    """Превью до/после: оригинал (в исходных пропорциях) vs letterbox 224×224."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = _sample_paths(per_class=1)
    fig, axes = plt.subplots(len(paths), 2, figsize=(6, 3 * len(paths)))
    for r, p in enumerate(paths):
        with Image.open(p) as im:
            orig = ImageOps.exif_transpose(im).convert("RGB")
        proc = letterbox_resize(orig)
        axes[r, 0].imshow(orig); axes[r, 0].set_title(f"до: {orig.size[0]}×{orig.size[1]}", fontsize=9)
        axes[r, 1].imshow(proc); axes[r, 1].set_title("после: letterbox 224×224", fontsize=9)
        cls = next(c for c in ["infertile", "fertile", "dead"] if c in p.name.lower())
        axes[r, 0].set_ylabel(cls, fontsize=11)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Этап 2: препроцессинг до/после (сохраняем пропорции яйца)")
    fig.tight_layout(); fig.savefig(REPORTS / "preprocess_before_after.png", dpi=110); plt.close(fig)


def demo_augmentations() -> None:
    """Превью аугментаций на одном фото — что именно делает каждая."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = _sample_paths(per_class=1)[0]
    base = preprocess_path(p)
    variants = [
        ("оригинал (letterbox)", base),
        ("hflip", aug_hflip(base)),
        ("rotate +15°", aug_rotate(base, 15)),
        ("brightness ×1.2", aug_brightness(base, 1.2)),
        ("contrast ×1.2", aug_contrast(base, 1.2)),
        ("zoom ×1.1", aug_zoom(base, 1.1)),
    ]
    fig, axes = plt.subplots(1, len(variants), figsize=(2.2 * len(variants), 2.6))
    for ax, (name, img) in zip(axes, variants):
        ax.imshow(img); ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Этап 2: аугментации, обоснованные под овоскопирование (train-only)")
    fig.tight_layout(); fig.savefig(REPORTS / "preprocess_augmentations.png", dpi=110); plt.close(fig)


def main() -> int:
    if not DATA.exists() or not any(DATA.rglob("*")):
        print(f"Нет данных в {DATA}."); return 1
    REPORTS.mkdir(parents=True, exist_ok=True)
    demo_letterbox()
    demo_augmentations()
    print("Готово. Превью сохранены:")
    print("  ml/reports/preprocess_before_after.png")
    print("  ml/reports/preprocess_augmentations.png")
    print(f"Единый вход: {IMG_SIZE}×{IMG_SIZE}, letterbox, чёрный паддинг.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
