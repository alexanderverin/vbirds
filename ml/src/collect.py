"""Этап 9 — утилита сбора перепелиных фото овоскопирования с разметкой НА ЛЕТУ.

Зачем разметка в момент съёмки, а не потом: на факте вылупления фермер уже не помнит визуальную
картину каждого конкретного яйца на просвете. Если не подписать сразу (класс + партия + ID + дата),
связь «фото ↔ визуальная оценка ↔ реальный итог» теряется, и датасет становится бесполезен для
обучения. Поэтому метка ставится в тот же момент, что и снимок.

Что делает:
  * принимает фото (готовый файл --image ИЛИ кадр с камеры --capture),
  * спрашивает/принимает класс + номер партии + ID яйца + день инкубации,
  * сохраняет фото с именем `partiya{N}_yaico{M}_{YYYY-MM-DD}_{class}.jpg` в data/quail_real/,
  * дописывает строку в CSV-лог data/quail_real/labels.csv (поле final_result — пустое, заполняется
    позже, после вылупления, командой `result`).

Примеры:
  # записать снимок (интерактивно спросит недостающее):
  python ml/src/collect.py add --image ~/Desktop/shot.jpg
  # то же полностью из аргументов (без вопросов):
  python ml/src/collect.py add --image shot.jpg --class fertile --partiya 3 --yaico 12 --day 10
  # снять кадр с подключённой камеры:
  python ml/src/collect.py add --capture --class dead --partiya 3 --yaico 5
  # позже, после вылупления, проставить итог:
  python ml/src/collect.py result --partiya 3 --yaico 12 --result hatched
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "quail_real"
CSV_PATH = DEST / "labels.csv"
CLASSES = ["fertile", "infertile", "dead"]
# Дни овоскопирования — НАСТРАИВАЕМЫ (не константа): у перепела/курицы и у разных практик
# они разные. Ниже — лишь типичные подсказки; фактический день принимается любой.
SUGGESTED_DAYS = {"quail": [7, 10, 14], "chicken": [7, 14, 18]}
FIELDS = ["filename", "partiya", "yaico", "date", "candling_day",
          "class_at_candling", "final_result", "notes"]


def ask(prompt: str, options: list | None = None, default=None):
    """Интерактивный ввод с валидацией по списку опций."""
    suffix = f" {options}" if options else ""
    if default is not None:
        suffix += f" [{default}]"
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if not val and default is not None:
            return default
        if options and val not in [str(o) for o in options]:
            print(f"  нужно одно из {options}")
            continue
        if val:
            return val


def ensure_csv() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def capture_frame(out_path: Path) -> bool:
    """Снять один кадр с камеры (opencv). Возвращает True при успехе."""
    try:
        import cv2
    except ImportError:
        print("Для --capture нужен opencv (pip install opencv-python-headless). "
              "Либо используй --image с готовым фото.", file=sys.stderr)
        return False
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Камера не найдена. Используй --image с файлом фото.", file=sys.stderr)
        return False
    for _ in range(5):  # прогреть экспозицию
        ok, frame = cam.read()
    cam.release()
    if not ok:
        print("Не удалось снять кадр.", file=sys.stderr)
        return False
    cv2.imwrite(str(out_path), frame)
    return True


def cmd_add(args) -> int:
    ensure_csv()
    # источник фото
    if not args.image and not args.capture:
        args.image = ask("Путь к фото (--image) или пусто для отмены")
        if not args.image:
            return 1
    if args.image and not Path(args.image).expanduser().exists():
        print(f"Файл не найден: {args.image}", file=sys.stderr)
        return 1

    # метки: из аргументов или спросить
    cls = args.cls or ask("Класс на просвете", CLASSES)
    partiya = args.partiya or ask("Номер партии")
    yaico = args.yaico or ask("ID яйца в партии")
    hint = "/".join(str(d) for d in SUGGESTED_DAYS.get(args.species, SUGGESTED_DAYS["quail"]))
    day = args.day or ask(f"День инкубации (любое число, обычно {hint}, или пусто)", default="")
    dt = args.date or date.today().isoformat()
    notes = args.notes or ""

    fname = f"partiya{partiya}_yaico{yaico}_{dt}_{cls}.jpg"
    out_path = DEST / fname

    if args.capture:
        if not capture_frame(out_path):
            return 1
    else:
        shutil.copy(Path(args.image).expanduser(), out_path)

    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({
            "filename": fname, "partiya": partiya, "yaico": yaico, "date": dt,
            "candling_day": day, "class_at_candling": cls, "final_result": "", "notes": notes})
    print(f"OK: сохранено {out_path.relative_to(ROOT)} + строка в labels.csv")
    return 0


def cmd_result(args) -> int:
    """Проставить итог (вылупилось/нет) по яйцу после инкубации."""
    if not CSV_PATH.exists():
        print("Нет labels.csv — сначала собери фото командой add.", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    hit = 0
    for r in rows:
        if r["partiya"] == str(args.partiya) and r["yaico"] == str(args.yaico):
            r["final_result"] = args.result
            hit += 1
    if not hit:
        print(f"Не найдено записей для партии {args.partiya}, яйца {args.yaico}.", file=sys.stderr)
        return 1
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"OK: обновил итог '{args.result}' для {hit} записей (партия {args.partiya}, яйцо {args.yaico}).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Сбор перепелиных фото овоскопирования с разметкой на лету")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("add", help="записать фото + метку")
    a.add_argument("--image", help="путь к готовому фото")
    a.add_argument("--capture", action="store_true", help="снять кадр с камеры")
    a.add_argument("--class", dest="cls", choices=CLASSES, help="класс на просвете")
    a.add_argument("--partiya", help="номер партии")
    a.add_argument("--yaico", help="ID яйца (карандашный номер)")
    a.add_argument("--species", choices=list(SUGGESTED_DAYS), default="quail",
                   help="вид (для подсказки типичных дней овоскопирования)")
    a.add_argument("--day", help="день инкубации (любое число; овоскоп — настраиваемый)")
    a.add_argument("--date", help="дата YYYY-MM-DD (по умолч. сегодня)")
    a.add_argument("--notes", help="заметка")
    a.set_defaults(func=cmd_add)

    r = sub.add_parser("result", help="проставить итог после вылупления")
    r.add_argument("--partiya", required=True)
    r.add_argument("--yaico", required=True)
    r.add_argument("--result", required=True, choices=["hatched", "not_hatched"])
    r.set_defaults(func=cmd_result)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help(); return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
