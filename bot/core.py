"""Ядро Telegram-бота сбора данных (Этап 9+): логика партий, счётчик, запись фото+меток,
отбраковка. Отделено от телеграм-слоя, чтобы тестировать локально без токена.

Данные: фото -> data/quail_real/, метки -> data/quail_real/labels.csv (та же схема, что collect.py),
состояние партий/счётчиков -> bot/state.json.

Ключ яйца = партия + карандашный номер (сквозной по партии). Номера предлагаются по порядку.
Счётчик: total (сфотографировано) - removed (изъято) = в инкубации.
"""
from __future__ import annotations

import csv
import json
import io
import os
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "quail_real"
CSV_PATH = DEST / "labels.csv"
# Состояние — в data/, чтобы весь персистентный слой (фото + метки + состояние) жил в одном
# томе при деплое в Docker. Переопределяется переменной окружения при желании.
STATE_PATH = Path(os.environ.get("VERINBIRDS_STATE", ROOT / "data" / "bot_state.json"))
CLASSES = ["fertile", "infertile", "dead"]
# Типичные дни овоскопирования по видам — ПОДСКАЗКА (день настраиваемый, не константа).
SUGGESTED_DAYS = {"quail": [7, 10, 14], "chicken": [7, 14, 18]}
# Срок инкубации до вылупления (дней) по видам — для даты вылупа и напоминаний.
HATCH_DAY = {"quail": 17, "chicken": 21}
ROLLOUT_DAY = {"quail": 15, "chicken": 18}
REASON2CLASS = {"empty": "infertile", "dead": "dead"}  # причина изъятия -> класс
FIELDS = ["filename", "partiya", "yaico", "date", "candling_day",
          "class_at_candling", "final_result", "notes"]


# ---------- состояние ----------
def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(s: dict) -> None:
    STATE_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def _chat(s: dict, chat_id) -> dict:
    return s.setdefault(str(chat_id), {"active": None, "batches": {}})


def _active_batch(s: dict, chat_id) -> dict | None:
    c = _chat(s, chat_id)
    if not c["active"]:
        return None
    return c["batches"].get(c["active"])


# ---------- операции ----------
def new_batch(chat_id, partiya: str, species: str = "quail",
              candling_days: list[int] | None = None, rollout_day: int | None = None,
              set_date: str | None = None) -> dict:
    s = _load_state()
    c = _chat(s, chat_id)
    c["batches"][str(partiya)] = {
        "partiya": str(partiya), "species": species,
        "candling_days": candling_days or SUGGESTED_DAYS.get(species, [7, 10, 14]),
        "rollout_day": rollout_day or ROLLOUT_DAY.get(species, 15),
        "hatch_day": HATCH_DAY.get(species, 17),
        "set_date": set_date or date.today().isoformat(),  # дата закладки
        "day": None, "total": 0, "next_num": 1, "removed": {}, "reminded": [],
    }
    c["active"] = str(partiya)
    _save_state(s)
    return c["batches"][str(partiya)]


def schedule(b: dict) -> list[tuple[str, str, str]]:
    """Расписание партии: список (ключ_события, дата ISO, текст напоминания)."""
    base = date.fromisoformat(b["set_date"])
    ev = []
    for d in b["candling_days"]:
        ev.append((f"candle_{d}", (base + timedelta(days=d)).isoformat(),
                   f"🔦 Партия {b['partiya']}: день {d} — пора овоскопировать. "
                   f"Задай /day {d} и присылай фото яиц."))
    ev.append(("rollout", (base + timedelta(days=b["rollout_day"])).isoformat(),
               f"📦 Партия {b['partiya']}: день {b['rollout_day']} — раскатка на вывод. "
               f"Изъятые яйца отмечай /minus <номер>."))
    ev.append(("hatch", (base + timedelta(days=b["hatch_day"])).isoformat(),
               f"🐣 Партия {b['partiya']}: ожидается вылупление! После вывода отметь "
               f"невылупившиеся: /nothatched <номера>."))
    return ev


def due_reminders(today: str | None = None) -> list[tuple[int, str]]:
    """Найти напоминания, срок которых наступил и ещё не отправлены. Помечает отправленными."""
    today = today or date.today().isoformat()
    s = _load_state()
    out = []
    for chat_id, c in s.items():
        for b in c.get("batches", {}).values():
            if "set_date" not in b:
                continue
            for key, when, text in schedule(b):
                if when <= today and key not in b.get("reminded", []):
                    out.append((int(chat_id), text))
                    b.setdefault("reminded", []).append(key)
    if out:
        _save_state(s)
    return out


def set_day(chat_id, day: int) -> dict | None:
    s = _load_state()
    b = _active_batch(s, chat_id)
    if b is None:
        return None
    b["day"] = day
    _save_state(s)
    return b


def _ensure_csv() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def save_egg(chat_id, photo_bytes: bytes, cls: str, num: int | None = None) -> dict:
    """Сохранить фото + метку. num=None -> взять следующий по порядку."""
    s = _load_state()
    b = _active_batch(s, chat_id)
    if b is None:
        raise ValueError("Нет активной партии — сначала new_batch")
    if cls not in CLASSES:
        raise ValueError(f"класс {cls!r} не из {CLASSES}")
    num = num if num is not None else b["next_num"]
    dt = date.today().isoformat()
    fname = f"partiya{b['partiya']}_yaico{num}_{dt}_{cls}.jpg"
    (DEST / fname).write_bytes(photo_bytes)

    _ensure_csv()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow({
            "filename": fname, "partiya": b["partiya"], "yaico": num, "date": dt,
            "candling_day": b["day"] or "", "class_at_candling": cls,
            "final_result": "", "notes": ""})

    b["total"] += 1
    b["next_num"] = max(b["next_num"], num + 1)
    _save_state(s)
    return {"num": num, "cls": cls, "total": b["total"], "remaining": _remaining(b), "file": fname}


def remove_egg(chat_id, num: int, reason: str) -> dict:
    """Отбраковка (кнопка «минус»): reason in {empty, dead}. Обновляет метку + счётчик."""
    if reason not in REASON2CLASS:
        raise ValueError(f"причина {reason!r} не из {list(REASON2CLASS)}")
    s = _load_state()
    b = _active_batch(s, chat_id)
    if b is None:
        raise ValueError("Нет активной партии")
    b["removed"][str(num)] = reason
    _save_state(s)

    # обновить строку(и) яйца в CSV: итог = removed_<reason>, класс уточняем по причине
    if CSV_PATH.exists():
        rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
        for r in rows:
            if r["partiya"] == str(b["partiya"]) and r["yaico"] == str(num):
                r["final_result"] = f"removed_{reason}"
                r["class_at_candling"] = REASON2CLASS[reason]
                r["notes"] = (r["notes"] + "; изъято при овоскопе").strip("; ")
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return {"num": num, "reason": reason, "removed": len(b["removed"]), "remaining": _remaining(b)}


def mark_not_hatched(chat_id, nums: list[int]) -> dict:
    """После вывода: отметить невылупившиеся номера (реальный ground truth)."""
    s = _load_state()
    b = _active_batch(s, chat_id)
    if b is None:
        raise ValueError("Нет активной партии")
    nums_set = {str(n) for n in nums}
    updated = 0
    if CSV_PATH.exists():
        rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
        for r in rows:
            if r["partiya"] == str(b["partiya"]):
                if r["yaico"] in nums_set:
                    r["final_result"] = "not_hatched"; updated += 1
                elif not r["final_result"]:  # остальные, кто не изъят — вылупились
                    r["final_result"] = "hatched"
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    b["not_hatched"] = sorted(nums_set)
    _save_state(s)
    return {"not_hatched": len(nums_set), "updated": updated}


def _remaining(b: dict) -> int:
    return b["total"] - len(b["removed"])


def summary(chat_id) -> dict | None:
    s = _load_state()
    b = _active_batch(s, chat_id)
    if b is None:
        return None
    reasons = {}
    for rsn in b["removed"].values():
        reasons[rsn] = reasons.get(rsn, 0) + 1
    return {"partiya": b["partiya"], "species": b["species"], "day": b["day"],
            "total": b["total"], "removed": len(b["removed"]), "removed_by_reason": reasons,
            "remaining": _remaining(b), "next_num": b["next_num"],
            "candling_days": b["candling_days"], "rollout_day": b["rollout_day"]}


# ---------- проверка «это фото овоскопирования?» ----------
_check_fn = None


def photo_ok(photo_bytes: bytes) -> tuple[bool, str]:
    """(True,'ok') если фото похоже на овоскопирование, иначе (False, причина).
    Эвристика не требует обученной модели. Если недоступна — не блокируем."""
    global _check_fn
    if _check_fn is None:
        try:
            import sys
            sys.path.insert(0, str(ROOT / "backend"))
            from model import looks_like_candling  # noqa
            _check_fn = looks_like_candling
        except Exception:
            _check_fn = False
    if not _check_fn:
        return True, "ok"
    from PIL import Image
    return _check_fn(Image.open(io.BytesIO(photo_bytes)))


# ---------- предсказание (модель-ассист; опционально) ----------
_predict_fn = None


def predict(photo_bytes: bytes):
    """Вернуть (класс, confidence) или (None, None), если модель недоступна."""
    global _predict_fn
    if _predict_fn is None:
        try:
            import sys
            sys.path.insert(0, str(ROOT / "backend"))
            from model import predict_pil  # noqa
            _predict_fn = predict_pil
        except Exception:
            _predict_fn = False
    if not _predict_fn:
        return None, None
    from PIL import Image
    top, probs = _predict_fn(Image.open(io.BytesIO(photo_bytes)))
    return top, probs[top]
