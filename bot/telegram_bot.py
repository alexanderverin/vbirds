"""Telegram-бот сбора перепелиных данных (телеграм-слой поверх bot/core.py).

Поток: новая партия -> овоскоп (фото -> подсказка класса -> подтверждение, счётчик) ->
отбраковка «минус» (номер + причина) -> сводка -> отметка невылупившихся после вывода.

Токен берётся из переменной окружения VERINBIRDS_BOT_TOKEN (получить у @BotFather).
Реализация — прямые вызовы Telegram Bot API через requests (без внешних фреймворков).

Запуск:
    export VERINBIRDS_BOT_TOKEN="123456:AAE..."
    python bot/telegram_bot.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def _load_dotenv():
    """Подхватить токен из .env в корне проекта (для локального запуска без export).
    В Docker его и так грузит docker-compose (env_file), тут — дублирующее удобство."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
TOKEN = os.environ.get("VERINBIRDS_BOT_TOKEN", "").strip()
API = f"https://api.telegram.org/bot{TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"

# фото, ожидающие подтверждения класса: chat_id -> {"bytes": ..., "num": int, "pred": str}
_pending: dict[int, dict] = {}


def api(method: str, **params):
    r = requests.post(f"{API}/{method}", json=params, timeout=65)
    return r.json()


def send(chat_id, text, keyboard=None):
    p = {"chat_id": chat_id, "text": text}
    if keyboard:
        p["reply_markup"] = {"inline_keyboard": keyboard}
    return api("sendMessage", **p)


def btn(text, data):
    return {"text": text, "callback_data": data}


def download_photo(file_id: str) -> bytes:
    info = api("getFile", file_id=file_id)
    path = info["result"]["file_path"]
    return requests.get(f"{FILE_API}/{path}", timeout=65).content


HELP = (
    "🐣 VerinBirds — сбор данных овоскопирования\n\n"
    "1) /newbatch <партия> [вид]  — начать партию (вид: quail/chicken)\n"
    "2) /day <день>  — задать день овоскопирования (настраиваемый)\n"
    "3) Пришли ФОТО яйца — бот предложит класс, подтверди кнопкой. Счётчик сам считает.\n"
    "4) /minus <номер>  — изъять яйцо (спросит причину: пусто/замерло)\n"
    "5) /summary  — сводка по партии\n"
    "6) /schedule  — даты овоскопа / раскатки / вылупления\n"
    "7) /nothatched <номера через пробел>  — после вывода отметить невылупившиеся\n\n"
    "Номер яйца = твой карандашный номер (сквозной по партии).\n"
    "Бот сам напомнит в дни овоскопирования и к вылуплению."
)


def reminder_loop():
    """Фоновый поток: раз в час проверяет наступившие напоминания и шлёт их."""
    while True:
        try:
            for chat_id, text in core.due_reminders(date.today().isoformat()):
                send(chat_id, text)
        except Exception as e:
            print("ошибка напоминаний:", e, file=sys.stderr)
        time.sleep(3600)


def _schedule_text(b) -> str:
    lines = ["📅 Расписание (по дате закладки):"]
    for key, when, _ in core.schedule(b):
        label = ("овоскоп день " + key.split("_")[1]) if key.startswith("candle_") \
            else "раскатка" if key == "rollout" else "вылупление 🐣"
        lines.append(f"  {when} — {label}")
    return "\n".join(lines)


def handle_message(msg):
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()

    if "photo" in msg:
        b = download_photo(msg["photo"][-1]["file_id"])  # [-1] = наибольший размер
        s = core.summary(chat_id)
        if s is None:
            send(chat_id, "Сначала начни партию: /newbatch <номер>")
            return
        ok, why = core.photo_ok(b)
        if not ok:
            send(chat_id, f"⚠️ Фото не принято: {why}.\nПришли снимок яйца НА ПРОСВЕТ "
                          f"(тёмное помещение, тёмный фон + светящееся яйцо).")
            return
        num = s["next_num"]
        pred, conf = core.predict(b)
        _pending[chat_id] = {"bytes": b, "num": num, "pred": pred}
        pred_txt = f"подсказка: {pred} ({conf:.0%})" if pred else "модель недоступна"
        row = [btn(("✅ " if c == pred else "") + c, f"cls:{c}:{num}") for c in core.CLASSES]
        send(chat_id, f"Яйцо №{num} · {pred_txt}\nВыбери класс:", [row])
        return

    if text.startswith("/start") or text.startswith("/help"):
        send(chat_id, HELP)
    elif text.startswith("/newbatch"):
        parts = text.split()
        if len(parts) < 2:
            send(chat_id, "Формат: /newbatch <номер партии> [quail|chicken]")
            return
        partiya = parts[1]
        species = parts[2] if len(parts) > 2 and parts[2] in core.SUGGESTED_DAYS else "quail"
        b = core.new_batch(chat_id, partiya, species=species)
        send(chat_id, f"🥚 Партия {partiya} ({species}) начата, дата закладки {b['set_date']}.\n"
                      + _schedule_text(b)
                      + "\nБуду напоминать в эти дни. Присылай фото яиц; /day <число> — сменить день.")
    elif text.startswith("/day"):
        parts = text.split()
        if len(parts) < 2 or not parts[1].isdigit():
            send(chat_id, "Формат: /day <число>")
            return
        b = core.set_day(chat_id, int(parts[1]))
        send(chat_id, f"День овоскопирования = {parts[1]}" if b else "Сначала /newbatch")
    elif text.startswith("/minus"):
        parts = text.split()
        if len(parts) < 2 or not parts[1].isdigit():
            send(chat_id, "Формат: /minus <номер яйца>")
            return
        num = int(parts[1])
        kb = [[btn("🟡 пусто", f"rsn:empty:{num}"), btn("⚫ замерло", f"rsn:dead:{num}")]]
        send(chat_id, f"Изъять яйцо №{num}. Причина?", kb)
    elif text.startswith("/summary"):
        s = core.summary(chat_id)
        if s is None:
            send(chat_id, "Нет активной партии. /newbatch <номер>")
            return
        rb = ", ".join(f"{k}={v}" for k, v in s["removed_by_reason"].items()) or "—"
        send(chat_id,
             f"📊 Партия {s['partiya']} ({s['species']})\n"
             f"День овоскопа: {s['day']} · дни: {s['candling_days']} · раскатка: {s['rollout_day']}\n"
             f"Заложено (снято): {s['total']}\n"
             f"Изъято: {s['removed']} ({rb})\n"
             f"В инкубации: {s['remaining']}\n"
             f"Следующий номер: {s['next_num']}")
    elif text.startswith("/schedule"):
        b = core._active_batch(core._load_state(), chat_id)
        send(chat_id, _schedule_text(b) if b else "Нет активной партии. /newbatch <номер>")
    elif text.startswith("/nothatched"):
        nums = [int(x) for x in text.split()[1:] if x.isdigit()]
        if not nums:
            send(chat_id, "Формат: /nothatched 4 7 12 ...")
            return
        r = core.mark_not_hatched(chat_id, nums)
        send(chat_id, f"Отмечено невылупившихся: {r['not_hatched']}. Остальные — вылупились.")
    elif text:
        send(chat_id, "Не понял. /help — список команд.")


def handle_callback(cq):
    chat_id = cq["message"]["chat"]["id"]
    data = cq["data"]
    api("answerCallbackQuery", callback_query_id=cq["id"])
    kind, val, num = data.split(":")
    num = int(num)

    if kind == "cls":
        pend = _pending.pop(chat_id, None)
        if not pend or pend["num"] != num:
            send(chat_id, "Фото устарело, пришли заново.")
            return
        r = core.save_egg(chat_id, pend["bytes"], val, num=num)
        send(chat_id, f"✅ №{r['num']} · {r['cls']} · снято {r['total']} · в инкубации {r['remaining']}")
    elif kind == "rsn":
        r = core.remove_egg(chat_id, num, val)
        send(chat_id, f"➖ №{r['num']} изъято ({val}) · изъято всего {r['removed']} · в инкубации {r['remaining']}")


def main():
    if not TOKEN:
        print("Нет токена. Сделай: export VERINBIRDS_BOT_TOKEN=\"<токен от @BotFather>\"", file=sys.stderr)
        return 1
    me = api("getMe")
    if not me.get("ok"):
        print("Токен неверный:", me, file=sys.stderr)
        return 1
    print(f"Бот @{me['result']['username']} запущен. Ctrl+C для остановки.")
    threading.Thread(target=reminder_loop, daemon=True).start()  # напоминания в фоне
    offset = None
    while True:
        try:
            upd = api("getUpdates", offset=offset, timeout=50)
            for u in upd.get("result", []):
                offset = u["update_id"] + 1
                if "message" in u:
                    handle_message(u["message"])
                elif "callback_query" in u:
                    handle_callback(u["callback_query"])
        except KeyboardInterrupt:
            print("\nостановлен"); return 0
        except Exception as e:
            print("ошибка цикла:", e, file=sys.stderr); time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main())
