# Лёгкий образ для Telegram-бота сбора данных и/или API /predict.
# Инференс baseline-модели: без torch/opencv (держим образ лёгким).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Зависимости отдельным слоем для кэша
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Только код, нужный для инференса/бота (без датасетов, ноутбуков, отчётов)
COPY bot/ ./bot/
COPY backend/ ./backend/
COPY ml/src/ ./ml/src/
COPY ml/models/ ./ml/models/

# Каталог для персистентных данных (монтируется томом)
RUN mkdir -p /app/data/quail_real

# По умолчанию — бот. Для API переопредели command в docker-compose.
CMD ["python", "bot/telegram_bot.py"]
