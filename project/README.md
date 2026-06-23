# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

Сервис кредитного риск-скоринга: оценка вероятности дефолта клиента по кредитной карте на основе открытого датасета UCI.

---

## 1. Паспорт проекта

- **Название проекта:** Сервис кредитного риск-скоринга
- **Автор:** Лебедев Антон Владимирович
- **Группа:** ИКБО-11-24
- **Контакт:** @bnbqwerty1

**Краткое описание:** Проект решает задачу оценки риска дефолта клиента по кредитной карте на следующий месяц, на основе открытого датасета UCI "Default of Credit Card Clients" (30000 клиентов). Используются классические ML-модели (логистическая регрессия как baseline, RandomForest и GradientBoosting с подбором гиперпараметров через GridSearchCV) с калибровкой вероятностей. Финальная модель — GradientBoosting. Результат — REST API на FastAPI, который по данным клиента возвращает вероятность дефолта и категорию риска.

---

## 2. Структура проекта

- `requirements.txt` — зависимости.
- `report.md` — отчёт по проекту (постановка задачи, данные, эксперименты, результаты).
- `self-checklist.md` — чек-лист самопроверки.
- `notebooks/` — `01_eda.ipynb` (разведочный анализ, очистка аномалий), `02_baselines.ipynb` (тюнинг моделей, выбор финальной).
- `src/data/` — `prepare_data.py` (очистка реального датасета и feature engineering), `preprocessing.py` (загрузка).
- `src/models/` — обучение, GridSearchCV-тюнинг, калибровка, сохранение артефактов.
- `src/service/` — FastAPI-сервис (`/health`, `/predict`).
- `data/raw/` — исходный датасет UCI (`uci_credit_card_default_raw.csv`).
- `data/` — `credit_data_full.csv` (30000 строк, после очистки и feature engineering), `credit_data_sample.csv` (300 строк, демо).
- `configs/` — `config.yaml`, `.env.example` (в корне `project/`).
- `tests/` — unit-тесты для подготовки данных и сервиса.
- `artifacts/` — `model.joblib`, `model_metadata.json`, `model_comparison.csv`, `feature_importance.csv`.

---

## 3. Требования и установка

### 3.1. Требования

- Python >= 3.10.

### 3.2. Установка окружения

```bash
cd project
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Подготовка данных и обучение модели

```bash
cd project
python -m src.data.prepare_data   # очищает data/raw/*.csv, строит признаки, сохраняет data/credit_data_full.csv
python -m src.models.train        # baseline + GridSearchCV-тюнинг, калибровка, сохраняет artifacts/
```

Обучение с тюнингом гиперпараметров занимает около 2 минут на CPU.

### 4.2. Запуск сервиса

```bash
cd project
uvicorn src.service.main:app --port 8000
```

Или через Docker:

```bash
docker build -t credit-scoring-service .
docker run -p 8000:8000 credit-scoring-service
```

Сервис поднимается на порту **8000**. Эндпоинты:

- `GET /health` — проверка работоспособности.
- `POST /predict` — оценка риска дефолта по профилю клиента.
- `GET /docs` — Swagger UI с интерактивной документацией.

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":200000,"EDUCATION":2,"MARRIAGE":1,"AGE":35,"max_delay":0,"mean_delay":-0.5,"n_months_delayed":0,"avg_bill_amt":45000,"avg_pay_amt":6000,"payment_to_bill_ratio":0.8,"credit_utilization":0.225,"bill_trend":1200.0}'
```

Ответ:

```json
{"prediction":0,"proba_default":0.0374,"risk_category":"low","model_version":"gradient_boosting_tuned"}
```

---

## 5. Данные

Используется открытый датасет **UCI "Default of Credit Card Clients"** (Yeh, I-C. & Lien, C.-H., 2009): 30000 анонимизированных клиентов тайваньского банка, источник — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. Сырой файл лежит в `data/raw/uci_credit_card_default_raw.csv`. Обработанная версия с очисткой аномальных категорий и построенными признаками — `data/credit_data_full.csv` (30000 строк), плюс демо-выборка `data/credit_data_sample.csv` (300 строк). Полный пайплайн обработки воспроизводится командой `python -m src.data.prepare_data`.

---

## 6. Тесты

```bash
cd project
pytest tests -v
```

Тесты покрывают: очистку аномальных категорий и корректность feature engineering (`tests/test_data.py`), а также работу сервиса — `/health`, валидный `/predict`, отклонение некорректного возраста и отсутствующего поля с кодом `422`, и монотонность вероятности риска относительно профиля клиента (`tests/test_service.py`).

---

## 7. Демонстрация на защите

1. Структура репозитория: `notebooks/`, `src/`, `artifacts/`.
2. Запуск сервиса (`uvicorn src.service.main:app --port 8000`), пара запросов через Swagger UI (`/docs`) — профиль с низким и высоким риском.
3. Ноутбук `01_eda.ipynb`: аномалии в сырых данных (недокументированные коды `EDUCATION`/`MARRIAGE`) и их очистка.
4. Ноутбук `02_baselines.ipynb`: сравнение baseline и тюнингованных моделей, ROC-кривая, обоснование выбора финальной модели.
5. Логи в консоли во время демонстрации запросов (request_id, latency, вероятность, решение).

---

## 8. Ограничения и дальнейшая работа

Текущие ограничения и направления развития — в `report.md`, раздел 8.

---
