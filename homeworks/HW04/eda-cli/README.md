# HW04 – eda_cli: мини-EDA для CSV + HTTP API

Небольшое CLI-приложение и HTTP-сервис для базового анализа CSV-файлов.
Используется в рамках Семинаров 03-04 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## HTTP API

Проект включает HTTP-сервис на базе FastAPI для оценки качества датасетов.

### Запуск сервера

```bash
.venv/bin/uvicorn eda_cli.api:app --reload --port 8000
```

После запуска API документация доступна по адресу: http://127.0.0.1:8000/docs

### Эндпоинты

#### GET /health

Проверка работоспособности сервиса.

**Пример запроса:**
```bash
curl http://127.0.0.1:8000/health
```

**Ответ:**
```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

#### POST /quality

Оценка качества датасета по агрегированным признакам.

**Параметры:**
- `n_rows` – количество строк
- `n_cols` – количество колонок
- `max_missing_share` – максимальная доля пропусков (0..1)
- `numeric_cols` – количество числовых колонок
- `categorical_cols` – количество категориальных колонок

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality \
  -H 'Content-Type: application/json' \
  -d '{"n_rows": 1000, "n_cols": 10, "max_missing_share": 0.1, "numeric_cols": 5, "categorical_cols": 5}'
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.9,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 0.023,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {"n_rows": 1000, "n_cols": 10}
}
```

#### POST /quality-from-csv

Оценка качества датасета из CSV-файла с использованием EDA-ядра.

**Параметры:**
- `file` – CSV-файл (multipart/form-data)

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality-from-csv \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.744,
  "message": "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам).",
  "latency_ms": 15.9,
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_many_zero_values": false
  },
  "dataset_shape": {"n_rows": 36, "n_cols": 14}
}
```

#### POST /quality-flags-from-csv

Получение полного набора флагов качества из CSV-файла (доработка из HW03).

**Параметры:**
- `file` – CSV-файл (multipart/form-data)

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/quality-flags-from-csv \
  -F "file=@data/example.csv"
```

**Ответ:**
```json
{
  "flags": {
    "too_few_rows": true,
    "too_many_columns": false,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_many_zero_values": false
  }
}
```

### Описание флагов качества

Флаги качества, реализованные в HW03:

- `too_few_rows` – в датасете менее 100 строк
- `too_many_columns` – в датасете более 100 колонок
- `too_many_missing` – максимальная доля пропусков превышает 50%
- `has_constant_columns` – есть колонки с единственным уникальным значением
- `has_high_cardinality_categoricals` – есть категориальные колонки с кардинальностью > 50
- `has_many_zero_values` – есть числовые колонки, где более 70% значений равны нулю
