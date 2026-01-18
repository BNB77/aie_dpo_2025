# HW06: Деревья решений и ансамбли

## Описание

Домашнее задание по теме "Деревья решений и ансамбли (bagging / random forest / boosting / stacking) и честный ML-эксперимент".

Датасет: **S06-hw-dataset-04.csv** - бинарная классификация с сильным дисбалансом классов (fraud-like задача).

## Структура

```
HW06/
├── HW06.ipynb              # Основной ноутбук с анализом
├── report.md               # Отчет по результатам
├── S06-hw-dataset-04.csv   # Датасет
└── artifacts/              # Папка для артефактов
    ├── figures/            # Графики и визуализации
    ├── metrics_test.json   # Финальные метрики на test
    ├── search_summaries.json   # Результаты подбора параметров
    ├── best_model.joblib   # Сохраненная лучшая модель
    └── best_model_meta.json    # Метаданные лучшей модели
```

## Запуск

1. Установите необходимые библиотеки:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

2. Откройте ноутбук `HW06.ipynb` в Jupyter

3. Запустите все ячейки последовательно (Run All Cells)

4. После выполнения ноутбука:
   - Все графики сохранятся в `artifacts/figures/`
   - Метрики и артефакты будут в `artifacts/`
   - Результаты описаны в `report.md`

## Реализованные модели

### Baseline:
- DummyClassifier (most_frequent)
- LogisticRegression (с подбором C)

### Модели недели 6:
- DecisionTreeClassifier (с контролем сложности через max_depth и min_samples_leaf)
- RandomForestClassifier (bagging + случайность по признакам)
- GradientBoostingClassifier (boosting)
- StackingClassifier (композиция моделей)

## Метрики

Для оценки используются:
- **Accuracy** - базовая метрика
- **F1-score** - для учета дисбаланса классов
- **ROC-AUC** - основная метрика для сравнения (наиболее подходит для дисбаланса)

## Особенности

- Датасет имеет сильный дисбаланс классов (~95-98% класс 0)
- Использован честный ML-протокол:
  - Фиксированный train/test split (random_state=42, stratify=y)
  - Подбор параметров только на train через CV
  - Финальная оценка только один раз на test
- Все модели используют `class_weight='balanced'` для учета дисбаланса
- Визуализация включает ROC-curves, PR-curves, confusion matrix, permutation importance

## Автор

Студент курса AIE DPO 2025
