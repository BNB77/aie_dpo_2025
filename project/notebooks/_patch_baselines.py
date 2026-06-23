import nbformat as nbf

nb = nbf.read("02_baselines.ipynb", as_version=4)

cells_to_add = [
    nbf.v4.new_markdown_cell("## Интерпретируемость финальной модели: feature importance"),
    nbf.v4.new_code_cell(
        'import pandas as pd\n'
        'importance = pd.read_csv("../artifacts/feature_importance.csv", index_col=0)["importance"]\n'
        'importance.sort_values(ascending=False)'
    ),
    nbf.v4.new_code_cell(
        'from IPython.display import Image\n'
        'Image("figures/feature_importance.png")'
    ),
    nbf.v4.new_markdown_cell(
        "**Рисунок 2.2.** Доминирующий признак — `max_delay` (максимальный код задержки "
        "платежа за 6 месяцев, importance≈0.54), далее `mean_delay` (≈0.16). Это совпадает "
        "с корреляционным анализом в EDA, где агрегаты истории просрочек были сильнее всего "
        "связаны с целью — модель не выучила ничего противоречащего здравому смыслу "
        "(никакой утечки целевой переменной через косвенные признаки). Демографические "
        "признаки (`AGE`, `EDUCATION`, `MARRIAGE`) дают суммарно меньше 2% важности — "
        "подтверждает наблюдение из EDA, что они слабо связаны с риском в этом датасете. "
        "Этого достаточно, чтобы объяснить риск-менеджеру, какие факторы определяют решение "
        "по конкретному клиенту, даже без построения полноценного SHAP-объяснения для "
        "каждого отдельного предсказания (это осталось в направлениях развития, раздел 8 отчёта)."
    ),
]

idx = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "Выбор финальной модели" in c.source)
nb.cells[idx:idx] = cells_to_add

nbf.write(nb, "02_baselines.ipynb")
print("inserted at", idx)
