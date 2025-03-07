import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# ==========================
# 1. Заголовок приложения
# ==========================
st.title("📈 Прогнозирование уровня самоубийств")
st.write(
    """
    Данный демо-проект демонстрирует процесс предобработки данных, 
    обучения нескольких регрессионных моделей и визуализации результатов для задачи 
    "Прогнозирование уровня самоубийств (на 100k) на основе исторических и социально-экономических показателей".
    """
)

# ==========================
# 2. Боковая панель (параметры)
# ==========================
st.sidebar.header("Настройки и параметры")

test_size = st.sidebar.slider("Доля тестовой выборки (test_size)", 0.1, 0.5, 0.2, 0.05)
max_depth = st.sidebar.slider("Максимальная глубина дерева (Decision Tree)", 1, 20, 5, 1)
n_estimators = st.sidebar.slider("Количество деревьев (Random Forest)", 10, 300, 100, 10)
k_neighbors = st.sidebar.slider("Количество соседей (KNN)", 1, 15, 5, 1)

# ==========================
# 3. Функция загрузки и предобработки
# ==========================
@st.cache_data
def load_and_preprocess_data(csv_file: str):
    # Читаем CSV
    data = pd.read_csv(csv_file)
    # Удаляем лишние столбцы, чистим NaN и т.д. – согласно вашему пайплайну
    # Например:
    data.drop(columns=['country-year', 'HDI for year'], inplace=True, errors='ignore')

    # Переименуем некоторые колонки, если нужно
    data.rename(columns={
        'suicides/100k pop': 'suicides_100k_pop',
        'gdp_for_year ($)': 'gdp_for_year',
        'gdp_per_capita ($)': 'gdp_per_capita'
    }, inplace=True, errors='ignore')

    # Преобразуем 'gdp_for_year' (если есть запятые)
    if 'gdp_for_year' in data.columns and data['gdp_for_year'].dtype == 'object':
        data['gdp_for_year'] = data['gdp_for_year'].str.replace(",", "", regex=False).astype(float, errors='ignore')

    # Преобразуем категориальные столбцы (sex, age, generation) в числа
    age_map = {
        '5-14 years': 0, '15-24 years': 1, '25-34 years': 2,
        '35-54 years': 3, '55-74 years': 4, '75+ years': 5
    }
    if 'age' in data.columns:
        data['age'] = data['age'].map(age_map).fillna(-1)

    sex_map = {'male': 0, 'female': 1}
    if 'sex' in data.columns:
        data['sex'] = data['sex'].map(sex_map).fillna(-1)

    if 'generation' in data.columns:
        data = pd.get_dummies(data, columns=['generation'])

    # Удалим дубликаты, если есть
    data.drop_duplicates(inplace=True)

    # Удалим пропуски в целевом столбце, если таковые вдруг есть
    data.dropna(subset=['suicides_100k_pop'], inplace=True)

    # Масштабируем числовые признаки (по желанию).
    # Выберем некоторые столбцы для скейлинга:
    # (Проверяем, что они действительно есть в данных)
    numeric_cols = ['population', 'gdp_for_year', 'gdp_per_capita']
    for col in numeric_cols:
        if col not in data.columns:
            numeric_cols.remove(col)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data

# Загрузка данных (предполагаем, что master.csv лежит в папке с приложением).
# При желании можно заменить на st.file_uploader(...)
csv_file = "master.csv"
data = load_and_preprocess_data(csv_file)

# ==========================
# 4. Отображение данных
# ==========================
st.subheader("Обзор подготовленных данных")
st.write(f"Размер набора данных: {data.shape}")
st.dataframe(data.head(10))

# ==========================
# 5. Определяем фичи и таргет
# ==========================
# Предположим, что suicides_100k_pop – это таргет
target_col = 'suicides_100k_pop'
feature_cols = [c for c in data.columns if c not in ['suicides_100k_pop','suicides_no','country']]

X = data[feature_cols]
y = data[target_col]

# ==========================
# 6. Разделяем на train/test
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42
)

# ==========================
# 7. Обучение моделей
# ==========================
st.subheader("Обучение и оценка качества моделей")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=max_depth, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=n_estimators, max_depth=None, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=k_neighbors)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2
    })

# Выводим табличку с результатами
results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True)
st.write(results_df)

# ==========================
# 8. Визуализации
# ==========================

st.subheader("Сравнение фактических и предсказанных значений (примеры)")

selected_model = st.selectbox(
    "Выберите модель для просмотра детальной визуализации:",
    list(models.keys())
)

model_obj = models[selected_model]
y_pred_sel = model_obj.predict(X_test)

# График: scatter "y_test vs y_pred"
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(y_test, y_pred_sel, alpha=0.6)
ax1.set_xlabel("Фактический suicides_100k_pop")
ax1.set_ylabel("Предсказанный suicides_100k_pop")
ax1.set_title(f"{selected_model}: Реальные vs Предсказанные")
# Линия y=x для наглядности
min_val = min(y_test.min(), y_pred_sel.min())
max_val = max(y_test.max(), y_pred_sel.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
st.pyplot(fig1)

st.write("**Примеры (первые 10 точек из теста):**")
comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred_sel[:10]
})
st.dataframe(comparison_df)

# ==========================
# 9. (Опционально) Многомерная визуализация
# ==========================
# Если захотим 2D scatter по 2 признакам, а цвет = suicides_100k_pop
st.subheader("2D Scatter по выбранным признакам (цвет = целевая переменная)")

numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
default_2d = numeric_features[:2] if len(numeric_features) >= 2 else None

two_feats = st.multiselect("Выберите 2 числовых признака", numeric_features, default=default_2d)
if len(two_feats) == 2:
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    scatter = ax2.scatter(
        data[two_feats[0]],
        data[two_feats[1]],
        c=data[target_col],
        cmap='viridis',
        alpha=0.5
    )
    ax2.set_xlabel(two_feats[0])
    ax2.set_ylabel(two_feats[1])
    ax2.set_title("Распределение точек (цвет = suicides_100k_pop)")
    plt.colorbar(scatter, label="Уровень самоубийств (на 100k)")
    st.pyplot(fig2)
else:
    st.write("Выберите ровно 2 признака, чтобы увидеть 2D scatter.")

# ==========================
# 10. Завершающая часть
# ==========================
st.write("""
---
**Вывод**: в результате мы видим, какая модель лучше предсказывает 
числовой показатель (suicides_100k_pop) на тестовой выборке, 
с учётом выбранных гиперпараметров и доли разделения.
""")
