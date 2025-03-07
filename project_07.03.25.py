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

# =========================================
# 1) Заголовок приложения
# =========================================
st.title("Прогнозирование уровня самоубийств")
st.write("""
Это Streamlit-приложение позволяет загрузить данные (по умолчанию из master.csv),
выполнить базовый EDA и обучить несколько регрессионных моделей (LinearRegression,
DecisionTree, RandomForest, KNN) для прогноза столбца suicides_100k_pop.
""")

# =========================================
# 2) Боковая панель с настройками
# =========================================
st.sidebar.header("Параметры приложения")

test_size = st.sidebar.slider(
    "Test size (доля тестовой выборки)",
    min_value=0.1, max_value=0.5, value=0.2, step=0.05
)
max_depth_tree = st.sidebar.slider(
    "Max depth (Decision Tree)",
    1, 20, 5, 1
)
n_estimators_rf = st.sidebar.slider(
    "Число деревьев (Random Forest)",
    10, 300, 100, 10
)
k_neighbors = st.sidebar.slider(
    "Число соседей (KNN)",
    1, 15, 5, 1
)

# =========================================
# 3) Функция загрузки + препроцессинга данных
# =========================================
@st.cache_data
def load_and_preprocess_data(csv_file: str):
    # 1. Загрузка данных
    data = pd.read_csv(csv_file)

    # 2. Удаляем колонки, которые точно не нужны
    drop_cols = ["country-year", "HDI for year", "suicides_no"]
    for col in drop_cols:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    # 3. Переименуем нужные колонки (если они есть)
    rename_dict = {
        "gdp_for_year ($)": "gdp_for_year",
        "gdp_per_capita ($)": "gdp_per_capita",
        "suicides/100k pop": "suicides_100k_pop"
    }
    for old_name, new_name in rename_dict.items():
        if old_name in data.columns:
            data.rename(columns={old_name: new_name}, inplace=True)

    # 4. Преобразуем строки в float (например, gdp_for_year содержит запятые)
    if "gdp_for_year" in data.columns and data["gdp_for_year"].dtype == object:
        data["gdp_for_year"] = (
            data["gdp_for_year"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float, errors="coerce")
        )

    # 5. Преобразуем "year" в число, если есть
    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce")

    # 6. Маппим возрастные интервалы и пол
    age_map = {
        "5-14 years": 0,
        "15-24 years": 1,
        "25-34 years": 2,
        "35-54 years": 3,
        "55-74 years": 4,
        "75+ years": 5
    }
    if "age" in data.columns:
        data["age"] = data["age"].map(age_map)

    sex_map = {"male":0, "female":1}
    if "sex" in data.columns:
        data["sex"] = data["sex"].map(sex_map)

    # 7. Преобразуем generation в дамми (если оно есть)
    if "generation" in data.columns:
        data = pd.get_dummies(data, columns=["generation"])

    # Удаляем дубликаты
    data.drop_duplicates(inplace=True)

    # Удаляем строки, где нет целевого столбца
    if "suicides_100k_pop" in data.columns:
        data.dropna(subset=["suicides_100k_pop"], inplace=True)

    # =========================================
    # Фильтруем только числовые колонки
    # =========================================
    # Допустим, мы не хотим country, etc.
    columns_to_ignore = ["country", "country-year"]  # дополните, если есть др.
    data = data.drop(columns=[c for c in columns_to_ignore if c in data.columns], errors="ignore")

    # =========================================
    # Масштабируем numeric (population, gdp_for_year, gdp_per_capita и т.п.)
    # =========================================
    # Найдём все числовые колонки
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Но целевую колонку масштабировать не нужно
    target_col = "suicides_100k_pop"
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Выполним стандартизацию
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # =========================================
    # Удаляем оставшиеся NaN / inf (если появились после преобразований)
    # =========================================
    data = data.replace([np.inf, -np.inf], np.nan)
    data.dropna(inplace=True)

    return data

# =========================================
# 4) Загружаем датасет
# =========================================
csv_file = "master.csv"
data = load_and_preprocess_data(csv_file)

st.write("## Пример данных")
st.write(data.head(10))
st.write("**Размер датасета**:", data.shape)

# =========================================
# 5) Краткий EDA
# =========================================
st.write("## Краткий EDA")

# Описание числовых признаков
st.write("Статистика по числовым признакам:")
st.dataframe(data.describe())

# График распределения suicides_100k_pop
if "suicides_100k_pop" in data.columns:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.histplot(data["suicides_100k_pop"], bins=30, kde=True, ax=ax1, color="orange")
    ax1.set_title("Распределение suicides_100k_pop")
    st.pyplot(fig1)
else:
    st.warning("Колонки 'suicides_100k_pop' нет в данных!")

# Корреляционная матрица
df_numeric = data.select_dtypes(include=[np.number])
corr_matrix = df_numeric.corr()

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(corr_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax2)
ax2.set_title("Correlation Matrix")
st.pyplot(fig2)

# =========================================
# 6) Формируем X, y
# =========================================
st.write("## Подготовка данных для модели")

target_col = "suicides_100k_pop"
if target_col not in data.columns:
    st.error("В датасете нет столбца suicides_100k_pop. Приложение остановлено.")
    st.stop()

# Выбрасываем также любую текстовую колонку (country и т.п.) если ещё осталось
ignore_cols = ["country"]  # можно дополнить
features = [c for c in data.columns if c != target_col and c not in ignore_cols]

X = data[features].select_dtypes(include=[np.number])  # берём только числа
y = data[target_col]

# На всякий случай заново проверим на NaN
valid_idx = X.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# =========================================
# 7) train_test_split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# =========================================
# 8) Обучение нескольких регрессоров
# =========================================
st.write("## Обучение моделей")

models_dict = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=max_depth_tree, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=n_estimators_rf, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=k_neighbors)
}

results = []
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2
    })

res_df = pd.DataFrame(results).sort_values("RMSE")
st.dataframe(res_df)

# =========================================
# 9) Детальный просмотр для одной из моделей
# =========================================
st.write("## Детальный просмотр результатов одной из моделей")

model_name = st.selectbox(
    "Выберите модель",
    list(models_dict.keys())
)
selected_model = models_dict[model_name]
y_pred_sel = selected_model.predict(X_test)

# Scatter: y_test vs y_pred
fig3, ax3 = plt.subplots(figsize=(6,6))
ax3.scatter(y_test, y_pred_sel, alpha=0.5)
ax3.set_xlabel("Истинные values (suicides_100k_pop)")
ax3.set_ylabel("Предсказанные values")
ax3.set_title(f"{model_name}: Real vs Predicted")

min_val = min(y_test.min(), y_pred_sel.min())
max_val = max(y_test.max(), y_pred_sel.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--')

st.pyplot(fig3)

# Показываем первые 10 примеров
compare_df = pd.DataFrame({
    "Real": y_test.iloc[:10].values,
    "Predicted": y_pred_sel[:10]
})
st.write("Примеры (первые 10):")
st.dataframe(compare_df)

st.write("---")
st.write("**Вывод:** теперь в коде учитываются возможные проблемы с типами данных, NaN, inf и т.д. Если ошибка сохраняется, проверьте, что в csv-файле действительно есть нужные столбцы и нет текстовых колонок, которые мешают.") 
