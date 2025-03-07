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
st.title("Прогнозирование уровня самоубийств (лог-преобразованный таргет)")
st.write("""
Это Streamlit-приложение загружает данные (по умолчанию из master.csv), 
выполняет базовый EDA и **обучает несколько регрессионных моделей** 
на логарифмированном таргете (suicides_100k_pop). 
Для итогового сравнения с реальными значениями 
используется обратное преобразование expm1.
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
    """
    Функция читает CSV, удаляет ненужные колонки,
    переименовывает, обрабатывает пропуски, 
    масштабирует признаки и возвращает чистый DataFrame.
    """
    data = pd.read_csv(csv_file)

    # Удаляем колонки, которые точно не нужны
    drop_cols = ["country-year", "HDI for year", "suicides_no"]
    for col in drop_cols:
        if col in data.columns:
            data.drop(columns=col, inplace=True, errors='ignore')

    # Переименуем, если есть
    rename_dict = {
        "gdp_for_year ($)": "gdp_for_year",
        "gdp_per_capita ($)": "gdp_per_capita",
        "suicides/100k pop": "suicides_100k_pop"
    }
    for old_name, new_name in rename_dict.items():
        if old_name in data.columns:
            data.rename(columns={old_name: new_name}, inplace=True)

    # Преобразуем gdp_for_year (если он строковый с запятыми)
    if "gdp_for_year" in data.columns and data["gdp_for_year"].dtype == object:
        data["gdp_for_year"] = (
            data["gdp_for_year"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float, errors="coerce")
        )

    # Преобразуем year -> float/int
    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce")

    # Маппим age, sex
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

    sex_map = {"male": 0, "female": 1}
    if "sex" in data.columns:
        data["sex"] = data["sex"].map(sex_map)

    # Получаем дамми для generation (если есть)
    if "generation" in data.columns:
        data = pd.get_dummies(data, columns=["generation"])

    # Удаляем дубликаты
    data.drop_duplicates(inplace=True)

    # Удаляем строки без suicides_100k_pop
    if "suicides_100k_pop" in data.columns:
        data.dropna(subset=["suicides_100k_pop"], inplace=True)

    # Удаляем (если ещё осталась) колонку country
    if "country" in data.columns:
        data.drop(columns=["country"], inplace=True, errors='ignore')

    # Масштабируем числовые признаки (кроме целевого)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if "suicides_100k_pop" in numeric_cols:
        numeric_cols.remove("suicides_100k_pop")

    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Удаляем inf/NaN
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

# Гистограмма suicides_100k_pop
if "suicides_100k_pop" in data.columns:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.histplot(data["suicides_100k_pop"], bins=30, kde=True, ax=ax1, color="orange")
    ax1.set_title("Распределение suicides_100k_pop (Обычная шкала)")
    st.pyplot(fig1)
else:
    st.warning("Нет колонки 'suicides_100k_pop' в данных.")

# Корреляционная матрица
df_numeric = data.select_dtypes(include=[np.number])
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(df_numeric.corr(), cmap='viridis', annot=True, fmt=".2f", ax=ax2)
ax2.set_title("Correlation Matrix")
st.pyplot(fig2)

# =========================================
# 6) Формируем X, y с ЛОГ-преобразованным target
# =========================================
st.write("## Подготовка данных (лог-преобразованный target)")

if "suicides_100k_pop" not in data.columns:
    st.error("В датасете нет 'suicides_100k_pop'. Приложение остановлено.")
    st.stop()

# Выделяем признаки (X) и целевую (y)
X = data.drop(columns=["suicides_100k_pop"], errors='ignore')
y = data["suicides_100k_pop"]

# Логарифмируем целевую переменную
# log1p(y) = ln(y+1)
# Модель будет обучаться на y_log
y_log = np.log1p(y)

# Сплитим
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=test_size, random_state=42
)

st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# =========================================
# 7) Обучение нескольких регрессоров (log target)
# =========================================
st.write("## Обучение моделей (LOG-трансформированный target)")

models_dict = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=max_depth_tree, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=n_estimators_rf, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=k_neighbors)
}

results = []
for name, model in models_dict.items():
    # Обучаем на лог-таргете
    model.fit(X_train, y_train_log)

    # Предсказание в лог-пространстве
    y_pred_log = model.predict(X_test)

    # Обратное преобразование к обычной шкале
    y_pred = np.expm1(y_pred_log)  # expm1(x)=e^x -1

    # Сравниваем с реальными y (которые тоже в обычной шкале)
    y_test = np.expm1(y_test_log)  # Чтобы метрики считались корректно

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
# 8) Детальный просмотр для одной из моделей
# =========================================
st.write("## Детальный просмотр результатов (лог-трансформированный target)")

model_name = st.selectbox(
    "Выберите модель",
    list(models_dict.keys())
)
selected_model = models_dict[model_name]

# Предсказания
y_pred_log = selected_model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Фактические значения (тоже 'разлогарифмируем')
y_test_real = np.expm1(y_test_log)

fig3, ax3 = plt.subplots(figsize=(6,6))
ax3.scatter(y_test_real, y_pred, alpha=0.5)
ax3.set_xlabel("Истинные (suicides_100k_pop)")
ax3.set_ylabel("Предсказанные (expm1)")
ax3.set_title(f"{model_name}: Real vs Predicted (LOG target)")

min_val = min(y_test_real.min(), y_pred.min())
max_val = max(y_test_real.max(), y_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--')

st.pyplot(fig3)

# Показываем примеры
compare_df = pd.DataFrame({
    "Real": y_test_real.iloc[:10].values,
    "Predicted": y_pred[:10]
})
st.write("Примеры (первые 10):")
st.dataframe(compare_df)
