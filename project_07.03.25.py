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
Это Streamlit‑приложение загружает данные (master.csv), 
выполняет базовый EDA и обучает несколько регрессионных моделей 
на логарифмированном таргете (suicides_100k_pop). 
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
            data.drop(columns=col, inplace=True, errors='ignore')

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
    # Фильтруем только числовые колонки (убираем country и т.п.)
    # =========================================
    columns_to_ignore = ["country", "country-year"]
    data = data.drop(columns=[c for c in columns_to_ignore if c in data.columns], errors="ignore")

    # =========================================
    # Масштабируем numeric (population, gdp_for_year, gdp_per_capita и т.п.)
    # =========================================
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    target_col = "suicides_100k_pop"
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # =========================================
    # Удаляем оставшиеся NaN / inf
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

st.write("Статистика по числовым признакам:")
st.dataframe(data.describe())

if "suicides_100k_pop" in data.columns:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.histplot(data["suicides_100k_pop"], bins=30, kde=True, ax=ax1, color="orange")
    ax1.set_title("Распределение suicides_100k_pop (Обычная шкала)")
    st.pyplot(fig1)
else:
    st.warning("Нет колонки 'suicides_100k_pop' в данных!")

df_numeric = data.select_dtypes(include=[np.number])
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(df_numeric.corr(), cmap='viridis', annot=True, fmt=".2f", ax=ax2)
ax2.set_title("Correlation Matrix")
st.pyplot(fig2)

# =========================================
# 5a) Лог-трансформация признаков
# =========================================
# Определяем столбцы, для которых хотим применить log1p (например, gdp_for_year, gdp_per_capita, population, suicides_100k_pop)
cols_to_check = ["gdp_for_year", "gdp_per_capita", "population", "suicides_100k_pop"]
data_log = data.copy()

for col in cols_to_check:
    if col in data_log.columns:
        data_log[col + "_log"] = np.log1p(data_log[col])

st.write("### Пример лог-трансформированных столбцов (data_log)")
st.write(data_log.head(10))

if "suicides_100k_pop_log" in data_log.columns:
    fig_log, ax_log = plt.subplots(figsize=(6,4))
    sns.histplot(data_log["suicides_100k_pop_log"], bins=30, kde=True, ax=ax_log, color="green")
    ax_log.set_title("Распределение log1p(suicides_100k_pop)")
    st.pyplot(fig_log)

# =========================================
# 6) Формируем X, y (с лог-трансформированным target)
# =========================================
st.write("## Подготовка данных для модели (LOG-трансформированный таргет)")

target_col = "suicides_100k_pop"
if target_col not in data.columns:
    st.error("В датасете нет столбца suicides_100k_pop. Приложение остановлено.")
    st.stop()

# Выбираем все признаки, кроме target
features = [c for c in data.columns if c != target_col]

X = data[features].select_dtypes(include=[np.number])
y = data[target_col]

# Логарифмируем целевую переменную: log1p(y) = ln(y+1)
y_log = np.log1p(y)

# На всякий случай убираем строки с NaN
valid_idx = X.dropna().index
X = X.loc[valid_idx]
y_log = y_log.loc[valid_idx]

# =========================================
# 7) train_test_split
# =========================================
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=test_size, random_state=42
)
st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# =========================================
# 8) Обучение нескольких регрессоров (на лог-таргете)
# =========================================
st.write("## Обучение моделей (LOG-трансформированный таргет)")

models_dict = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=max_depth_tree, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=n_estimators_rf, random_state=42),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=k_neighbors)
}

results = []
for name, model in models_dict.items():
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)       # Обратное преобразование
    y_test_real = np.expm1(y_test_log)  # Обратное преобразование для метрик

    mae = mean_absolute_error(y_test_real, y_pred)
    mse = mean_squared_error(y_test_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2
    })

res_df = pd.DataFrame(results).sort_values("RMSE")
st.dataframe(res_df)

# =========================================
# 9) Детальный просмотр для выбранной модели
# =========================================
st.write("## Детальный просмотр результатов (LOG-трансформированный таргет)")

model_name = st.selectbox(
    "Выберите модель",
    list(models_dict.keys())
)
selected_model = models_dict[model_name]
y_pred_log = selected_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
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

compare_df = pd.DataFrame({
    "Real": y_test_real.iloc[:10].values,
    "Predicted": y_pred[:10]
})
st.write("Примеры (первые 10):")
st.dataframe(compare_df)

# =========================================
# 10) Feature Importances для моделей
# =========================================
st.write("## Feature Importances для обученных моделей")

for model_name, model in models_dict.items():
    st.write(f"### {model_name}")
    # Для линейной регрессии
    if hasattr(model, "coef_"):
        importance = model.coef_
        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance
        })
        importance_df["abs_importance"] = importance_df["importance"].abs()
        importance_df = importance_df.sort_values(by="abs_importance", ascending=False)
        st.dataframe(importance_df.drop("abs_importance", axis=1))
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x="importance", y="feature", data=importance_df, ax=ax_imp)
        ax_imp.set_title(f"{model_name} Feature Importances (Coefficients)")
        st.pyplot(fig_imp)
    # Для деревьев и случайного леса
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        st.dataframe(importance_df)
        
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x="importance", y="feature", data=importance_df, ax=ax_imp)
        ax_imp.set_title(f"{model_name} Feature Importances")
        st.pyplot(fig_imp)
    else:
        st.write(f"Feature importance не доступна для модели {model_name}.")
