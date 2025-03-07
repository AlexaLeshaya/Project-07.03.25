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

# ==============================
# 1) Заголовок и краткое описание приложения
# ==============================
st.title("Прогнозирование уровня самоубийств на основании исторических данных")
st.write("""
Данный демо-проект повторяет логику ноутбука *Project 07.03.25.ipynb*,
демонстрирует загрузку и подготовку данных, обзор EDA и обучение
нескольких регрессионных моделей:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNN Regressor
""")

# ==============================
# 2) Боковая панель с настройками
# ==============================
st.sidebar.header("Параметры приложения")

test_size = st.sidebar.slider("Test size (доля тестовой выборки)", 0.1, 0.5, 0.2, 0.05)
max_depth_tree = st.sidebar.slider("Max depth (Decision Tree)", 1, 20, 5, 1)
n_estimators_rf = st.sidebar.slider("Число деревьев (Random Forest)", 10, 300, 100, 10)
k_neighbors = st.sidebar.slider("Число соседей (KNN)", 1, 15, 5, 1)

# ==============================
# 3) Функция загрузки + препроцессинга данных
# ==============================
@st.cache_data
def load_and_preprocess_data(csv_file: str):
    # 1. Загрузка
    data = pd.read_csv(csv_file)
    
    # 2. Удаляем нерелевантные столбцы (пример, как в ноутбуке)
    drop_cols = ['country-year', 'HDI for year', 'suicides_no']
    for c in drop_cols:
        if c in data.columns:
            data.drop(columns=[c], inplace=True)
    
    # 3. Переименуем, если нужно
    rename_dict = {
        'gdp_for_year ($)': 'gdp_for_year',
        'gdp_per_capita ($)': 'gdp_per_capita',
        'suicides/100k pop': 'suicides_100k_pop'
    }
    for old_name, new_name in rename_dict.items():
        if old_name in data.columns:
            data.rename(columns={old_name: new_name}, inplace=True)
    
    # 4. Преобразуем столбец gdp_for_year (если он строковый с запятыми)
    if 'gdp_for_year' in data.columns and data['gdp_for_year'].dtype == object:
        data["gdp_for_year"] = data["gdp_for_year"].str.replace(",", "", regex=False).astype(float)
    
    # 5. Маппим age (если нужно)
    age_map = {'5-14 years':0, '15-24 years':1, '25-34 years':2, '35-54 years':3, '55-74 years':4, '75+ years':5}
    if 'age' in data.columns:
        data['age'] = data['age'].map(age_map)
    
    # 6. Маппим sex (если нужно)
    sex_map = {'male':0, 'female':1}
    if 'sex' in data.columns:
        data['sex'] = data['sex'].map(sex_map)
    
    # 7. Одна горячая кодировка для generation (пример, если было у вас)
    if 'generation' in data.columns:
        data = pd.get_dummies(data, columns=['generation'])
    
    # Удаляем дубликаты
    data.drop_duplicates(inplace=True)

    # Смотрим на пропуски в suicides_100k_pop, выкидываем, если есть
    data.dropna(subset=['suicides_100k_pop'], inplace=True)

    # Масштабируем numeric
    numeric_cols = ['population', 'gdp_for_year', 'gdp_per_capita']
    numeric_cols = [col for col in numeric_cols if col in data.columns]
    
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

# ==============================
# 4) Загружаем датасет (пример: master.csv)
# ==============================
csv_file = "master.csv"  # подставьте свой путь, если нужно
data = load_and_preprocess_data(csv_file)

st.write("### Пример данных")
st.write(data.head(10))

st.write("**Размер датасета**:", data.shape)

# ==============================
# 5) EDA-краткий обзор (пара метрик, пара графиков)
# ==============================
st.write("### Краткий EDA")

# a) describe
st.write("Статистика по числовым признакам:")
st.dataframe(data.describe())

# b) график: распределение suicides_100k_pop
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.histplot(data['suicides_100k_pop'], bins=30, kde=True, ax=ax1, color='orange')
ax1.set_title("Распределение suicides_100k_pop")
st.pyplot(fig1)

# c) корреляция
df_numeric = data.select_dtypes(include=[np.number])
corr_matrix = df_numeric.corr()

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.heatmap(corr_matrix, cmap='viridis', annot=True, fmt=".2f", ax=ax2)
ax2.set_title("Correlation Matrix")
st.pyplot(fig2)

# ==============================
# 6) Формирование X, y и train_test_split
# ==============================
st.write("### Подготовка данных для модели")

target_col = 'suicides_100k_pop'
ignore_cols = ['country', 'suicides_100k_pop', 'suicides_no', 'year']  # или любые другие тексты, если остались
feature_cols = [c for c in data.columns if c != target_col and c not in ignore_cols]

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
st.write(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ==============================
# 7) Обучение нескольких регрессоров
# ==============================
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
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2
    })

res_df = pd.DataFrame(results).sort_values("RMSE")
st.dataframe(res_df)

# ==============================
# 8) Детальный обзор для выбранной модели
# ==============================
st.write("### Детальный просмотр результатов одной из моделей")

selected_model_name = st.selectbox(
    "Выберите модель для детального просмотра",
    list(models_dict.keys())
)
selected_model = models_dict[selected_model_name]
y_pred_sel = selected_model.predict(X_test)

# строим scatter "y_test vs y_pred"
fig3, ax3 = plt.subplots(figsize=(6,6))
ax3.scatter(y_test, y_pred_sel, alpha=0.5)
ax3.set_xlabel("Истинные значения suicides_100k_pop")
ax3.set_ylabel("Предсказанные значения")
ax3.set_title(f"{selected_model_name}: Real vs Predicted")

# Линия y=x
min_val = min(y_test.min(), y_pred_sel.min())
max_val = max(y_test.max(), y_pred_sel.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--')

st.pyplot(fig3)

# Показать несколько примеров
st.write("Примеры (первые 10 из тестовой выборки):")
comparison_df = pd.DataFrame({
    'Real': y_test.iloc[:10].values,
    'Predicted': y_pred_sel[:10]
})
st.dataframe(comparison_df)
