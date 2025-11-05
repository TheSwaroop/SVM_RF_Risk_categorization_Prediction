# smart_cafeteria_app.py
"""
Smart Cafeteria — Interactive Streamlit App
Predicts food waste and risk levels using Random Forest & SVM
with optional environmental data and hyperparameter tuning.

Run: streamlit run smart_cafeteria_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Smart Cafeteria System", layout="wide")
st.title("🥗 Smart Cafeteria — Food Waste Prediction & Risk Categorization")

# File upload
uploaded_file = st.file_uploader("📤 Upload integrated cafeteria dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Please upload your dataset to start analysis.")
    st.stop()

# ------------------------
# Load and preprocess data
# ------------------------
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Day'] = df['Day'].fillna('Unknown')
    df['Season'] = df['Season'].fillna('Unknown')

    le_day, le_season = LabelEncoder(), LabelEncoder()
    df['Day_Encoded'] = le_day.fit_transform(df['Day'])
    df['Season_Encoded'] = le_season.fit_transform(df['Season'])

    df['DayOfYear'] = df['Date'].dt.dayofyear.fillna(0).astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.fillna(0).astype(int)

    winter_foods = ['Meat', 'Fish', 'Chicken', 'Rice', 'Pasta', 'Potatoes', 'Soup', 'Beans', 'Lentils', 'Bread']
    summer_foods = ['Vegetables', 'Fruits', 'Milk', 'Cheese', 'Butter', 'Yogurt', 'Juice', 'Coffee', 'Icecream']

    for f in winter_foods + summer_foods:
        if f not in df.columns:
            df[f] = 0
        if f + "_Waste" not in df.columns:
            df[f + "_Waste"] = 0

    df['Winter_Food_Sum'] = df[winter_foods].sum(axis=1)
    df['Summer_Food_Sum'] = df[summer_foods].sum(axis=1)
    df['Total_Waste'] = df['Total_Waste'].fillna(df['Total_Waste'].mean())

    waste_columns = [f + "_Waste" for f in winter_foods + summer_foods]
    return df, waste_columns, le_day, le_season

df, waste_columns, le_day, le_season = load_and_preprocess(uploaded_file)
st.success(f"✅ Data loaded successfully — {df.shape[0]} rows, {df.shape[1]} columns.")

# ------------------------
# Sidebar settings
# ------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    include_env = st.toggle("Include Environmental Data", value=True)
    tune_hyperparams = st.toggle("Enable Hyperparameter Tuning (GridSearchCV)", value=True)
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random Seed", 0, 9999, 42)
    st.divider()

# ------------------------
# Prepare Data
# ------------------------
base_features = ['Attendance', 'Day_Encoded', 'Season_Encoded']
env_features = ['Avg_Temperature', 'DayOfYear', 'WeekOfYear', 'Winter_Food_Sum', 'Summer_Food_Sum']
features = base_features + (env_features if include_env else [])

df = df.dropna(subset=['Total_Waste'])
X = df[features]
y_reg = df['Total_Waste']
y_class = df['Risk_Level']

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=test_size, random_state=random_state)
X_train_c, X_test_c, y_class_train, y_class_test = train_test_split(X, y_class, test_size=test_size, random_state=random_state)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# ------------------------
# Train Models (with optional GridSearch)
# ------------------------
@st.cache_resource
def train_models(X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, tune=False):
    results = {}

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=random_state)
    if tune:
        params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
        grid = GridSearchCV(rf_reg, params, cv=3, scoring='r2')
        grid.fit(X_train, y_train_reg)
        rf_reg = grid.best_estimator_
    else:
        rf_reg.fit(X_train, y_train_reg)
    rf_pred = rf_reg.predict(X_test)
    results['RF_Regression'] = {'MSE': mean_squared_error(y_test_reg, rf_pred), 'R2': r2_score(y_test_reg, rf_pred)}

    # SVM Regressor
    svm_reg = SVR(kernel='rbf')
    if tune:
        params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        grid = GridSearchCV(svm_reg, params, cv=3, scoring='r2')
        grid.fit(X_train, y_train_reg)
        svm_reg = grid.best_estimator_
    else:
        svm_reg.fit(X_train, y_train_reg)
    svm_pred = svm_reg.predict(X_test)
    results['SVM_Regression'] = {'MSE': mean_squared_error(y_test_reg, svm_pred), 'R2': r2_score(y_test_reg, svm_pred)}

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=random_state)
    if tune:
        params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
        grid = GridSearchCV(rf_clf, params, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train_cls)
        rf_clf = grid.best_estimator_
    else:
        rf_clf.fit(X_train, y_train_cls)
    rf_pred = rf_clf.predict(X_test)
    results['RF_Classifier'] = {'Accuracy': accuracy_score(y_test_cls, rf_pred)}

    # SVM Classifier
    svm_clf = SVC(kernel='rbf', probability=True)
    if tune:
        params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        grid = GridSearchCV(svm_clf, params, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train_cls)
        svm_clf = grid.best_estimator_
    else:
        svm_clf.fit(X_train, y_train_cls)
    svm_pred = svm_clf.predict(X_test)
    results['SVM_Classifier'] = {'Accuracy': accuracy_score(y_test_cls, svm_pred)}

    return results, rf_reg, rf_clf, svm_reg, svm_clf

results, rf_reg, rf_clf, _, _ = train_models(X_train_s, X_test_s, y_reg_train, y_reg_test, y_class_train, y_class_test, tune=tune_hyperparams)
st.subheader("📊 Model Performance Summary")
st.dataframe(pd.DataFrame(results).T)

# ------------------------
# Seasonal Waste Visualization
# ------------------------
st.subheader("🌤 Seasonal Waste Patterns")
seasonal = df.groupby('Season')['Total_Waste'].mean().sort_values()
fig, ax = plt.subplots()
seasonal.plot(kind='bar', ax=ax)
ax.set_ylabel("Average Waste (kg)")
st.pyplot(fig)

# ------------------------
# Prediction Section
# ------------------------
st.divider()
st.subheader("🍽 Food Preparation Recommendation")

food_item = st.text_input("Enter a food item (e.g., Rice, Chicken, Vegetables):", "Rice")
if st.button("Predict Recommendation"):
    current_date = df['Date'].max()
    target_date = current_date + timedelta(days=30) if pd.notna(current_date) else datetime.now() + timedelta(days=30)

    avg_attendance = df['Attendance'].mean()
    day_encoded = le_day.transform([target_date.strftime('%A')])[0]
    season = 'Summer' if target_date.month in [6,7,8] else 'Winter'
    season_encoded = le_season.transform([season])[0] if season in le_season.classes_ else 0
    avg_temp = df[df['Season']==season]['Avg_Temperature'].mean()
    day_of_year = target_date.timetuple().tm_yday
    week_of_year = target_date.isocalendar().week
    winter_sum = df[df['Season']==season].filter(like='Meat').sum(axis=1).mean()
    summer_sum = df[df['Season']==season].filter(like='Vegetables').sum(axis=1).mean()

    input_features = np.array([[avg_attendance, day_encoded, season_encoded, avg_temp, day_of_year, week_of_year, winter_sum, summer_sum]])
    input_scaled = scaler.transform(input_features)

    predicted_total_waste = rf_reg.predict(input_scaled)[0]
    predicted_risk = rf_clf.predict(input_scaled)[0]
    risk_proba = rf_clf.predict_proba(input_scaled)[0][1] * 100

    waste_col = food_item + "_Waste"
    avg_ratio = df[waste_col].mean()/df["Total_Waste"].mean() if waste_col in df.columns else 0.05
    predicted_food_waste = predicted_total_waste * avg_ratio
    avg_prepared = df[food_item].mean() if food_item in df.columns else 0
    recommended_qty = max(0, avg_prepared - predicted_food_waste)

    st.success(f"📅 Prediction for {target_date.strftime('%d-%m-%Y')}")
    st.write(f"- **Predicted {food_item} Waste:** {predicted_food_waste:.2f} kg")
    st.write(f"- **Average {food_item} Prepared:** {avg_prepared:.2f} kg")
    st.write(f"- **Recommended {food_item} to Prepare:** {recommended_qty:.2f} kg")
    st.write(f"- **Waste Risk Level:** {predicted_risk} (High Risk Probability: {risk_proba:.2f}%)")
