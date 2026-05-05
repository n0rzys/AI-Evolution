import numpy as np
import pandas as pd
import random
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

np.random.seed(42)
random.seed(42)

data = pd.read_csv("data/heart_2020_cleaned.csv")
data = data.dropna()

y = data['HeartDisease']
data = data.drop(columns=['HeartDisease'])
y = LabelEncoder().fit_transform(y)

age_map = {val: i for i, val in enumerate(sorted(data['AgeCategory'].unique()))}
data['AgeCategory'] = data['AgeCategory'].map(age_map)

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
data['GenHealth'] = data['GenHealth'].map(gen_health_map)

numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
categorical_col = [c for c in data.columns if c not in numeric_col]

for col in numeric_col:
    data[col] = data[col].astype(float)
    upper = data[col].quantile(0.99)
    data.loc[data[col] > upper, col] = upper

for col in categorical_col:
    counts = data[col].value_counts()
    rare = counts[counts < 3].index
    data[col] = data[col].replace(rare, 'other')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(data[categorical_col])

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(data[numeric_col])

X_encoded_scaled = np.hstack([X_cat_encoded, X_num_scaled])
ratio = np.sum(y == 0) / np.sum(y == 1)

scoring = ['accuracy', 'f1', 'roc_auc']
X_train, X_val, y_train, y_val = train_test_split(X_encoded_scaled, y, test_size=0.2, random_state=42, stratify=y)

dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dval = xgb.QuantileDMatrix(X_val, label=y_val)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=False),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=False),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=False),

        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'scale_pos_weight': ratio,
        'eval_metric': 'auc'
        }

    model = XGBClassifier(**params, early_stopping_rounds=50)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model.best_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\nОтрисовка графиков...")
try:
    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.show()

    fig2 = optuna.visualization.plot_optimization_history(study)
    fig2.show()
except ImportError:
    print("Для графиков установи plotly: pip install plotly")

print(f"Лучший результат: {study.best_value:.4f}")
print(f"Параметры: {study.best_params}")

best_model = XGBClassifier(**study.best_params, device='cuda', tree_method='hist')
best_model.fit(X_train, y_train)

best_model.save_model("src/model_xgb.json")

with open("src/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("src/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

y_pred = best_model.predict(X_val)

print("\nФинальный отчет по классификации:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Реальность')
plt.xlabel('Прогноз модели')
plt.show()
