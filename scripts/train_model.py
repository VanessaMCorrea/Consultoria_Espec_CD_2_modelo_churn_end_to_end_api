import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "churn_pipeline.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["customerID"])
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.replace({
    "No internet service": "No",
    "No phone service": "No"
})

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# ✅ NOVO: divisão treino/teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numeric_pipeline, numeric_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    ))
])

# ✅ NOVO: treinar apenas com X_train
model.fit(X_train, y_train)

# ✅ NOVO: avaliar no conjunto de teste
y_pred = model.predict(X_test)
print("\n=== Métricas no conjunto de teste ===")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

joblib.dump(model, MODEL_PATH)

print(f"\nModelo salvo em: {MODEL_PATH}")
print("Colunas categóricas:", categorical_cols)
print("Colunas numéricas:", numeric_cols)