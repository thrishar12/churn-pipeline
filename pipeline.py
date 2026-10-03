import os, joblib, mlflow, pandas as pd
from prefect import flow, task
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

@task
def load():
    url = URL.create("postgresql+psycopg2", username="postgres", password=os.environ["PG_PASS"],
                     host="localhost", port=5432, database="churn")
    return pd.read_sql("SELECT * FROM customers", create_engine(url))

@task
def preprocess(df):
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df

@task
def train_eval(df):
    X, y = df.drop(columns="Churn"), df["Churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", StandardScaler(), num)])
    pipe = Pipeline([("pre", pre),
                     ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))])
    grid = {"rf__n_estimators": [100, 300], "rf__max_depth": [8, 15, None]}
    search = GridSearchCV(pipe, grid, scoring="f1", cv=3, n_jobs=-1).fit(Xtr, ytr)
    best = search.best_estimator_
    pred, proba = best.predict(Xte), best.predict_proba(Xte)[:, 1]
    metrics = {"precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
               "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba)}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn")
    with mlflow.start_run():
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    joblib.dump(best, "model.joblib")
    return metrics

@flow(name="churn-flow")
def churn_flow():
    print(train_eval(preprocess(load())))

if __name__ == "__main__":
    churn_flow()