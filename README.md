# Telecom Churn Prediction Pipeline
PostgreSQL, Pandas, Seaborn, Scikit-learn (Random Forest + GridSearchCV), MLflow, Prefect, Streamlit.

## Run
1. Download the Telco Customer Churn dataset (Kaggle) to data/telco.csv
2. Create a PostgreSQL database named churn and set PG_PASS
3. python load_to_postgres.py
4. python pipeline.py        (trains and logs to MLflow)
5. streamlit run app.py

## Results (test set)
Precision 0.53, Recall 0.76, F1 0.63, ROC-AUC 0.84
