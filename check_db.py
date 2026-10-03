import os, pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

url = URL.create("postgresql+psycopg2", username="postgres", password=os.environ["PG_PASS"],
                 host="localhost", port=5432, database="churn")
engine = create_engine(url)
print(pd.read_sql('SELECT "Contract", COUNT(*) FROM customers GROUP BY "Contract"', engine))
print(pd.read_sql('SELECT COUNT(*) AS total_rows FROM customers', engine))