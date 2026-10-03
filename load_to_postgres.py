import os, pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

url = URL.create("postgresql+psycopg2", username="postgres", password=os.environ["PG_PASS"],
                 host="localhost", port=5432, database="churn")
engine = create_engine(url)
df = pd.read_csv("data/telco.csv")
df.to_sql("customers", engine, if_exists="replace", index=False)
print(pd.read_sql("SELECT Contract, COUNT(*) FROM customers GROUP BY Contract", engine))