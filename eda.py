import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # blanks become NaN

print(df.shape)
print(df.isnull().sum()[df.isnull().sum() > 0])
print(df["Churn"].value_counts(normalize=True))

sns.countplot(data=df, x="Contract", hue="Churn")
plt.savefig("eda_contract.png"); plt.clf()

sns.boxplot(data=df, x="Churn", y="tenure")
plt.savefig("eda_tenure.png"); plt.clf()

sns.heatmap(df.select_dtypes("number").corr(), annot=True)
plt.savefig("eda_corr.png")