import pandas as pd

df = pd.read_csv("./data/synthetic_data.csv",encoding="cp1252")
print(df.columns)
#print(list(df.loc[0]))

first_line = df.loc[30]
print(first_line)

print(df["reference"])
