#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# %%
df = pd.read_csv('BankChurners.csv')
df = df[df.columns[:-2]]

#%%

# %%
features = ['Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
# %%
X = df[features]
y = df['Attrition_Flag']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# %%
model = XGBClassifier()
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# %%
