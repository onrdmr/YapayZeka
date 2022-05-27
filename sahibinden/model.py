# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from scipy import stats


# %%
df = pd.read_csv('volvo_cleaned.csv')

print(df.info())

# %%
x = df.drop(['Fiyatı'], axis=1)
y = df['Fiyatı']

enc = OrdinalEncoder()
x = enc.fit_transform(x)
y = y.to_numpy()

# %%
kf = KFold(n_splits=10)
results = []
models = [SVR(), LinearRegression(), DecisionTreeRegressor(), KNeighborsRegressor(), RandomForestRegressor()]

# for model in models:
#     results.append(cross_val_score(model, x, y, cv = 10))



for model in models:
    sum = 0
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        sum = sum + model.score(x_test, y_test)
    results.append(sum / 10)
print(results)

# %%
plt.style.use('fivethirtyeight')
plt.figure(dpi=100)

plt.bar('SVR', results[0], label='Support Vector Regressor')
plt.bar('LR', results[1], label='Linear Regressor')
plt.bar('DTR', results[2], label='Decision Tree Regressor')
plt.bar('KNR', results[3], label='KNeighbors Regressor')
plt.bar('RFR', results[4], label='Random Forest Regressor')

plt.title('r2-score')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.savefig('r2scores.png', bbox_inches='tight')

plt.show()

# %%
LR = model[1].predict(x_test)
DT = model[2].predict(x_test)

# %%
KNREG = model[3].predict(x_test)
RFREG = model[4].predict(x_test)

# %%
plt.hist(LR,bins=20)

# %%
plt.hist(DT,bins=20)

# %%
plt.hist(RFREG,bins=20)


# %%
plt.hist(KNREG,bins=20)

# %%
print(stats.ttest_ind(KNREG, DT))
print(stats.ttest_ind(RFREG, DT))
print(stats.ttest_ind(LR, LR))
print(stats.ttest_ind(KNREG, RFREG))
print(stats.ttest_ind(LR, KNREG))

# %%
len(LR)

# %%
LR = model[1].predict(x)
DT = model[2].predict(x)

print(stats.ttest_ind(LR, DT))

# %%
for id in range(len(LR)):
    print(LR[id],y[id])
    # print(id)

# %%
len(y_train),len(y_test)

# %%
import seaborn as sns
df.head()


# %%
sns.displot(df, x="KM", col="Kimden")

# %%
sns.displot(df, x="Fiyatı", col="Garanti")
plt.savefig('hist_garanti.png')
plt.savefig('hist.png')


