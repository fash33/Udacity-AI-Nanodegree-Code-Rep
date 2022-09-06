import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = np.asarray(pd.read_csv('data.csv', header=None))

X = data[:,0:2]
y = data[:,2]


model = DecisionTreeClassifier(max_depth=5, min_samples_split=7 ,min_samples_leaf=3)
model.fit(X,y)
y_pred = model.predict(X)

acc = accuracy_score(y,y_pred)

print(acc)