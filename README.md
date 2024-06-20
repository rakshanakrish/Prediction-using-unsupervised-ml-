# Prediction-using-unsupervised-ml-
Prediction using unsupervised ml with the iris dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

iris = load_iris()
data = iris.data
target = iris.target

df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target
print(df.head())
sns.countplot(x='target', data=df)
plt.show()
corr = df.corr()
sns.heatmap(corr, annot=True,cmap='coolwarm')
plt.show()
x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler
