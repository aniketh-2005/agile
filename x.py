import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv("lungcancer.csv")
df.head()
df.describe()
scaler = MinMaxScaler(feature_range=(1,2))
df[['AGE']] = scaler.fit_transform(df[['AGE']])
df.head()
le= LabelEncoder()
col=['GENDER','LUNG_CANCER']
for i in col:
    df[i]= le.fit_transform(df[i])
df.head(10000)
x=df.drop("LUNG_CANCER", axis=1)
y=df["LUNG_CANCER"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', random_state=0),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
    "Naive Bayes": GaussianNB()
}


accuracy_scores = {}


for name, model in models.items():
    model.fit(x_train, y_train)  
    y_pred = model.predict(x_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    accuracy_scores[name] = accuracy * 100 


plt.figure(figsize=(10, 5))

accuracy_scores_float = {k: float(v) for k, v in accuracy_scores.items()}

max_acc = max(accuracy_scores_float.values())

bar_colors = ["green" if abs(float(score) - max_acc) < 1e-6 else "red" for score in accuracy_scores_float.values()]

bars = sns.barplot(x=list(accuracy_scores_float.keys()), y=list(accuracy_scores_float.values()))

for bar, color in zip(bars.patches, bar_colors):
    bar.set_facecolor(color)

plt.ylabel("Accuracy Score (%)")
plt.xlabel("Models")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.show()
