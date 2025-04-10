import pandas as pd
import numpy as np

dataset = pd.read_csv(r"D:\DATA SCIENCE\15. Logistic regression with future prediction\Social_Network_Ads.csv")

x = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train, y_train)
print(bias)
variance = classifier.score(x_test, y_test)
print(variance)
#high bias and high variance = best fit model with 81% accuracy