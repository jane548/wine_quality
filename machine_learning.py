import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
wine = pd.read_csv('winequality-red.csv', sep=';')

# Print the first 5 rows of data
print(wine.head())

# Variables and data information
print()
wine.info()

# Check to see how many null values are in each column
print()
print(wine.isnull().sum())

# Preprocessing the data
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
wine['quality'].unique()

# Use 'bad = 0' and 'good = 1'
label_quality = LabelEncoder()

# Apply to the data
wine['quality'] = label_quality.fit_transform(wine['quality'])
print(wine.head(10))

# Count the bad and good wine
print(wine['quality'].value_counts())

# Display as a bar graph
sns.countplot(wine['quality'])
plt.show()

# Seperate the dataset as response variable and feature variables
X = wine.drop('quality', axis=1)  # All the features except 'quality'
y = wine['quality']

# Train and test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train[:10]

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)                               # Fit data to the model
predict_rfc = rfc.predict(X_test)                       # Make prediction
predict_rfc[:20]                                        # Predict first 20 variables
print(classification_report(y_test, predict_rfc))       # Check performance
print(confusion_matrix(y_test, predict_rfc))

# SVM Classifier
clf=svm.SVC()
clf.fit(X_train, y_train)                               # Fit data to the model
predict_clf = clf.predict(X_test)                       # Make prediction
print(classification_report(y_test, predict_clf))       # Check performance
print(confusion_matrix(y_test, predict_clf))

# Neural Network
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11), max_iter=1500)
mlpc.fit(X_train, y_train)                              # Fit data to the model
predict_mlpc = mlpc.predict(X_test)                     # Make prediction
print(classification_report(y_test, predict_mlpc))      # Check performance
print(confusion_matrix(y_test, predict_mlpc))

# Model precision
cm = accuracy_score(y_test, predict_rfc)
print(cm)

# Evaluation on new wine using random forest
wine.head(10)
Xnew = [[7.3, 0.58, 0.00, 2.0, 0.065, 15.0, 21.0, 0.9946, 3.36, 0.47, 10.0]]
Xnew = sc.transform(Xnew)
ynew = rfc.predict(Xnew)
ynew