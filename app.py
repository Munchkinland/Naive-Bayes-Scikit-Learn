
import sklearn

"""Generate DataSets"""

from sklearn.datasets import make_classification

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");

"""División en training y testing"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

"""Creating training models"""

from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[6]])

print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

"""Model evaluation using confusion matrix"""

labels = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();

"""Naive Bayes Classifier

Loading data
"""

import pandas as pd
df = pd.read_csv('loan_data.csv')
df.head()

"""Data Exploration"""

df.info()

"""Loading Data"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df,x='purpose',hue='not.fully.paid')
plt.xticks(rotation=45, ha='right');

"""Processing data"""

pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)
pre_df.head()

from sklearn.model_selection import train_test_split

X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train);

"""Model Evaluation Script"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred = model.predict(X_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

labels = ["Fully Paid", "Not fully Paid"]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();

"""True positive: 368
False negative: 29
False positive: 69
True negative: 6

✅ True Positive (TP): Represents the number of instances that were correctly classified as positive by the model. In this case, there are 368 cases that were positive, and the model correctly identified them as positive.

⛔ False Negative (FN): Indicates the number of instances that are actually positive but that the model incorrectly classified as negative. Here, there are 29 cases that are positive, but the model incorrectly classified them as negative.

⛔ False Positive (FP): Refers to the number of instances that are actually negative but that the model incorrectly classified as positive. There are 69 cases that are negative, but the model incorrectly classified them as positive.

✅ True Negative (TN): Represents the number of instances that were correctly classified as negative by the model. In this case, there are 6 cases that were negative, and the model correctly identified them as negative.

Otimize GaussianNB model

Hyperparameter tuning
"""

nb_classifier = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=nb_classifier,
                 param_grid=params_NB,
                 cv=cv_method,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')
gs_NB.fit(x_train, y_train)

gs_NB.best_params_

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Create the Gaussian Naive Bayes model
model = GaussianNB()

# Define the parameters to tune
params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}

# Configure GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=model,
                           param_grid=params_NB,
                           cv=5,  # You can adjust the number of cross-validation folds
                           scoring='accuracy',
                           verbose=1)

# Fit the model with grid search
grid_search.fit(X_train, y_train)

# Get the best model with the best hyperparameters
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Print metrics
print("Best hyperparameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("F1 Score:", f1)