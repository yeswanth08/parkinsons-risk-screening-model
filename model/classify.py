# CLASSIFICATION MODEL FOR CLASSIFYING VOICE BTW HEALTHY AND PARKISON +VE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from sklearn.metrics import recall_score,f1_score,r2_score,log_loss,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from IPython.display import display

# data collection
df = pd.read_csv("data/parkinsons.data")

# data pre-processing

# print(df.head())
# print(df.isna().sum())
# print(df.shape)
# print(df.info())

# data cleaning to avoid data leakage (memo)
df = df.drop(columns=["name"])

# splitting the data 
x = df.drop(columns=["status"])
y = df["status"]

X_train , X_test , Y_train , Y_test = train_test_split(x, y, test_size=0.20, random_state=20, stratify=y)

# Training Models with vairous Ml Algos

# RANDOM FOREST MODEL WITH GRIDSEARCHCV

# pipeline = scale -> smote (balance) -> train & tune
rfc_model_pipeline = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(-1,1))),
    ("smote",SMOTE(random_state=300)),
    ("rfc", RandomForestClassifier(random_state=42))
])

# tuning grid for better opti
rfc_param_grid = {
    "rfc__n_estimators": [150, 200],
    "rfc__max_depth": [5, 7, None],
    "rfc__max_features": ["sqrt"],
    "rfc__criterion": ["gini", "entropy"],
    "rfc__class_weight": ["balanced", {0:1, 1:2}] 
}

rfc_grid = GridSearchCV(
    rfc_model_pipeline,
    param_grid=rfc_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rfc_grid.fit(X_train,Y_train)

# get best fit from the gird and then analyze
best_rfc_pipeline = rfc_grid.best_estimator_
pred_rfc = best_rfc_pipeline.predict(X_test)

# print("rfc",classification_report(Y_test,pred_rfc))

# confusion metrix
rfc_confusion_metrix_disp = ConfusionMatrixDisplay.from_estimator(
    best_rfc_pipeline,
    X_test,
    Y_test,
    cmap="Blues",
    display_labels=best_rfc_pipeline.named_steps['rfc'].classes_
)

plt.title("Random Forest Classifier Model - Confusion Matrix")
plt.show()


# LOGISTIC REGRESSION MODEL

# pipeline
lr_model_pipeline = Pipeline([
    ("scaler",MinMaxScaler(feature_range=(-1,1))),
    ("lr",LogisticRegression(max_iter=1000))
])

lr_model_pipeline.fit(X_train,Y_train)
pred_lr = lr_model_pipeline.predict(X_test)

# print("lr",classification_report(Y_test,pred_lr))

#confusion metrix
lr_cm = confusion_matrix(Y_test,pred_lr)
lr_confusion_metrix_disp = ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=lr_model_pipeline.named_steps["lr"].classes_
)

lr_confusion_metrix_disp.plot(cmap="Blues")
plt.title("Logistic Regression Model - confusion metrix")
plt.show()


# KNN CLASSIFIER MODEL WITH GRIDSEARCHCV

# pipeline
knn_model_pipeline = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(-1,1))),
    ("knn", KNeighborsClassifier())
])

# Searching for best K
knn_model_param_grid = {
    'knn__n_neighbors': range(2,11),
    'knn__weights': ['uniform','distance'],
    'knn__metric': ['euclidean','manhattan']
}

knn_grid = GridSearchCV(
    knn_model_pipeline,
    param_grid=knn_model_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

knn_grid.fit(X_train,Y_train)

# get best fit from grid search and then analyze
best_knn_pipeline = knn_grid.best_estimator_
pred_knn = best_knn_pipeline.predict(X_test)

# print("knn",classification_report(Y_test,pred_knn))

#confusion metrix
knn_cm = confusion_matrix(Y_test,pred_knn)
knn_confusion_matrix_disp = ConfusionMatrixDisplay(
    confusion_matrix=knn_cm,
    display_labels=best_knn_pipeline.named_steps['knn'].classes_
)

knn_confusion_matrix_disp.plot(cmap="Blues")
plt.title("Knn Model - Confusion Metrix")
plt.show()


# comparision table -> for doc

comparision_table = {
        'Metric':["Accuracy", "F1-Score", "Recall", "Precision", "R2-Score"],
        'RF':[accuracy_score(Y_test, pred_rfc), f1_score(Y_test, pred_rfc), recall_score(Y_test, pred_rfc), precision_score(Y_test, pred_rfc), r2_score(Y_test, pred_rfc)],
        'LR':[accuracy_score(Y_test, pred_lr), f1_score(Y_test, pred_lr), recall_score(Y_test, pred_lr), precision_score(Y_test, pred_lr), r2_score(Y_test, pred_lr)],
        'KNN':[accuracy_score(Y_test, pred_knn), f1_score(Y_test, pred_knn), recall_score(Y_test, pred_knn), precision_score(Y_test, pred_knn), r2_score(Y_test, pred_knn)],
}

comparision_table = pd.DataFrame(comparision_table)

display(comparision_table)