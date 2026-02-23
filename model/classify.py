import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression

# data collection
df = pd.read_csv("data/parkinsons.data")

# data pre-processing
# print(df.head())
# print(df.isna().sum())
# print(df.shape)
# print(df.info())

# ref -> for doc

# data analysis using seaborn
# sns.countplot(x='status',data=df)

# co-relation of features
# plt.rcParams['figure.figsize'] = (15, 4)
# sns.pairplot(df,hue = 'status', vars = ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ', 'Jitter:DDP'] )
# plt.show() 

# drop -> unuseful identifiers
df = df.drop(columns=["name"])

x = df.drop(columns=["status"])
y = df["status"]

# splitting the data for test and train
X_train , X_test , Y_train , Y_test = train_test_split(x, y, test_size=0.20, random_state=20, stratify=y)

# pipeline = scale -> smote (balance) -> train & tune
pipeline = Pipeline([
    ("scaler", MinMaxScaler(feature_range=(-1,1))),
    ("smote",SMOTE(random_state=300)),
    ("rfc", RandomForestClassifier(random_state=42))
])

# tuning grid for better opti
param_grid = {
    "rfc__n_estimators": [150, 200],
    "rfc__max_depth": [5, 7, None],
    "rfc__max_features": ["sqrt"],
    "rfc__criterion": ["gini", "entropy"],
    "rfc__class_weight": ["balanced", {0:1, 1:2}] 
}

grid = GridSearchCV(pipeline,param_grid=param_grid,cv=5,scoring="recall",n_jobs=-1)

grid.fit(X_train,Y_train)

best_rfc = grid.best_estimator_
pred_rfc = best_rfc.predict(X_test)

# print("best param", grid.best_params_)
print(classification_report(Y_test,pred_rfc))

cm_rf = confusion_matrix(Y_test, pred_rfc)

print("cm", cm_rf)

# ref for doc
# plt.figure()
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title("Confusion Matrix - Random Forest")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# logistic regression

lrmodel = LogisticRegression()
lrmodel.fit(X_train, Y_train)
pred_lr = lrmodel.predict(X_test)

# print(classification_report(Y_test,pred_lr))

cm_lr = confusion_matrix(Y_test, pred_lr)
