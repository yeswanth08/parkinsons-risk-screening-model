import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/parkinsons.data")

# # inspecting first few rows
# print(data.head())


# # checking the shape => gives entries and cols
# print("shape",data.shape)

# # this is a data sanity check for missing values as ml cannot handle missing values => we should aim for all zeros
# print("null chec\n",data.isnull().sum())

# # simple check for status of disease
# print(data["status"].value_counts())

# # feature type
# print("info",data.info())

# data cleaning 

# if a column is used to directly reveals the target(y) or strongly derived from the target then drop it

# # drop -> unuseful identifiers
data = data.drop(columns=["name"])

# # sepearing x and y for feed and outcome
y = data["status"]
x = data.drop(columns=["status"])


# # splitting train-test (80-20)
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# stratify = y -> bcs dataset is imbalanced we need to ensure the same ration of 0,1 are in test and train

# testing of data preparation & splitting

# print(X_test.shape,"x-test")
# print(X_train.shape,"x-train")

# print(Y_test.shape,"y-test")
# print(Y_train.shape,"y-train")

# print(Y_test.value_counts())
# print(Y_train.value_counts())

# standardization (to increase the accuracy of model by not to giving more priotiy to large value)
# scaling features
scaler = StandardScaler()

# # only train data
X_train = scaler.fit_transform(X_train)

# # use same on test data , using transform to avoid data leaking (learning from test data)
X_test = scaler.transform(X_test) 

# # model
model = LogisticRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

# # tesing the model
print("Acc",accuracy_score(Y_test,Y_pred))
print("\nconfusion matrix",confusion_matrix(Y_test,Y_pred))
print("\nclassification report",classification_report(Y_test,Y_pred))


