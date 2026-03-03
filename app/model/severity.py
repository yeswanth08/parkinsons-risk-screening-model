# REGRESSION MODEL FOR SEVERITY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl

from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from IPython.display import display

# read csv
df = pd.read_csv('../data/parkinsons_updrs.data')

# data cleaning to avoid data leakage
df = df.drop(columns=['subject#','total_UPDRS'])
df = df.drop_duplicates()

# print(df.info())

# splitting the data
y = df['motor_UPDRS']
x = df.drop(columns=['motor_UPDRS'])

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.20,random_state=42)

# training model with different ML algos

# RANDOM FOREST REGRESSIVE MODEL WITH GIRDSEARCHCV

# pipeline = scale(normalize into genric number range)-> train & tune
rfr_model_pipeline = Pipeline([
    ('rfr',RandomForestRegressor(random_state=42))
])

# tuning grid
rfr_param_grid = {
    "rfr__n_estimators": [200],
    "rfr__max_depth": [20, None],
    "rfr__min_samples_split": [2, 5],
    "rfr__min_samples_leaf": [1, 2]
}

rfr_grid = GridSearchCV(
    rfr_model_pipeline,
    param_grid=rfr_param_grid,
    cv=3, 
    # cv = 5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

rfr_grid.fit(X_train,Y_train)

# get best fit from the grid using grid search and analyze
best_rfr = rfr_grid.best_estimator_
pred_rfr = best_rfr.predict(X_test)

rfr_mae = mean_absolute_error(Y_test,pred_rfr)
rfr_rmse = np.sqrt(mean_squared_error(Y_test,pred_rfr))
rfr_r2 = r2_score(Y_test,pred_rfr)



# SUPPORT VECTOR REGREESSION MODEL 

svr_model_pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('svr',SVR())
])

svr_param_grid = {
    "svr__kernel": ["rbf"],
    "svr__C": [1, 10, 100],
    "svr__gamma": ["scale", 0.01, 0.001],
    "svr__epsilon": [0.1, 0.2]
}

svr_grid = GridSearchCV(
    svr_model_pipeline,
    param_grid=svr_param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

svr_grid.fit(X_train, Y_train)

best_svr = svr_grid.best_estimator_
pred_svr = best_svr.predict(X_test)

svr_mae = mean_absolute_error(Y_test, pred_svr)
svr_rmse = np.sqrt(mean_squared_error(Y_test, pred_svr))
svr_r2 = r2_score(Y_test, pred_svr)


# --------------------------------------- used for testing another tree model ----------------------


# GRADIENT BOOSTING REGRESSION

# gbr_model_pipeline = Pipeline([
#     ('gbr',GradientBoostingRegressor(random_state=42))
# ])

# gbr_param_grid = {
#     "gbr__n_estimators": [200],
#     "gbr__learning_rate": [0.05, 0.1],
#     "gbr__max_depth": [3, 5]
# }

# gbr_grid = GridSearchCV(
#     gbr_model_pipeline,
#     param_grid=gbr_param_grid,
#     cv=3,
#     scoring="neg_mean_absolute_error",
#     n_jobs=-1
# )

# gbr_grid.fit(X_train, Y_train)

# best_gbr = gbr_grid.best_estimator_
# pred_gbr = best_gbr.predict(X_test)

# gbr_mae = mean_absolute_error(Y_test, pred_gbr)
# gbr_rmse = np.sqrt(mean_squared_error(Y_test, pred_gbr))
# gbr_r2 = r2_score(Y_test, pred_gbr)

# -------------------------------------------------------------


# XGBOOST REGRESSIOR

xgbr_model_pipeline = Pipeline([
    ('xgbr',XGBRegressor(objective="reg:squarederror",random_state=42,verbosity=0))
])

xgbr_param_grid = {
    "xgbr__n_estimators": [200],
    "xgbr__learning_rate": [0.05, 0.1],
    "xgbr__max_depth": [3, 5]
}

xgbr_grid = GridSearchCV(
    xgbr_model_pipeline,
    param_grid=xgbr_param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

xgbr_grid.fit(X_train, Y_train)

best_xgbr = xgbr_grid.best_estimator_
pred_xgbr = best_xgbr.predict(X_test)

xgbr_mae = mean_absolute_error(Y_test, pred_xgbr)
xgbr_rmse = np.sqrt(mean_squared_error(Y_test, pred_xgbr))
xgbr_r2 = r2_score(Y_test, pred_xgbr)

# REF FOR DOCUMENTAION

def plot_graph(y_true, y_pred, model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'r--'
    )
    
    plt.xlabel("Actual motor_UPDRS")
    plt.ylabel("Predicted motor_UPDRS")
    plt.title(f"{model_name} - Actual vs Predicted")
    
    plt.tight_layout()
    plt.show()


# plotting graph for documentaion
plot_graph(Y_test, pred_svr, "SVR")
plot_graph(Y_test, pred_xgbr, "XGBoost")
plot_graph(Y_test, pred_rfr, "Random Forest")

# comparision table -> for doc

comparision_table = {
    'Metric':["MAE", "RMSE", "R2"],
    'RFR': [rfr_mae,rfr_rmse,rfr_r2],
    # 'GBR': [gbr_mae,gbr_rmse,gbr_r2],
    'SVR': [svr_mae,svr_rmse,svr_r2],
    'XGBR':[xgbr_mae,xgbr_rmse,xgbr_r2]
}

comparision_table = pd.DataFrame(comparision_table)

display(comparision_table)


# from the obersvation random forest model has best r2 score and eff
# extracting the best bin during the run-time

jl.dump(rfr_model_pipeline,"../bin/severity_model.pkl")