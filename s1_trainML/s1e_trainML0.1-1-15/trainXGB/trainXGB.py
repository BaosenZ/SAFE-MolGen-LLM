# %%
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import joblib
import xgboost as xgb

# %% [markdown]
# ## Data preparation

# %%
dataset_root_path = "../1_convertToMLdata/"
trainVal_dataset_path = os.path.join(dataset_root_path, "trainVal_dataset.csv")
test_dataset_path = os.path.join(dataset_root_path, "test_dataset.csv")

# file path for trainVal SMILES
dataset_SMILES_path = "../0_splitData/"
trainVal_SMILES_path = os.path.join(
    dataset_SMILES_path, "output_trainset_uniqueSMILES.xlsx"
)

# %%
trainVal_dataset_df = pd.read_csv(trainVal_dataset_path)
test_dataset_df = pd.read_csv(test_dataset_path)

# Read trainVal SMILES
trainVal_SMILES_df = pd.read_excel(trainVal_SMILES_path)

# %%
# Get train/val indices stratifiedShuffleSplit by SMILES functional group

splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.15, random_state=14)
custom_indices = []

for train_idx, val_idx in splitter.split(
    trainVal_SMILES_df, trainVal_SMILES_df["Class_by_SMARTS_combineRare"]
):
    train_SMILES_set = trainVal_SMILES_df.iloc[train_idx]
    val_SMILES_set = trainVal_SMILES_df.iloc[val_idx]

    train_indices = trainVal_dataset_df[
        trainVal_dataset_df["SMILES"].isin(train_SMILES_set["SMILES"])
    ].index.tolist()
    val_indices = trainVal_dataset_df[
        trainVal_dataset_df["SMILES"].isin(val_SMILES_set["SMILES"])
    ].index.tolist()

    custom_indices.append((train_indices, val_indices))

print("Splits number: ", len(custom_indices))

# %%
print("Train size for one split: ", len(train_indices))
print("Val size for one split:", len(val_indices))
print("Val SMILES number: ", val_SMILES_set.shape[0])

# %%
# Split x and y

x_trainVal_df = trainVal_dataset_df.iloc[:, 0:1860]
y_trainVal_df = trainVal_dataset_df.iloc[:, 1860]
x_test_df = test_dataset_df.iloc[:, 0:1860]
y_test_df = test_dataset_df.iloc[:, 1860]

# Convert df to numpy array
x_trainVal = x_trainVal_df.to_numpy()
y_trainVal = y_trainVal_df.to_numpy()
x_test = x_test_df.to_numpy()
y_test = y_test_df.to_numpy()

print("x_trainVal shape: ", x_trainVal.shape)
print("y_trainVal shape: ", y_trainVal.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)

# %%
x_trainVal_df.head(3)

# %% [markdown]
# ## Build and train the model

# %% [markdown]
# Check the xgb doc for params we can tune: https://xgboost.readthedocs.io/en/stable/parameter.html

# %%
# Initialize the classifier
# We can use 'mlogloss' or 'merror' for 'eval_metric'
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax", num_class=4, eval_metric="merror"
)

param_dist = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.3, 0.5, 0.8],
    "max_depth": [12, 15, 18, 21],
    "min_child_weight": [0.5, 1, 3],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "alpha": [0, 0.01, 0.1, 0.5],
    "lambda": [0.5, 1, 1.5],
}

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_dist,
    scoring="accuracy",
    cv=custom_indices,
    refit=True,
    return_train_score=True,
    # n_jobs=-1
)

grid_search.fit(x_trainVal, y_trainVal)

# %%
# Save and print the training results

cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_sort_df = cv_results_df.sort_values(by="mean_test_score", ascending=False)

# Print top 5 models
top_5_models_df = cv_results_sort_df.head(5)
print("Top models:")
for index, row in top_5_models_df.iterrows():
    print(f"Rank {index+1}:")
    print(f"Parameters: {row['params']}")
    print(f"Mean Train Score: {row['mean_train_score']}")
    print(f"Mean Test Score: {row['mean_test_score']}")
    print("-" * 50)

# Save top 100 models
cv_results_sort_df.head(100).to_excel("cv_results_sort.xlsx", index=False)

# %%
print("Best parameters found: ", grid_search.best_params_)

# Get and save best estimator
best_xgb = grid_search.best_estimator_
model_filename = "xgb_model.sav"
joblib.dump(best_xgb, model_filename)
# print("Best iter found: ", best_xgb.best_iteration)
print(f"Model saved as {model_filename}")

# %% [markdown]
# ## Evaluate the model

# %%
# Load the saved model

xgb_model_name = "xgb_model.sav"
loaded_model = joblib.load(xgb_model_name)

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %% [markdown]
# ### Test set evaluation

# %%
y_test_pred = loaded_model.predict(x_test)

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# %%
# Save eval report
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred, output_dict=True)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual 0", "Actual 1", "Actual 2", "Actual 3"],
    columns=["Predicted 0", "Predicted 1", "Predicted 2", "Predicted 3"],
)
class_report_df = pd.DataFrame(class_report).transpose()

with pd.ExcelWriter("metrics_output.xlsx") as writer:
    conf_matrix_df.to_excel(writer, sheet_name="Confusion Matrix")
    class_report_df.to_excel(writer, sheet_name="Classification Report")
