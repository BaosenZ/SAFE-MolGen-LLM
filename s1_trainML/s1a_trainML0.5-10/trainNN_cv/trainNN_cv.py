# %% [markdown]
# # Training
#

# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

print("tf version: ", tf.__version__)
print("keras version: ", keras.__version__)
print("np version: ", np.__version__)
print("matplotlib version: ", matplotlib.__version__)
print("pd version: ", pd.__version__)
print("sklearn version: ", sklearn.__version__)
print("scikeras version: ", scikeras.__version__)

# %%
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json
import joblib

# %% [markdown]
# # Data preparation

# %%
dataset_root_path = "../1_convertToMLdata/"
trainVal_dataset_path = os.path.join(dataset_root_path, "trainVal_dataset.csv")
test_dataset_path = os.path.join(dataset_root_path, "test_dataset.csv")

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

for train_SMILES_idx, val_SMILES_idx in splitter.split(
    trainVal_SMILES_df, trainVal_SMILES_df["Class_by_SMARTS_combineRare"]
):
    train_SMILES_set = trainVal_SMILES_df.iloc[train_SMILES_idx]
    val_SMILES_set = trainVal_SMILES_df.iloc[val_SMILES_idx]

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
# # Build and train the model


# %%
# Define the model architecture with a variable number of hidden layers
def build_model(
    optimizer="adam",
    units=512,
    units2=128,
    activation="prelu",
    dropout_rate=0.3,
    learning_rate=1e-3,
    num_layers=3,
):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(1860,)))

    if num_layers == 2:
        if activation == "prelu":
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    kernel_initializer="normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.PReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=int(units2 / 2),
                    kernel_initializer="normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.PReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

        elif activation == "relu":
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.ReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=int(units2 / 2),
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.ReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

        else:
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=keras.initializers.GlorotNormal(),
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=int(units2 / 2),
                    activation=activation,
                    kernel_initializer=keras.initializers.GlorotNormal(),
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

    if num_layers == 3:
        if activation == "prelu":
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    kernel_initializer="normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.PReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=units2,
                    kernel_initializer="normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.PReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 3rd layer
            model.add(
                keras.layers.Dense(
                    units=16,
                    kernel_initializer="normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            model.add(keras.layers.PReLU())
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

        elif activation == "relu":
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=units2,
                    activation=activation,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 3rd layer
            model.add(
                keras.layers.Dense(
                    units=16,
                    activation=activation,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

        else:
            # 1st layer
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=keras.initializers.GlorotNormal(),
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 2nd layer
            model.add(
                keras.layers.Dense(
                    units=units2,
                    activation=activation,
                    kernel_initializer=keras.initializers.GlorotNormal(),
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

            # 3rd layer
            model.add(
                keras.layers.Dense(
                    units=16,
                    activation=activation,
                    kernel_initializer=keras.initializers.GlorotNormal(),
                    kernel_regularizer=keras.regularizers.l2(0.005),
                )
            )
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(rate=dropout_rate))
            model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(3, activation="softmax"))

    # Configure optimizer
    if optimizer == "adam":
        optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        optimizer_instance = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer_instance,
        metrics=["accuracy"],
    )

    return model


# %%
keras_clf = KerasClassifier(model=build_model, verbose=1)
pipeline = Pipeline([("scaler", StandardScaler()), ("keras_clf", keras_clf)])

param_grid = {
    "keras_clf__model__optimizer": ["adam", "rmsprop", "sgd"],
    "keras_clf__model__units": [128, 256, 384, 512],
    "keras_clf__model__units2": [64, 128],
    "keras_clf__model__activation": ["relu", "tanh", "prelu"],
    "keras_clf__model__dropout_rate": [0.0, 0.2, 0.3, 0.5],
    "keras_clf__model__learning_rate": [1e-3, 1e-4, 1e-5, 0.01],
    "keras_clf__fit__batch_size": [32, 64, 128],
    "keras_clf__model__num_layers": [2, 3],
    "keras_clf__fit__epochs": [100, 200, 300, 500, 800],
}

rand_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=100,
    random_state=42,
    scoring="accuracy",
    cv=custom_indices,
    refit=True,
    return_train_score=True,
    # n_jobs=-1,
    verbose=2,
)

rand_search_results = rand_search.fit(x_trainVal, y_trainVal)

# %%
# Save and print the training results

cv_results_df = pd.DataFrame(rand_search_results.cv_results_)
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
print("Best parameters found: ", rand_search_results.best_params_)

# Get best estimator
best_pipeline_nn = rand_search_results.best_estimator_
print(best_pipeline_nn.named_steps["keras_clf"].model_.summary())

# Save the model
pipeline_model_filename = "pipeline_nn_model.joblib"
joblib.dump(best_pipeline_nn, pipeline_model_filename)
print(f"Model saved as {pipeline_model_filename}")

# %%
# Save the training history to json file
with open("training_history.json", "w") as file:
    json.dump(best_pipeline_nn.named_steps["keras_clf"].history_, file)

# Open the json file
with open("training_history.json", "r") as file:
    history = json.load(file)

# %% [markdown]
# # Evaluate the model

# %%
pipeline_model_filename = "pipeline_nn_model.joblib"
loaded_pipeline_model = joblib.load(pipeline_model_filename)

# %%
y_trainVal_pred = loaded_pipeline_model.predict(x_trainVal)

print("TrainVal set eval: ")
print(confusion_matrix(y_trainVal, y_trainVal_pred))
print(classification_report(y_trainVal, y_trainVal_pred))

# %% [markdown]
# ### Test set evaluation

# %%
y_test_pred = loaded_pipeline_model.predict(x_test)

print("Test set eval: ")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# %%
# Save eval report
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred, output_dict=True)

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual 0", "Actual 1", "Actual 2"],
    columns=["Predicted 0", "Predicted 1", "Predicted 2"],
)
class_report_df = pd.DataFrame(class_report).transpose()

with pd.ExcelWriter("metrics_output.xlsx") as writer:
    conf_matrix_df.to_excel(writer, sheet_name="Confusion Matrix")
    class_report_df.to_excel(writer, sheet_name="Classification Report")
