# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,qmd
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %reload_ext jupyter_ai
# %load_ext jupyter_ai_magics

# +
# # %ai chatgpt --format code
# The best regression for a pandas dataframe

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14607, "status": "ok", "timestamp": 1715686115731, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="8942e27c-8c8b-4f36-a876-0cea800bbd6b" outputId="71b57740-c439-412d-e376-e2566f683382"
# #!pip install pandas numpy matplotlib xgboost scikit-learn tqdm
# -

# !pip install scikit-learn-intelex

# ## Optuna
#
# * https://optuna.org/
# * https://practicaldatascience.co.uk/machine-learning/how-to-tune-an-xgbregressor-model-with-optuna
# * https://practicaldatascience.co.uk/machine-learning/how-to-use-optuna-for-xgboost-hyperparameter-tuning
# * https://medium.com/optuna/using-optuna-to-optimize-xgboost-hyperparameters-63bfcdfd3407
# * https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/

# ! pip install optuna optuna-dashboard --quiet

OPTUNA_DB = "sqlite:///optuna.sqlite3"
get_ipython().system_raw(f"optuna-dashboard {OPTUNA_DB} &")

# ## MLFlow
# * https://www.kaggle.com/code/sharanharsoor/mlflow-end-to-end-ml-models

# !pip install mlflow 'mxnet<=1.9.1' --quiet
# !pip install pyngrok --quiet
get_ipython().system_raw("mlflow ui --port 5555 &")

# +
import mlflow

mlflow.autolog()
mlflow.lightgbm.autolog()
mlflow.xgboost.autolog()
# mlflow.gluon.autolog()
# -

# # Weight and Biases
# * https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W%26B.ipynb#scrollTo=agAgfdIeuPdXj
#

# !pip install --upgrade -q wandb

import wandb
WANDB_PROJECT = "intive-interstellar-sales-showdown"

try:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    wandb.login(wandb_api=client.get_secret("wandb_api"))
except ModuleNotFoundError:
    print("Falling back to environmenet variable")
    wandb.login()

# + [markdown] id="37978eaf-15a0-4760-a0d9-98ba2b0ed177"
# # Autogluon
# * https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html
# * https://auto.gluon.ai/stable/tutorials/tabular/advanced/index.html

# + _kg_hide-output=true colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 33805, "status": "ok", "timestamp": 1715657226953, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="1ffc441d-3476-4a29-a16f-a11ba279d563" outputId="137016a8-5c4e-447e-8f82-04450f0f7229"
# !pip install -U autogluon ipywidgets
# # !pip uninstall lightgbm -y
# # !pip install lightgbm --install-option=--gpu
# -

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

# # Links
#  * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
#  * https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics


import datetime
import pickle
import warnings
from dataclasses import dataclass
from datetime import date
from getpass import getpass
from pathlib import Path

# +
import lightgbm
import lightgbm as lgbm
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import optuna
import optuna.visualization as ov
import pandas as pd
import xgboost
import xgboost as xgb
from numpy import mean, std
from optuna.visualization import plot_intermediate_values
from pandas import read_csv
from pyngrok import ngrok
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearnex import patch_sklearn

patch_sklearn()

# + colab={"base_uri": "https://localhost:8080/", "height": 373} executionInfo={"elapsed": 929, "status": "error", "timestamp": 1715686147921, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="ca5a4a77-8d42-454f-b275-dbf4fdbf39a9" outputId="193769d9-0699-4f40-f2c1-170d512f12b9"
dataset_name = "original"
output_path = "."
# test_path = '/kaggle/input/dataset/test.csv'
# train_path = '/kaggle/input/dataset/train.csv'
test_path = f"{dataset_name}_test.csv"
train_path = f"{dataset_name}_train.csv"
test_csv = pd.read_csv(test_path)
train_csv = pd.read_csv(train_path)
# sample_submission = pd.read_csv('sample_submission.csv')
train_csv

# -

results = pd.DataFrame(columns=["dataset", "method", "variant", "rmse", "mse", "mae", "timestamp"])
results

# + colab={"base_uri": "https://localhost:8080/", "height": 141} executionInfo={"elapsed": 6, "status": "error", "timestamp": 1715686148819, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="c027fd42-b742-4b0c-8127-904d2026dd2d" outputId="29394d2a-2952-48b1-b2ed-d1426d3a2404"
train_csv

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1715652495729, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="e8831397-22a6-4e43-a908-510772184669" outputId="b5f36cf2-7892-4829-df0e-1e0f0588a211"
train_csv.columns

# + colab={"base_uri": "https://localhost:8080/", "height": 444} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1715652495729, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="516606cc-09c1-495b-ba3c-2689ddf49446" outputId="6fe19aff-e849-496e-a6e8-1e6eb9d19e4e"
test_csv

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1715652495729, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="d070b3e2-fac2-40fc-a8ca-9497dc990d51" outputId="9c76a57b-b994-430e-b7dc-0cd873c29777"
all_features = list(test_csv.columns)
test_csv.columns

# + id="482a627c-459e-4b56-8aa0-76ca54e5dea0"
patch_sklearn()

warnings.filterwarnings(
    action="ignore",
    message=".*Could not infer format.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*autologging encountered a warning.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*No visible GPU is found.*",
)
warnings.filterwarnings(
    action="ignore",
    message=".*deprecated binary model.*",
)


def load_numeric(file, datetime_to_numeric=True):
    csv = pd.read_csv(file)

    csv["created_date"] = pd.to_datetime(csv["created_date"])
    csv["refitted_date"] = pd.to_datetime(csv["refitted_date"])
    csv["refitted_date"] = csv["refitted_date"].fillna(csv["created_date"])

    csv["sale_date"] = pd.to_datetime(csv["sale_date"])
    age_baseline = max(
        csv["created_date"].max(), csv["refitted_date"].max(), csv["sale_date"].max()
    )

    calculate_age = (
        lambda x: age_baseline.year
        - x.year
        - ((age_baseline.month, age_baseline.day) < (x.month, x.day))
    )
    csv["created_age"] = csv["created_date"].apply(calculate_age)
    csv["refitted_age"] = csv["refitted_date"].apply(calculate_age)
    csv["sale_age"] = csv["sale_date"].apply(calculate_age)

    if datetime_to_numeric:
        csv["created_date"] = pd.to_numeric(csv["created_date"])
        csv["refitted_date"] = pd.to_numeric(csv["refitted_date"])
        csv["sale_date"] = pd.to_numeric(csv["sale_date"])
    print(csv.select_dtypes(include=["object"]).dtypes)
    return csv


def load_train(datetime_to_numeric=True):
    return load_numeric(train_path, datetime_to_numeric=datetime_to_numeric)


def load_test(datetime_to_numeric=True):
    return load_numeric(test_path, datetime_to_numeric=datetime_to_numeric)


def store_submission(alias, test_data, model=None):
    now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "")
    submissions = Path(output_path) / "submissions"
    submissions.mkdir(exist_ok=True)
    filename = f"{submissions}/{now}_{alias}_{dataset_name}_submission"
    if "id" not in test_data.columns:
        test_data["id"] = pd.read_csv("test.csv")["id"]
    test_data[["id", "price"]].to_csv(f"{filename}.csv", index=False)

    if model is not None:
        model_path = f"{filename}.pkl"
        pickle.dump(model, open(model_path, "wb"))


def predict_and_store(alias, model, test_x):
    test_data = load_test()
    test_data["price"] = model.predict(test_x)
    store_submission(alias, test_data, model)
    return test_data


@dataclass
class LocalDataset:
    X: pd.DataFrame
    y: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    test_data: pd.DataFrame
    train_data: pd.DataFrame
    features: list


@dataclass
class ModelTrain:
    X = None
    y = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    features = None
    model = None


def load_split(
    datetime_to_numeric=True,
    test_size=0.2,
    features=None,
    drop=None,
    standard_scaler=True,
):
    train_data = load_train(datetime_to_numeric=datetime_to_numeric)
    test_data = load_test(datetime_to_numeric=datetime_to_numeric)

    df = train_data
    X = df.drop(["price"], axis=1)
    if drop is not None:
        X = df.drop(drop, axis=1)
        test_data = test_data.drop(drop, axis=1)
    if features is None:
        features = list(X.columns)
    else:
        X = X[features]
        test_data = test_data[features]
        print(f"Keeping features {features}")
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1
    )

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    test_data = pd.DataFrame(test_data, columns=X.columns)
    train_data = pd.DataFrame(train_data[X.columns], columns=X.columns)
    if standard_scaler:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        test_data = pd.DataFrame(scaler.transform(test_data), columns=X.columns)
        train_data = pd.DataFrame(
            scaler.transform(train_data[X.columns]), columns=X.columns
        )
    train_data["price"] = y

    return LocalDataset(
        X, y, X_train, X_test, y_train, y_test, test_data, train_data, features
    )


def calculate_metrics(ld: LocalDataset, model, method, variant="", y_pred=None):
    if y_pred is None:
        y_pred = model.predict(ld.X_test)
    now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "")
    print("MSE: ", mean_squared_error(ld.y_test, y_pred))
    print("RMSE: ", np.sqrt(mean_squared_error(ld.y_test, y_pred)))
    print("MAE: ", mean_absolute_error(ld.y_test, y_pred))
    metrics = {
        "rmse": np.sqrt(mean_squared_error(ld.y_test, y_pred)),
        "mse": mean_squared_error(ld.y_test, y_pred),
        "mae": mean_absolute_error(ld.y_test, y_pred),
    }
    results.loc[len(results) + 1] = {
        "dataset": dataset_name,
        "method": method,
        "variant": variant,
        "rmse": np.sqrt(mean_squared_error(ld.y_test, y_pred)),
        "mse": mean_squared_error(ld.y_test, y_pred),
        "mae": mean_absolute_error(ld.y_test, y_pred),
        "timestamp": now
    }
    mlflow.log_metrics(metrics)
    results.to_csv(f"results.csv", index=False)
    return results.sort_values(by="rmse")


_ = load_split()


# + [markdown] id="afed3f42-002c-4bb2-9d33-7120b3b71a83"
# # Sample code


# + id="80971d57-5308-4a69-ae51-69234708e2b2"
def execute_sample_code(features=["cabins", "decks", "bathrooms", "radar"], variant=""):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    with mlflow.start_run(
        run_name=f"linear-{variant}",
        tags={"version": "v1", "library": "sklearn", "optimization": "n/a"},
        description="sklearn",
    ):
        # load training and test data
        ld = load_split(features=features)
        train_data = ld.X_train
        # choose 4 features and price as training target
        # features = ['cabins', 'decks', 'bathrooms', 'radar']
        # train_x = train_data[features]
        # train_y = ld.y_train

        # simple linear regression model
        # price = a * cabins + b * decs + c * bathrooms + d * radar + e
        model = LinearRegression().fit(ld.X_train, ld.y_train)

        # predict price on test data
        # test_x = ld.test_data[features]
        # test_data = predict_and_store('linear_regression', model, test_x)

        # y_pred = model.predict(ld.X_test)
        # print('MSE: ', mean_squared_error(ld.y_test, y_pred))
        # print('RMSE: ', np.sqrt(mean_squared_error(ld.y_test, y_pred)))
        # print('MAE: ', np.sqrt(mean_absolute_error(ld.y_test, y_pred)))
        calculate_metrics(ld, model, method="linear", variant=variant)

        # mlflow.sklearn.log_model(sk_model=model, input_example=ld.X_train.sample(10))
        # mlflow.log_params(params)

        return model, ld.X_train


sample_model, sample_x = None, None
sample_model, sample_x = execute_sample_code(variant="demo")


# + [markdown] id="55c26323-abd4-4321-8eb7-bcae397c1dd7"
# # SelectKBest features
#
# * https://lifewithdata.com/2022/03/19/feature-selection-with-selectkbest-in-scikit-learn/


# + colab={"base_uri": "https://localhost:8080/", "height": 530} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1715652578103, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="4bdce252-e21c-49e5-82e6-0d8dfcfd5b4d" outputId="cbd0ba53-5f13-412c-c375-08cfe746a1b2"
def select_features(k=10):
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import MinMaxScaler

    # Assuming X and y are your input features and target labels, respectively
    # Select the top 10 features
    selector = SelectKBest(score_func=chi2, k=k)

    df = load_train()
    X = df[[col for col in df.columns if col != "price"]]
    y = df["price"]

    sc = MinMaxScaler()
    X_sc = sc.fit_transform(X)

    # X_new contains the selected features
    X_new = selector.fit_transform(X_sc, y)
    X_new

    kept_features = pd.DataFrame({"columns": X.columns, "Kept": selector.get_support()})
    #     print(kept_features)

    new_train_csv = X.iloc[:, selector.get_support()]
    print(list(new_train_csv.columns))
    return list(new_train_csv.columns), new_train_csv


kept_features, train_csv_kept_features = select_features()
train_csv_kept_features
# -

_ = execute_sample_code(features=kept_features, variant="kbest10")

_ = execute_sample_code(features=select_features(k=5)[0], variant="kbest5")

_ = execute_sample_code(features=select_features(k=15)[0], variant="kbest15")

# + id="0beb77ab-05e3-45fc-b1d8-e0c13ec7c285" outputId="b535b467-e946-4088-a9c9-f79649f5e88a"
_ = execute_sample_code(features=select_features(k=20)[0], variant="kbest20")

# + [markdown] id="6bbb0575-7551-4260-86f0-39b3fcfe811e"
# # SkLearn Pipeline
#
# https://machinelearningmastery.com/feature-extraction-on-tabular-data/

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 808, "status": "ok", "timestamp": 1715652594890, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="8f1c5bd8-4d9c-49b5-95d7-322e17661bc4" outputId="a7d7c0e2-d391-4025-a456-89d76d5fb59e"

train_data = load_train()
test_data = load_test()

df = train_data
X = df[[col for col in df.columns if col != "price"]]
# features = list(new_train_csv.columns)
# X = df[features]
y = df["price"]

# X = X.astype('float')
# y = LabelEncoder().fit_transform(y.astype('str'))

print(X.shape, y.shape)


# + id="59722030-85fc-4d64-8105-7e2f0f304d1a" outputId="d5cac7f2-fc55-4aca-e945-8132fe9100ff"
model = LogisticRegression(solver="liblinear")
# cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)
# scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# -


def show_results():
    return results.sort_values(by="rmse")


show_results()

# +

# test_data = predict_and_store('linear_regression_pipeline', model, X)
# mean_absolute_error(train_data['price'], model.predict(X))

# + id="68a251a7-efdb-483a-b42a-fb00720d5e6f"

# transforms for the feature union
transforms = list()
transforms.append(("mms", MinMaxScaler()))
transforms.append(("ss", StandardScaler()))
transforms.append(("rs", RobustScaler()))
transforms.append(
    ("qt", QuantileTransformer(n_quantiles=100, output_distribution="normal"))
)
transforms.append(
    ("kbd", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform"))
)
transforms.append(("pca", PCA(n_components=7)))
transforms.append(("svd", TruncatedSVD(n_components=7)))
# create the feature union
fu = FeatureUnion(transforms)

# + colab={"base_uri": "https://localhost:8080/", "height": 411} executionInfo={"elapsed": 2780843, "status": "error", "timestamp": 1715655403732, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="599d808e-c2b5-42fa-b52c-59a34e66ed5c" outputId="88e1b091-82ec-4ad4-fa3e-43b3da64aa7e"
model = LogisticRegression(solver="liblinear")
rfe = RFE(estimator=LogisticRegression(solver="liblinear"), n_features_to_select=8)
steps = list()
steps.append(("fu", fu))
steps.append(("rfe", rfe))
steps.append(("m", model))
pipeline = Pipeline(steps=steps)

# cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)
# scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# -

# # Autogloun

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1403, "status": "ok", "timestamp": 1715656125340, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="66b74089-14b1-433d-b5c4-b93e59416fad" outputId="43099207-0bc5-4821-d78b-0f11c0dffeee"

ld_autogluon = load_split(datetime_to_numeric=False, standard_scaler=False)

X = ld_autogluon.X
y = ld_autogluon.y

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 705, "status": "ok", "timestamp": 1715656130738, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="dd333822-1c3f-4319-b0e4-aae6ab471312" outputId="ec426427-0987-49e3-b91a-8533c48ea5d7"
label = "price"
ld_autogluon.train_data[label].describe()
# -


auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=X, y=y)

with mlflow.start_run(
    run_name="linear-automlfeatures",
    tags={
        "version": "v1",
        "library": "sklearn",
        "optimization": "AutoMLPipelineFeatureGenerator",
    },
    description="sklearn with AutoMLPipelineFeatureGenerator",
):
    model = LinearRegression().fit(
        auto_ml_pipeline_feature_generator.transform(X=ld_autogluon.X_train),
        ld_autogluon.y_train,
    )
    calculate_metrics(
        ld_autogluon,
        model,
        method="linear",
        variant="automlfeatures",
        y_pred=model.predict(
            auto_ml_pipeline_feature_generator.transform(X=ld_autogluon.X_test)
        ),
    )


# + colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="f04a1816-ec43-490d-b76d-3300b0326ea9" outputId="91a560c1-3a1d-4c7e-d941-137b685fb293"

predictor = None

time_limit = 3600
TIME_LIMIT_BASE=1800
presets = "medium_quality"


def execute_autogluon(
    time_limit=3600,
    presets="medium_quality",
    num_bag_folds=5,
    num_bag_sets=1,
    num_stack_levels=1,
    calibrate_decision_threshold=True,
):
    with mlflow.start_run(
        run_name=f"autogluon-{presets}",
        tags={"version": "v1", "library": "autogluon", "optimization": presets},
        description="autogluon",
    ):
        predictor = TabularPredictor(
            label=label, problem_type="regression", path=f"AutogluonModels/{presets}"
        ).fit(
            ld_autogluon.train_data,
            num_gpus=1,
            presets=presets,
            time_limit=time_limit,
            dynamic_stacking=False,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            num_stack_levels=num_stack_levels,
            calibrate_decision_threshold=calibrate_decision_threshold,
            feature_generator=auto_ml_pipeline_feature_generator,
        )
        calculate_metrics(
            ld_autogluon,
            model,
            method="autogluon",
            variant=f"{presets}_{time_limit}s",
            y_pred=predictor.predict(
                auto_ml_pipeline_feature_generator.transform(X=ld_autogluon.X_test)
            ),
        )
        metrics = predictor.evaluate(ld_autogluon.train_data, silent=True)
        mlflow.log_metrics(metrics)
        predictor.leaderboard(ld_autogluon.train_data)
        predict_and_store(f"autogluon_{presets}", predictor, ld_autogluon.test_data)
        return predictor


# + id="df7d9348-8c06-4586-921a-1df8d7d9ebf7"
medium_quality_predictor = execute_autogluon(presets="medium_quality", time_limit=TIME_LIMIT_BASE)
medium_quality_predictor.evaluate(ld_autogluon.train_data, silent=True)
# -

medium_quality_predictor.feature_importance(ld_autogluon.train_data, time_limit=int(0.5 * TIME_LIMIT_BASE))

# + id="f92cd337-2337-40ef-9ccc-5c5b98dd1ef4"
medium_quality_predictor.leaderboard(ld_autogluon.train_data)
# -

high_quality_predictor = execute_autogluon(presets="high_quality", time_limit=TIME_LIMIT_BASE * 2)
high_quality_predictor.leaderboard(ld_autogluon.train_data)

best_quality_predictor = execute_autogluon(presets="best_quality", time_limit=TIME_LIMIT_BASE * 3)
best_quality_predictor.leaderboard(ld_autogluon.train_data)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6115, "status": "ok", "timestamp": 1715656755966, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="83lfUlnyGlCp" outputId="50a169a1-56a8-4d00-c060-579908002500"
# test_data = load_test(datetime_to_numeric=False)
# df = auto_ml_pipeline_feature_generator.transform(test_data)
# test_x = df.drop([], axis=1)
# predict_and_store(f"autogluon_{presets}", medium_quality_predictor, test_x)
# -

show_results()


# # Optuna + Autogluon


def execute_autogluon_optuna(
    presets="medium_quality", time_limit=3600, n_trials=10, study_alias=""
):
    with mlflow.start_run(
        run_name="autogluon_optuna",
        tags={"version": "v1", "library": "autogluon", "optimization": "optuna"},
        description="autogluon_optuna",
    ) as parent_run:
        mlflow.log_param("parent", "yes")

        def objective(trial):
            number = trial.number
            with mlflow.start_run(
                run_name=f"autogluon_optuna_trial_{number}",
                description="autogluon_optuna_trial",
                tags={
                    "version": "v1",
                    "library": "autogluon",
                    "optimization": "optuna",
                    "trial": "true",
                },
                nested=True,
            ):
                params = {
                    "num_bag_folds": trial.suggest_int("num_bag_folds", 2, 5),
                    "num_bag_sets": trial.suggest_int("num_bag_sets", 1, 5),
                    "num_stack_levels": trial.suggest_int("num_stack_levels", 1, 5),
                    "calibrate_decision_threshold": trial.suggest_categorical(
                        "calibrate_decision_threshold", [False]
                    ),
                }
                ld = load_split()

                predictor = TabularPredictor(
                    label=label,
                    problem_type="regression",
                    path=f"AutogluonModels/{study_alias}_{presets}_trial_{number}_n{n_trials}_{time_limit}s",
                ).fit(
                    ld_autogluon.train_data,
                    num_gpus=1,
                    presets=presets,
                    time_limit=time_limit,
                    dynamic_stacking=False,
                    feature_generator=auto_ml_pipeline_feature_generator,
                    **params,
                )
                model = predictor
                y_pred = model.predict(ld.X_test)

                calculate_metrics(ld, model, method="autogluon_optuna_trial", variant="trial")

                return mean_squared_error(ld.y_test, y_pred)

        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna.sqlite3",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            study_name=f"autogluon-regression_{study_alias}_n{n_trials}_{time_limit}s_{presets}",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials)

        ld = load_split()
        best_trial = study.best_trial
        # model = TabularPredictor(label=label, problem_type="regression", path=f'AutogluonModels/optuna-{presets}').fit(
        #     ld_autogluon.train_data,
        #     num_gpus=1,
        #     presets=presets,
        #     time_limit=time_limit,
        #     dynamic_stacking=False,
        #     feature_generator=auto_ml_pipeline_feature_generator,
        #     **study.best_params,
        # )
        number = best_trial.number
        model = TabularPredictor.load(f"AutogluonModels/{study_alias}_{presets}_trial_{number}_n{n_trials}_{time_limit}s")
        # model.fit(ld.X, ld.y)
        # y_pred = model.predict(ld.X_test)

        calculate_metrics(ld, model, method="autogluon-optuna", variant=f"{presets}_n{n_trials}_{time_limit}s_{study_alias}")
        predict_and_store(f"autogluon-optuna_{presets}_{number}_n{n_trials}_{time_limit}s", model, ld_autogluon.test_data)

        return model, study


N_TRIALS=1
_, study = execute_autogluon_optuna(
    presets="medium_quality", n_trials=N_TRIALS, time_limit=TIME_LIMIT_BASE, study_alias="first"
)

# +
# _, hq_study = execute_autogluon_optuna(
#     presets="high_quality", n_trials=N_TRIALS * 2, time_limit=TIME_LIMIT_BASE * 2, study_alias="first"
# )

# +
# _, best_study = execute_autogluon_optuna(
#     presets="best_quality", n_trials=N_TRIALS * 3, time_limit=TIME_LIMIT_BASE * 3, study_alias="first"
# )
# -

show_results()[0:20]

ov.plot_optimization_history(study)

ov.plot_intermediate_values(study)

ov.plot_timeline(study)

ov.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)

# + [markdown] id="575da63f-9dab-4654-a7ae-65ed43d4a33b"
# # XGBoost
#
# * https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10611, "status": "ok", "timestamp": 1715652341691, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="3d23fc5e-82c6-4391-9e91-f8e577cefa6f" outputId="c909bdcf-67b3-4703-9a9c-37d15914fc9d"
# !pip install xgboost

# bst = xgboost.train(...)
# config = bst.save_config()
# print(config)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1489, "status": "ok", "timestamp": 1715655420922, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="a355108f-48cf-4739-933c-c00d64fab90a" outputId="fa23da10-32d5-496a-9398-d16f27b4d9a3"


def execute_xgboost(features=None, variant="default"):
    import numpy as np
    import xgboost as xgb
    from numpy import mean, std

    with mlflow.start_run(
        run_name=f"xgboost-{variant}",
        tags={"version": "v1", "library": "xgboost", "optimization": "n/a"},
        description="xgboost",
    ):
        ld = load_split(features=features)
        train_data = ld.X_train

        model = xgb.XGBRegressor(n_jobs=1, tree_method="hist", device="cuda")
        model = model.fit(ld.X_train, ld.y_train)

        calculate_metrics(ld, model, method="xgboost", variant=variant)

        return model, ld.X_train


# + colab={"base_uri": "https://localhost:8080/", "height": 321} executionInfo={"elapsed": 5, "status": "error", "timestamp": 1715655461028, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="e93693c6-afbb-434e-9a55-dc2618146d49" outputId="fa40bc3b-361d-4f97-bf59-9b58fc25df8b"
_ = execute_xgboost(features=select_features(k=10)[0], variant="kbest10")
_ = execute_xgboost(features=select_features(k=20)[0], variant="kbest20")
_ = execute_xgboost()

show_results()


# + id="1e2658a5-4a5a-4113-aa36-737e441de64f"


def execute_xgboost_cv(features=None, variant="default"):
    import numpy as np
    import xgboost as xgb
    from numpy import mean, std
    from sklearn.model_selection import RepeatedKFold, cross_val_score

    with mlflow.start_run(
        run_name=f"xgboost-cv-{variant}",
        tags={"version": "v1", "library": "xgboost", "optimization": "cv"},
        description="xgboost cross validation",
    ):
        ld = load_split(features=features)
        train_data = ld.X_train

        model = xgb.XGBRegressor(n_jobs=1, tree_method="hist", device="cuda")
        model = model.fit(ld.X, ld.y)

        cv = RepeatedKFold(n_splits=3, n_repeats=10, random_state=1)
        scores = cross_val_score(model, ld.X, ld.y, cv=cv, n_jobs=-1, verbose=1)

        calculate_metrics(ld, model, method="xgboost-cv", variant=variant)
        mlflow.log_metrics({"cv-mean": mean(scores), "cv-std": std(scores)})

        return model, ld.X


# xgb_model = xgb.XGBRegressor(n_jobs=1, tree_method="hist", device="cuda")
# xgb_model.fit(X, y)
# cv = RepeatedKFold(n_splits=3, n_repeats=10, random_state=1)
# scores = cross_val_score(xgb_model, X, y, cv=cv, n_jobs=-1, verbose=1)
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# -

_ = execute_xgboost_cv(features=select_features(k=10)[0], variant="kbest10")
_ = execute_xgboost_cv(features=select_features(k=20)[0], variant="kbest20")
_ = execute_xgboost_cv()


# + id="1b0988c2-5760-4800-8fdb-a2e436c7a758"
show_results()
# -


# # Optuna + XGBoost

# +
xgboost_study = None
xgboost_optuna_model = None
def execute_xgboost_optuna(n_trials):
    with mlflow.start_run(
        run_name="xgboost_optuna",
        tags={"version": "v1", "library": "xgboost", "optimization": "optuna"},
        description="xgboost_optuna",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
    
        def objective(trial):
            number = trial.number
            with mlflow.start_run(
                run_name=f"xgboost_optuna_trial_{number}",
                description="xgboost_optuna_trial",
                tags={
                    "version": "v1",
                    "library": "xgboost",
                    "optimization": "optuna",
                    "trial": "true",
                },
                nested=True,
            ):
                param = {
                    "max_depth": trial.suggest_int("max_depth", 1, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.01, 1.0),
                    "subsample": trial.suggest_float("subsample", 0.01, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
                    "random_state": trial.suggest_int("random_state", 1, 1000),
                    "device": "cuda",
                    "tree_method": "hist",
                }
                ld = load_split()
    
                model = xgb.XGBRegressor(**param)
                model = model.fit(ld.X_train, ld.y_train)
                y_pred = model.predict(ld.X_test)
    
                calculate_metrics(ld, model, method="xgboost_optuna_trial")
    
                return mean_squared_error(ld.y_test, y_pred)
    
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna.sqlite3",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            study_name="xgboost-regression",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials)
    
        ld = load_split()
        model = xgboost.XGBRegressor(**study.best_params)
        model.fit(ld.X, ld.y)
        y_pred = model.predict(ld.X_test)
        xgboost_study = study
        xgboost_optuna_model = model
    
        calculate_metrics(ld, model, method="xgboost-optuna", variant=f"n{n_trials}")
        predict_and_store("xgboost-optuna", xgboost_optuna_model, ld.test_data)
        return model, study

xgboost_optuna_model, xgboost_study = execute_xgboost_optuna(n_trials=N_TRIALS * 5)
# -

plot_intermediate_values(xgboost_study)

show_results()

# # Optuna + LightGBM
# * https://practicaldatascience.co.uk/machine-learning/how-to-tune-a-lightgbmclassifier-model-with-optuna
# * https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.htmloptuna

# +
dataset = load_split(datetime_to_numeric=True)

lightgbm_model = None
def execute_lightgbm(features=None, variant="default"):
    with mlflow.start_run(
        run_name="lightgbm_optuna_baseline",
        tags={"version": "v1", "library": "lightgbm", "optimization": "none"},
        description="lightgbm_optuna_baseline",
    ):
        ld = load_split(features=features)
        model = lgbm.LGBMRegressor()
        model.fit(ld.X_train, ld.y_train)
    
        y_pred = model.predict(ld.X_test)
        lightgbm_model = model
    
        calculate_metrics(ld, model, method="lightgbm", variant=variant)
        return model



# +
_ = execute_lightgbm(features=select_features(k=10)[0], variant="kbest10")
_ = execute_lightgbm(features=select_features(k=20)[0], variant="kbest20")
_ = execute_lightgbm()

show_results()
# -

lightgbm_study = None
lightgbm_optuna_model = None
def execute_lightgbm_optuna(n_trials):
    with mlflow.start_run(
        run_name="lightgbm_optuna",
        tags={"version": "v1", "library": "lightgbm", "optimization": "optuna"},
        description="lightgbm_optuna",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
    
        def objective(trial):
            number = trial.number
            with mlflow.start_run(
                run_name=f"lightgbm_optuna_trial_{number}",
                description="lightgbm_optuna_trial",
                tags={
                    "version": "v1",
                    "library": "lightgbm",
                    "optimization": "optuna",
                    "trial": "true",
                },
                nested=True,
            ):
                param = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-2, 0.25, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 1, 9),
                    "n_estimators": trial.suggest_categorical(
                        "n_estimators", [7000, 15000, 20000]
                    ),
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "metric": "rmse",
                    "verbose": -1,
                }
                
                ld = load_split()
                model = lgbm.LGBMRegressor(**param)
                model = model.fit(ld.X_train, ld.y_train)
                y_pred = model.predict(ld.X_test)
    
                calculate_metrics(ld, model, method="xgboost_optuna_trial")
    
                return mean_squared_error(ld.y_test, y_pred)
    
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///optuna.sqlite3",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
            study_name="lightgbm-regression",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials)
    
        ld = load_split()
        model = lgbm.LGBMRegressor(**study.best_params)
        model.fit(ld.X, ld.y)
        y_pred = model.predict(ld.X_test)
        lightgbm_study = study
        lightgbm_optuna_model = model

        calculate_metrics(ld, model, method="lightgbm-optuna", variant=f"n{n_trials}")
        predict_and_store("lightgbm-optuna", lightgbm_optuna_model, ld.test_data)
        return model, study


lightgbm_optuna_model, lightgbm_study = execute_lightgbm_optuna(n_trials=N_TRIALS * 5)

show_results()

# # cuML
#
# * https://docs.rapids.ai/api/cuml/stable/cuml_intro/#where-possible-match-the-scikit-learn-api

#

# + [markdown] id="5621936a-cfe1-4034-a0eb-62ad82abb2da"
# # AutoXGB
#
# * https://github.com/abhishekkrthakur/autoxgb

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 91106, "status": "ok", "timestamp": 1715655788636, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="XEh7kdym2hn_" outputId="2e4d58b9-5f12-4684-f17a-e41f3245ddf0"
# # !pip install autoxgb

# + colab={"base_uri": "https://localhost:8080/", "height": 748} executionInfo={"elapsed": 6300, "status": "error", "timestamp": 1715652353084, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="53f0bcfb-cec6-41c1-bbf6-ffdb12bbcbeb" outputId="746f66f6-912b-44d9-9c51-d58e56da849b"

# from autoxgb import AutoXGB


# # required parameters:
# train_filename = "train.csv"
# output = "autoxgb-output"

# # optional parameters
# test_filename = "test.csv"
# task = None
# idx = None
# targets = ["price"]
# features = None
# categorical_features = None
# use_gpu = False
# num_folds = 5
# seed = 42
# num_trials = 100
# time_limit = 360
# fast = True

# # Now its time to train the model!
# axgb = AutoXGB(
#     train_filename=train_filename,
#     output=output,
#     test_filename=test_filename,
#     task=task,
#     idx=idx,
#     targets=targets,
#     features=features,
#     categorical_features=categorical_features,
#     use_gpu=use_gpu,
#     num_folds=num_folds,
#     seed=seed,
#     num_trials=num_trials,
#     time_limit=time_limit,
#     fast=fast,
# )


# + id="6c9ae271-573d-4b6e-8839-bb5418376ab3"
# axgb.train()

# + id="f08a2450-972f-44a4-b84c-4fb8ece8daff"



# + [markdown] id="a424e78b-bd65-47b7-9355-36519e507b05"
# # Sample

# + id="90096eb3-e275-4970-a08d-4b2880c836bb" outputId="468e2879-e20f-4c5a-f4ae-27612c3314de"
sample_submission

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13615, "status": "ok", "timestamp": 1715655684042, "user": {"displayName": "Gonzalo Matheu", "userId": "07652169683581390753"}, "user_tz": 180} id="ccef5e48-a184-41f2-9c84-d6e268524832" outputId="28a0b5d4-99b7-4a20-a537-20ae5d4bffe0"
# ! pip install -U pip

# + id="Togv6ddlC_Ld"

# -

# # Notebooks with GPU
# * https://www.kaggle.com/ (30 hours)
# * https://colab.research.google.com/ (free-tier w GPU)
# * https://www.paperspace.com/pricing (free-tier no GPU)
# * https://saturncloud.io/plans/saturn_cloud_plans/ (free-tier w GPU?, Waitlist)
# * https://deepnote.com/ (free-tier no GPU)
# * https://studiolab.sagemaker.aws/login (waitlist, free-tier with GPU)
#

# ## MLFlow Proxy

# +


# Terminate open tunnels if exist
ngrok.kill()

# Setting the authtoken (optional)
# Get your authtoken from https://dashboard.ngrok.com/auth
NGROK_AUTH_TOKEN = "2VwE0ghPzEaPvlzIt9tdltGLFRZ_6nyxEMvYDUGjwfsW8A3xh"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open an HTTPs tunnel on port 5000 for http://localhost:5000
ngrok_tunnel = ngrok.connect(addr="5555", proto="http", bind_tls=True)
print("MLflow Tracking UI:", ngrok_tunnel.public_url)

# +
# run = wandb.init(project=WANDB_PROJECT)


# ds_art = wandb.Artifact(name="original", type="dataset", description="Original dataset")

# # Attach our processed data to the Artifact
# ds_art.add_file(train_path)
# ds_art.add_file(test_path)

# table = wandb.Table(dataframe=load_train().sample(1000))
# wandb.log({"dataset": table})


# run.log_artifact(ds_art)

# run.finish()
