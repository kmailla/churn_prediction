# -*- coding: utf-8 -*-
""" WEIGHT CALCULATOR FOR THE PRODUCT LIST

We fetch the data from Redshift from which we make a decision tree model out and keep its accuracy score
(classifying problem based on if the given product click had a conversion or not).

We calculate three different scores:
1. Gini score:
At each decision of the tree, we check which feature is used in the decision and how much more accurate
it makes the classification, then sum these scores. The feature having the biggest score is the most important.

2. Feature drop importance:
We make new models, where each of them is leaving out one feature, then in the end compare their accuracy to
the original's accuracy score. If a model is a lot less accurate than the original,
it means it had an important feature removed.

3. Permutation importance:
Instead of removing the features, we make models where we shuffle one feature's values (make them noisy).
If a model is a lot less accurate, it means the shuffled feature is an important one.

In the end we normalize each score then average them for the final score that we scale up into integers.
"""

import numpy as np
import tensorflow as tf
import traceback
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.base import clone
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder


os.environ['TIME_INTERVAL'] = '60'


def get_data():
    churn_data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    return churn_data


def convert_to_categorical(data, features):
    LE = LabelEncoder()
    for col in data:
        if col in features:
            data[col] = LE.fit_transform(data[col])
    return data


def drop_unimportant_features(data, features):
    data = data.drop(features, axis=1)
    return data


def drop_col_feat_imp(model, X_train, X_test, y_train, y_test, random_state=40):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_test, y_test)
    print('benchmark score: '+str(benchmark_score))
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_test.drop(col, axis=1), y_test)
        print(drop_col_score)
        importances.append(benchmark_score - drop_col_score)

    df_columns = pd.DataFrame(X_train.columns)
    df_scores = pd.DataFrame(importances)

    # concat the two data frames
    importances_df = pd.concat([df_columns, df_scores], axis=1)
    importances_df.columns = ['Feature', 'Score']

    return importances_df


def check_important_features(X_train, X_test, y_train, y_test):
    try:
        # make y shape compatible with sklearn functions
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # create and train the classifier model we will use
        # model = RandomForestClassifier(n_estimators=100)
        model = MLPClassifier()
        model.fit(X_train, y_train)

        # 1. Feature drop out
        drop_column_scores = drop_col_feat_imp(model, X_train, X_test, y_train, y_test)
        # normalization
        #total = drop_column_scores['Score'].sum()
        #drop_column_scores['Score'] = drop_column_scores['Score'] / total

        print('Drop column scores:', drop_column_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        # 2. Permutation feature importance
        perm = PermutationImportance(model, cv=None, refit=False, n_iter=50).fit(X_train, y_train)
        imp_scores = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(perm.feature_importances_)], axis=1)
        imp_scores.columns = ['Feature', 'Score']
        # normalization
        #total = imp_scores['Score'].sum()
        #imp_scores['Score'] = imp_scores['Score'] / total

        print('Permutation importance scores:', imp_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        # Sum up results
        average_scores = pd.concat([pd.DataFrame(drop_column_scores['Feature']),
                                    pd.DataFrame((drop_column_scores['Score'] + imp_scores['Score']) / 2)], axis=1)
        print('Averaging scores:', average_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        features_to_drop = average_scores.query('Score < 0')['Feature']
        print(features_to_drop.values.tolist())

        return features_to_drop.values.tolist()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


def main(event, context):
    churn_data = get_data()

    # find and drop rows with missing values
    churn_data = churn_data.replace(" ", np.NaN)
    churn_data = churn_data.dropna()

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    churn_data[numerical_features] = churn_data[numerical_features].apply(pd.to_numeric)

    categorical_features = [col for col in churn_data if col not in numerical_features]
    churn_data = convert_to_categorical(churn_data, categorical_features)

    # separate features and classification column
    y = churn_data[['Churn']]
    X = churn_data.drop(columns=['Churn'])

    # separate training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    features_to_drop = check_important_features(X_train, X_test, y_train, y_test)
    X_train = drop_unimportant_features(X_train, features_to_drop)
    print(X_train)
    X_test = drop_unimportant_features(X_test, features_to_drop)
    print(X_test)
    numerical_features = [n for n in numerical_features if n not in features_to_drop]
    print(numerical_features)
    categorical_features = [c for c in categorical_features if c not in features_to_drop]
    print(categorical_features)

    # create estimator
    # https://www.tensorflow.org/tutorials/estimator/linear
    feature_columns = []
    categorical_features.remove('Churn')
    for feature_name in categorical_features:
        vocabulary = X_train[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in numerical_features:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float64))

    train_input_fn = make_input_fn(X_train, y_train)
    eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    print(result)


main(None, None)
