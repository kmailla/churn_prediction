# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.base import clone
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from enum import Enum


class EstimatorType(Enum):
    DNN = 1
    Linear = 2
    BoostedTrees = 3


def get_data():
    churn_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    return churn_data


def convert_columns_to_categorical(data, features):
    LE = LabelEncoder()
    for col in data:
        if col in features:
            data[col] = LE.fit_transform(data[col])
    return data


def normalize_columns(data, features):
    min_max_scaler = MinMaxScaler()
    num_values = data[features].values
    num_values_scaled = min_max_scaler.fit_transform(num_values)
    data[features] = pd.DataFrame(num_values_scaled, columns=features, index=data.index)

    return data


def drop_unimportant_features(data, features):
    data = data.drop(features, axis=1)
    return data


def drop_column_importance(model, X_train, X_test, y_train, y_test, random_state):
    # clone the model to have the same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    #model_clone.random_state = random_state
    # train and evaluate the benchmark model
    #model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_test, y_test)
    print('Benchmark score for drop out: ' + str(benchmark_score))
    # list for storing feature importances
    importances = []

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis=1), y_train)
        drop_col_score = model_clone.score(X_test.drop(col, axis=1), y_test)
        importances.append(benchmark_score - drop_col_score)

    # create dataframe for storing the scores
    df_columns = pd.DataFrame(X_train.columns)
    df_scores = pd.DataFrame(importances)
    importances_df = pd.concat([df_columns, df_scores], axis=1)
    importances_df.columns = ['Feature', 'Score']

    return importances_df


def check_important_features(X_train, X_test, y_train, y_test):
    try:
        # make y shape compatible with sklearn functions
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # create and train a baseline model
        rnd_state = 40
        model = RandomForestClassifier(n_estimators=20, random_state=rnd_state)
        model.fit(X_train, y_train)

        # 1. feature drop out
        drop_column_scores = drop_column_importance(model, X_train, X_test, y_train, y_test, random_state=rnd_state)
        print('Drop column scores compared to the benchmark model:',
              drop_column_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        # 2. permutation feature importance
        # perm = PermutationImportance(model, cv=None, refit=False, n_iter=50).fit(X_train, y_train)
        # imp_scores = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(perm.feature_importances_)], axis=1)
        # imp_scores.columns = ['Feature', 'Score']
        # print('Permutation importance scores:', imp_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        # average results
        #average_scores = pd.concat([pd.DataFrame(drop_column_scores['Feature']),
        #                            pd.DataFrame((drop_column_scores['Score'] + imp_scores['Score']) / 2)], axis=1)
        average_scores = pd.concat([pd.DataFrame(drop_column_scores['Feature']),
                                    pd.DataFrame((drop_column_scores['Score'] + drop_column_scores['Score']) / 2)], axis=1)
        print('Averaging scores:', average_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')

        features_to_drop = average_scores.query('Score < 0')['Feature']
        print(features_to_drop.values.tolist())

        return features_to_drop.values.tolist()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    # input function for the tensorflow estimator
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


def run_classification(drop_out_features, upsample, estimator_type):
    try:
        churn_data = get_data()
        print(churn_data.head())

        # find and drop rows with missing values
        churn_data = churn_data.replace(" ", np.NaN)
        old_size = len(churn_data.index)
        churn_data = churn_data.dropna()
        print('Number or dropped rows with missing values: ', str(old_size-len(churn_data.index)))

        # look at number of unique values by each column
        pd.set_option('display.max_columns', None)
        print(churn_data.agg(['size', 'nunique']))

        # we can delete customerID as it is not relevant for the prediction
        churn_data = churn_data.drop(columns=['customerID'])

        # separate numeric and categorical features
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        churn_data[numeric_features] = churn_data[numeric_features].apply(pd.to_numeric)
        # normalize numerical features to be between [0, 1]
        churn_data = normalize_columns(churn_data, numeric_features)

        # collect categorical features
        categorical_features = [col for col in churn_data if col not in numeric_features]

        # plot churn ratio on by each categorical feature
        fig, ax = plt.subplots(3, 6, figsize=(18, 6))
        fig.tight_layout()
        axes = ax.flatten()
        for i, c in enumerate(categorical_features):
            churn_data.groupby([c, 'Churn']).size().unstack().plot(ax=axes[i], kind='bar', stacked=True)
            axes[i].tick_params(labelrotation=0, labelsize=6)
            axes[i].xaxis.set_label_position('top')
            axes[i].legend(prop=dict(size=6), loc="upper right", title='Churn')
        plt.show()

        # convert string values to categorical indices
        churn_data = convert_columns_to_categorical(churn_data, categorical_features)

        # check out if there is any correlation between the values
        correlation = churn_data.corr()
        plt.figure(figsize=(16, 16))
        sns.set(font_scale=0.6)
        sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
        plt.show()

        # use upsampling (conditional)
        if upsample:
            churn_data_negatives = churn_data[churn_data['Churn'] == 0]
            churn_data_positives = churn_data[churn_data['Churn'] == 1]

            # upsample positives to have 4000 samples in the dataset (will make up now 46% of the whole)
            churn_data_positives = resample(churn_data_positives, replace=True, n_samples=4000, random_state=41)
            churn_data = pd.concat([churn_data_negatives, churn_data_positives])

        # separate features and labels column
        y = churn_data[['Churn']]
        X = churn_data.drop(columns=['Churn'])
        categorical_features.remove('Churn')

        # separate training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

        # compute which columns are making the validation scores worse in a simple model
        # and drop them out (conditional)
        if drop_out_features:
            features_to_drop = check_important_features(X_train, X_test, y_train, y_test)
            X_train = drop_unimportant_features(X_train, features_to_drop)
            X_test = drop_unimportant_features(X_test, features_to_drop)
            numeric_features = [n for n in numeric_features if n not in features_to_drop]
            categorical_features = [c for c in categorical_features if c not in features_to_drop]
            print('Features to stay: ' + str(categorical_features + numeric_features))

        # create estimator
        # https://www.tensorflow.org/tutorials/estimator/linear
        feature_columns = []

        # convert all the columns to tensorflow's feature column type
        # first the categorical ones,
        for feature_name in categorical_features:
            vocabulary = X_train[feature_name].unique()
            if estimator_type == EstimatorType.Linear:
                feature_columns.append(
                    tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
            elif estimator_type in [EstimatorType.DNN, EstimatorType.BoostedTrees]:
                categorical_feature = tf.feature_column.categorical_column_with_vocabulary_list(
                    feature_name, vocabulary)
                feature_columns.append(tf.feature_column.indicator_column(categorical_feature))
            else:
                raise ValueError('Unknown estimator type')
        # secondly the numeric ones
        for feature_name in numeric_features:
            feature_columns.append(tf.feature_column.numeric_column(feature_name))#, dtype=tf.float64))

        # create the input for the estimator
        train_input_fn = make_input_fn(X_train, y_train)
        eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)

        # init the given estimator type
        if estimator_type == EstimatorType.Linear:
            estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)
        elif estimator_type == EstimatorType.DNN:
            estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[1024, 512, 256])
        elif estimator_type == EstimatorType.BoostedTrees:
            estimator = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, n_batches_per_layer=32)
        else:
            raise ValueError('Unknown estimator type')

        # train and evaluate
        estimator.train(train_input_fn, max_steps=5000)
        result = estimator.evaluate(eval_input_fn)

        # get true test labels and predicted labels into lists
        pred_dicts = list(estimator.predict(eval_input_fn))
        y_pred = [round(pred['probabilities'][1]) for pred in pred_dicts]
        # create confusion matrix on the test data
        conf_matrix = confusion_matrix(y_test.values.tolist(), y_pred)
        sns.heatmap(pd.DataFrame(conf_matrix), annot=True, fmt='g')
        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label\n accuracy={:0.4f}'.format(result['accuracy']))
        plt.show()

        print(result)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


main(drop_out_features=True, upsample=True, estimator_type=EstimatorType.BoostedTrees)
