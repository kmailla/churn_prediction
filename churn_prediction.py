# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from enum import Enum

tf.get_logger().setLevel('ERROR')
rnd_state = 37


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


def check_important_features(X_train, X_test, y_train, y_test):
    try:
        # make y shape compatible with sklearn functions
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # create and train a baseline model
        model = RandomForestClassifier(n_estimators=20, random_state=rnd_state)
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        print()

        # permutation feature importance
        perm = PermutationImportance(model, cv=None, random_state=rnd_state, refit=False).fit(X_test, y_test)
        imp_scores = pd.concat([pd.DataFrame(X_train.columns), pd.DataFrame(perm.feature_importances_)], axis=1)
        imp_scores.columns = ['Feature', 'Score']
        print('Permutation importance scores:', imp_scores.nlargest(20, 'Score'), sep='\n', end='\n\n')
        # features are dropped if they do not influence the accuracy much when their values are shuffled
        features_to_drop = imp_scores.query('Score < 0.005')['Feature']
        print('Features to drop out: {}'.format(features_to_drop.values.tolist()))

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


def examine_churn_data():
    try:
        churn_data = get_data()
        print(churn_data.head())

        # find and drop rows with missing values
        churn_data = churn_data.replace(" ", np.NaN)
        old_size = len(churn_data.index)
        churn_data = churn_data.dropna()
        print('Number of dropped rows with missing values: {}'.format(old_size-len(churn_data.index)))

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

        # convert string values to categories
        churn_data = convert_columns_to_categorical(churn_data, categorical_features)
        # since the churn column will be the label column, remove it from the features
        categorical_features.remove('Churn')

        # check out if there is any correlation between the values
        correlation = churn_data.corr()
        plt.figure(figsize=(16, 16))
        sns.set(font_scale=0.6)
        sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
        plt.show()

        return churn_data, numeric_features, categorical_features

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def run_classification(churn_data, numeric_features, categorical_features, drop_out_features, upsample, estimator_type):
    try:
        # use upsampling (conditional)
        if upsample:
            churn_data_negatives = churn_data[churn_data['Churn'] == 0]
            churn_data_positives = churn_data[churn_data['Churn'] == 1]

            # upsample positives to have 4000 samples in the dataset (will make up now 46% of the whole)
            churn_data_positives = resample(churn_data_positives, replace=True, n_samples=4000, random_state=rnd_state)
            churn_data = pd.concat([churn_data_negatives, churn_data_positives])

        # separate features and labels column
        y = churn_data[['Churn']]
        X = churn_data.drop(columns=['Churn'])

        # separate training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=rnd_state)

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
            estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[1024, 512])
        elif estimator_type == EstimatorType.BoostedTrees:
            estimator = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, n_batches_per_layer=64)
        else:
            raise ValueError('Unknown estimator type')

        # train and evaluate
        estimator.train(train_input_fn, max_steps=2500)
        result = estimator.evaluate(eval_input_fn)

        # get true test labels and predicted labels into lists
        pred_dicts = list(estimator.predict(eval_input_fn))
        y_pred = [round(pred['probabilities'][1]) for pred in pred_dicts]
        # create confusion matrix on the test data
        conf_matrix = confusion_matrix(y_test.values.tolist(), y_pred)

        print('Results for the setting - estimator type={}, upsampling={}, drop out features={}:'
              .format(estimator_type, upsample, drop_out_features))
        print(result)

        return result, conf_matrix

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def run_rf_classification(churn_data, numeric_features, categorical_features, drop_out_features, upsample):
    try:
        # use upsampling (conditional)
        if upsample:
            churn_data_negatives = churn_data[churn_data['Churn'] == 0]
            churn_data_positives = churn_data[churn_data['Churn'] == 1]

            # upsample positives to have 4000 samples in the dataset (will make up now 46% of the whole)
            churn_data_positives = resample(churn_data_positives, replace=True, n_samples=4000, random_state=rnd_state)
            churn_data = pd.concat([churn_data_negatives, churn_data_positives])

        # separate features and labels column
        y = churn_data[['Churn']]
        X = churn_data.drop(columns=['Churn'])

        # separate training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=rnd_state)

        # compute which columns are making the validation scores worse in a simple model
        # and drop them out (conditional)
        if drop_out_features:
            features_to_drop = check_important_features(X_train, X_test, y_train, y_test)
            X_train = drop_unimportant_features(X_train, features_to_drop)
            X_test = drop_unimportant_features(X_test, features_to_drop)
            numeric_features = [n for n in numeric_features if n not in features_to_drop]
            categorical_features = [c for c in categorical_features if c not in features_to_drop]
            print('Features to stay: ' + str(categorical_features + numeric_features))

        # train and evaluate
        rf_model = RandomForestClassifier(n_estimators=20, random_state=rnd_state)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        result = dict()
        result['accuracy'] = rf_model.score(X_test, y_test)
        result['precision'] = precision_score(y_test, y_pred, average="macro")
        result['recall'] = recall_score(y_test, y_pred, average="macro")

        conf_matrix = confusion_matrix(y_test.values.tolist(), y_pred)

        print('Results for the setting - random forest:')
        print(result)

        return result, conf_matrix

    except Exception as e:
        print(e)
        print(traceback.format_exc())


def linear_variants_comparison():
    churn_data, numeric_features, categorical_features = examine_churn_data()
    baseline_result, baseline_conf_matrix = run_classification(churn_data, numeric_features, categorical_features,
                                                               drop_out_features=False, upsample=False,
                                                               estimator_type=EstimatorType.Linear)
    upsampled_result, upsampled_conf_matrix = run_classification(churn_data, numeric_features, categorical_features,
                                                                 drop_out_features=False, upsample=True,
                                                                 estimator_type=EstimatorType.Linear)

    dropcolumn_result, dropcolumn_conf_matrix = run_classification(churn_data, numeric_features, categorical_features,
                                                                   drop_out_features=False, upsample=True,
                                                                   estimator_type=EstimatorType.Linear)
    results = [baseline_result, upsampled_result, dropcolumn_result]
    conf_matrices = [baseline_conf_matrix, upsampled_conf_matrix, dropcolumn_conf_matrix]
    titles = ['baseline linear estimator', 'upsampled data linear estimator',
              'drop columns + upsampled linear estimator']

    # plot linear model confusion matrices with and without upsampling
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    axes = ax.flatten()
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(pd.DataFrame(cm), ax=axes[i], annot=True, fmt='g')
        axes[i].set_ylabel('True label')
        axes[i].set_xlabel('Predicted label\naccuracy={:0.4f}\nprecision={:0.4f}\nrecall={:0.4f}'
                           .format(results[i]['accuracy'], results[i]['precision'], results[i]['recall']))
        axes[i].set_title('Confusion matrix \n{}'.format(titles[i]))
    fig.tight_layout()
    plt.gcf().subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    plt.show()


def estimator_comparison():
    churn_data, numeric_features, categorical_features = examine_churn_data()
    linear_result, linear_conf_matrix = run_classification(churn_data, numeric_features, categorical_features,
                                                           drop_out_features=False, upsample=True,
                                                           estimator_type=EstimatorType.Linear)

    dnn_result, dnn_conf_matrix = run_classification(churn_data, numeric_features, categorical_features,
                                                     drop_out_features=False, upsample=True,
                                                     estimator_type=EstimatorType.DNN)

    boosted_trees_result, boosted_trees_conf_matrix = run_classification(churn_data, numeric_features,
                                                                         categorical_features,
                                                                         drop_out_features=False, upsample=True,
                                                                         estimator_type=EstimatorType.BoostedTrees)

    results = [linear_result, dnn_result, boosted_trees_result]
    conf_matrices = [linear_conf_matrix, dnn_conf_matrix, boosted_trees_conf_matrix]
    titles = ['linear estimator', 'dnn estimator',
              'boosted trees estimator']

    # plot linear model confusion matrices with and without upsampling
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    axes = ax.flatten()
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(pd.DataFrame(cm), ax=axes[i], annot=True, fmt='g')
        axes[i].set_ylabel('True label', fontsize=10)
        axes[i].set_xlabel('Predicted label\naccuracy={:0.4f}\nprecision={:0.4f}\nrecall={:0.4f}'
                           .format(results[i]['accuracy'], results[i]['precision'], results[i]['recall']), fontsize=10)
        axes[i].set_title('Confusion matrix \n{}'.format(titles[i]), fontdict={'fontsize': 14})
        axes[i].tick_params(labelsize=12)
    plt.gcf().subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    fig.tight_layout()
    plt.show()


def tree_algorithm_comparison():
    churn_data, numeric_features, categorical_features = examine_churn_data()
    boosted_trees_result, boosted_trees_conf_matrix = run_classification(churn_data, numeric_features,
                                                                         categorical_features,
                                                                         drop_out_features=False, upsample=True,
                                                                         estimator_type=EstimatorType.BoostedTrees)

    random_forest_result, random_forest_conf_matrix = run_rf_classification(churn_data, numeric_features,
                                                                            categorical_features,
                                                                            drop_out_features=False, upsample=True)

    results = [boosted_trees_result, random_forest_result]
    conf_matrices = [boosted_trees_conf_matrix, random_forest_conf_matrix]
    titles = ['boosted trees estimator', 'sklearn random forest']

    # plot linear model confusion matrices with and without upsampling
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    axes = ax.flatten()
    for i, cm in enumerate(conf_matrices):
        sns.heatmap(pd.DataFrame(cm), ax=axes[i], annot=True, fmt='g')
        axes[i].set_ylabel('True label', fontsize=10)
        axes[i].set_xlabel('Predicted label\naccuracy={:0.4f}\nprecision={:0.4f}\nrecall={:0.4f}'
                           .format(results[i]['accuracy'], results[i]['precision'], results[i]['recall']), fontsize=10)
        axes[i].set_title('Confusion matrix \n{}'.format(titles[i]), fontdict={'fontsize': 14})
        axes[i].tick_params(labelsize=12)
    plt.gcf().subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    fig.tight_layout()
    plt.show()


tree_algorithm_comparison()
