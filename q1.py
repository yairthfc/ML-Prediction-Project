# Input.
# A csv where each row holds information of one participant’s perspective of a meeting they have
# participated in – all columns except matche, which indicates that the participants indeed decided to
# collaborate.

# Output.
# A csv file named “match_predictions.csv”, including two columns: unique_id and match. An
# example of the output is provided in Table 1 and in y_match_example.csv.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt


def preprocess_data(df, normalize=False):
    # preprocess the data
    # one hot encoding for the ethnicity column as float
    df = df.drop(columns=["unique_id", "has_missing_features"])
    ethnicity_columns = ["ethnic_background", "ethnic_background_o"]
    for col in ethnicity_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=float)], axis=1)
        df = df.drop(columns=[col])
    # label encoding for the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    # fill missing values
    y = None
    y = df["match"]
    X = df.drop(columns=["match"], errors='ignore')
    X = X.fillna(X.mean())
    if normalize:
        X = (X - X.mean()) / X.std()
    df = pd.concat([X, y], axis=1)
    return df


# Hyperparameters #
ITERATIONS = 10
THRESHOLD = 0.6


# Define the self-training function
def self_training(X_labeled, y_labeled, X_unlabeled, iterations=ITERATIONS, threshold=THRESHOLD):
    model = xgb.XGBClassifier()
    model.fit(X_labeled, y_labeled)

    for i in range(iterations):
        probas = model.predict_proba(X_unlabeled)
        max_confidence = np.max(probas, axis=1)
        high_confidence_mask = max_confidence > threshold
        if not high_confidence_mask.any():
            break

        X_high_conf = X_unlabeled[high_confidence_mask]
        y_high_conf = np.argmax(probas[high_confidence_mask], axis=1)

        X_labeled = pd.concat([X_labeled, X_high_conf])
        y_labeled = np.concatenate([y_labeled, y_high_conf])

        X_unlabeled = X_unlabeled[~high_confidence_mask]

        model.fit(X_labeled, y_labeled)

    return model


if __name__ == '__main__':
    # Load data
    df_train = pd.read_csv(r'path_to_csv/Datasets/mixer_event_training.csv')
    df_match = pd.read_csv(r'path_to_csv/Datasets/X_match.csv')
    df_match['match'] = np.nan

    # seperate the data to train and test, and labeled and unlabeled
    df_labeled_data = preprocess_data(df_train)
    df_labeled_data = df_labeled_data.dropna(subset=['match'])
    df_unlabeled_data = preprocess_data(df_train)

    df_labeled_train, df_labeled_test = train_test_split(df_labeled_data, test_size=0.2, random_state=0)
    # df_train_train_preprocessed = preprocess_data(df_train_train, normalize=True, isTest=False)
    # df_train_test_preprocessed = preprocess_data(df_train_test, normalize=True, isTest=True)

    df_match_preprocessed = preprocess_data(df_match)

    X_labeled = df_labeled_train.drop('match', axis=1)
    y_labeled = df_labeled_train['match'].astype(int)  # Ensure label type is integer

    X_unlabeled = df_unlabeled_data.drop('match', axis=1)

    # Execute self-training
    final_model = self_training(X_labeled, y_labeled, X_unlabeled)

    # Predict on X_match.csv
    X_match = df_match_preprocessed.drop('match', axis=1, errors='ignore')  # Ensure 'match' column is not in features
    df_match['match'] = final_model.predict(X_match)

    # Save the final predictions, with columns 'unique_id' and 'match'
    df_final = df_match[['unique_id', 'match']]
    df_final.to_csv('match_predictions.csv', index=False)

    # Evaluate the model
    y_test = df_labeled_test['match']
    X_test = df_labeled_test.drop(columns=['match'])
    y_pred = final_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f'F1 score: {f1}')

    # evaluate the accuracy of the model
    accuracy = final_model.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')

    # plot the different hyperparameters possible values
    f1_scores = []
    accuracies = []
    for i in range(15, 20):
        for j in range(6, 11):
            current_model = self_training(X_labeled, y_labeled, X_unlabeled, i, j / 10)
            y_pred = current_model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
            accuracy = current_model.score(X_test, y_test)
            accuracies.append(accuracy)

    plt.plot(f1_scores)
    plt.plot(accuracies)
    # legend
    plt.legend(['F1 score', 'Accuracy'])
    # title
    plt.title('F1 score and Accuracy for different hyperparameters')
    # x-label
    plt.xlabel('Hyperparameters')
    # y-label
    plt.ylabel('Value')
    # show the plot
    plt.show()


