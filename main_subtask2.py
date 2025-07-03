import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def preprocess_data(df, normalize=False, include_match=True):
    # preprocess the data
    # one hot encoding for the ethnicity column as float
    df = df.replace("b'?'", "b'Other'")
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
    if include_match:
        y = df["match"]
        X = df.drop(columns=["match"], errors='ignore')
        X = X.fillna(X.mean())
        if normalize:
            X = (X - X.mean()) / X.std()
        df = pd.concat([X, y], axis=1)
    else:
        df = df.fillna(df.mean())
        if normalize:
            df = (df - df.mean()) / df.std()
    return df


if __name__ == '__main__':
    # Load the data
    best_n_estimators = 150
    best_learning_rate = 0.15
    best_max_depth = 8
    data = pd.read_csv(r'path_to_csv\mixer_event_training.csv').drop(
        columns=['match'])
    data = preprocess_data(data, False, False)
    X = data.drop(['ambition_important', 'creativity_important'], axis=1)  # Drop other non-feature columns if necessary
    y = data[['ambition_important', 'creativity_important']]
    data_to_predict = pd.read_csv('X_importance_ratings.csv')
    predictions_df = data_to_predict[['unique_id']]
    data_to_predict = preprocess_data(data_to_predict, False, False)

    xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=best_n_estimators,
                                 learning_rate=best_learning_rate,
                                 max_depth=best_max_depth)
    multioutput_regressor = MultiOutputRegressor(xgb_regressor)
    multioutput_regressor.fit(X, y)
    predictions = multioutput_regressor.predict(data_to_predict)
    # Save the predictions with the unique_id
    predictions_df['ambition_important'] = predictions[:, 0]
    predictions_df['creativity_important'] = predictions[:, 1]
    predictions_df.to_csv('importance_ratings_predictions.csv', index=False)
