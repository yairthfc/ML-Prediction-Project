from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def preprocess(df, normalize=False):
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
    y = df["match"]
    x = df.drop(columns=["match"])
    x = x.fillna(x.mean())
    if normalize:
        x = (x - x.mean()) / x.std()
    df = pd.concat([x, y], axis=1)
    return df


if __name__ == '__main__':
    df = pd.read_csv('Datasets/mixer_event_training.csv')
    new_df = preprocess(df)
    new_df.to_csv('mixer_event_training_preprocessed.csv', index=False)
    new_df_normalized = preprocess(df, normalize=True)
    new_df_normalized.to_csv('mixer_event_training_preprocessed_normalized.csv', index=False)