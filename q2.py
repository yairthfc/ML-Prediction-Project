import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


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
    y = df["match"]
    X = df.drop(columns=["match"], errors='ignore')
    X = X.fillna(X.mean())
    if normalize:
        X = (X - X.mean()) / X.std()
    df = pd.concat([X, y], axis=1)
    return df


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv(r'path_to_csv/Datasets/mixer_event_training.csv')
    data = preprocess_data(data, True)

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le  # Store the label encoder

    # Split the data into features and targets
    X = data.drop(['ambition_important', 'creativity_important'], axis=1)  # Drop other non-feature columns if necessary
    y = data[['ambition_important', 'creativity_important']]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the XGBoost regressor
    xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=8)

    # Wrap it in a MultiOutputRegressor
    multioutput_regressor = MultiOutputRegressor(xgb_regressor)

    # Train the model
    multioutput_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = multioutput_regressor.predict(X_test)

    # Calculate the mean squared error for each target
    mse_ambition = mean_squared_error(y_test['ambition_important'], y_pred[:, 0])
    mse_creativity = mean_squared_error(y_test['creativity_important'], y_pred[:, 1])

    print(f"Mean Squared Error for Ambition Important: {mse_ambition}")
    print(f"Mean Squared Error for Creativity Important: {mse_creativity}")
