#pandas profiling
import pandas as pd
from ydata_profiling import ProfileReport
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Load the data
df = pd.read_csv('/home/itai/PycharmProjects/IML.Hackathon/Datasets/mixer_event_training.csv')
#drop rows with NaN 'match'
df = df.dropna(subset=['match'])

# Preprocess the data
def preprocess_data(df):
    # One hot encoding for the ethnicity column as float
    df = df.drop(columns=["unique_id", "has_missing_features"])
    ethnicity_columns = ["ethnic_background", "ethnic_background_o"]
    for col in ethnicity_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=float)], axis=1)
        df = df.drop(columns=[col])
    # Label encoding for the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    # Fill missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

df = preprocess_data(df)

# Split the data into features and target
X = df.drop(columns=["ambition_important"])
y = df["ambition_important"]

# Train a RandomForestClassifier to get feature importances
print(y)
model = RandomForestRegressor(random_state=0)
model.fit(X, y)
importances = model.feature_importances_

# Get the feature importances as a series
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Identify features with low importance (threshold can be adjusted)
threshold = 0.01  # Example threshold
important_features = feature_importances[feature_importances > threshold].index

# Drop the features that are not important
columns_amb = [col for col in df.columns if col not in important_features and col != "match"]


# Split the data into features and target
X = df.drop(columns=["creativity_important"])
y = df["creativity_important"]

# Train a RandomForestClassifier to get feature importances
model = RandomForestRegressor(random_state=0)
model.fit(X, y)
importances = model.feature_importances_

# Get the feature importances as a series
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Identify features with low importance (threshold can be adjusted)
threshold = 0.01  # Example threshold
important_features = feature_importances[feature_importances > threshold].index

# Drop the features that are not important
columns_cre=[col for col in df.columns if col not in important_features and col != "match"]

# intersection of the two lists
columns_to_drop = list(set(columns_amb).intersection(columns_cre))
df = df.drop(columns=columns_to_drop)

print(f"Important features: {important_features}")
print(f"Remaining columns: {df.columns}")

#pandas profiling
profile = ProfileReport(df)
profile.to_file("creativity_and_ambition_profiling.html")