import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def plot_clusters_with_ellipses(X_test, clusters):
    # Fit Gaussian Mixture Model (GMM) to the data
    gmm = GaussianMixture(n_components=len(np.unique(clusters)), random_state=42)
    gmm.fit(X_test)

    # Predict cluster labels
    cluster_labels = gmm.predict(X_test)

    # Plotting clusters with ellipses
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for cluster_label, color in zip(np.unique(cluster_labels), colors):
        # Plot data points that belong to the current cluster
        plt.scatter(X_test[cluster_labels == cluster_label][:, 0],
                    X_test[cluster_labels == cluster_label][:, 1],
                    s=50, color=color, label=f'Cluster {cluster_label}')

        # Plot ellipse representing the cluster
        plot_gaussian_ellipse(gmm, cluster_label, color)

    plt.title('Clusters with Ellipses (Gaussian Mixture Model)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def plot_gaussian_ellipse(gmm, cluster_label, color):
    # Extract cluster parameters
    covariances = gmm.covariances_[cluster_label]
    means = gmm.means_[cluster_label]

    # Compute ellipse parameters
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Calculate angle of rotation
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Ensure there are exactly two eigenvalues for width and height
    if len(eigenvalues) < 2:
        # Handle cases where there are not enough eigenvalues (shouldn't happen in GMM)
        width, height = 0.1, 0.1
    else:
        width, height = 2 * np.sqrt(eigenvalues[:2])  # Take only the first two eigenvalues

    # Plot ellipse
    ellipse = plt.matplotlib.patches.Ellipse(xy=means, width=width, height=height, angle=angle, color=color, alpha=0.3)
    ax = plt.gca()
    ax.add_patch(ellipse)


def preprocess(df, normalize=False):
    df = df.drop(columns=["unique_id", "has_missing_features"])

    ethnicity_columns = ["ethnic_background", "ethnic_background_o"]
    for col in ethnicity_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=float)], axis=1)
        df = df.drop(columns=[col])

    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

    y = df["match"]
    x = df.drop(columns=["match"])
    x = x.fillna(x.mean())

    if normalize:
        x = (x - x.mean()) / x.std()

    df = pd.concat([x, y], axis=1)
    return df


def load_and_split(path):
    datapre = pd.read_csv(path)
    datapre = datapre.dropna(subset=['match'])
    np.random.seed(42)
    datapre['random_feature'] = np.random.rand(len(datapre))
    features = [column for column in datapre.columns if column != 'match']
    target = 'match'
    X = datapre[features]
    y = datapre[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_xgb_classifier(X_train, y_train, X_test, y_test):
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model.named_steps['classifier']


def plot_feature_importance(model, X_train, y_train):
    importances = model.feature_importances_
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    feature_names = numerical_features + categorical_features

    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    onehot_encoder.fit(X_train[categorical_features])
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_features).tolist()

    importance_dict = {feature: 0 for feature in feature_names}

    for onehot_feature, importance in zip(onehot_feature_names, importances[len(numerical_features):]):
        original_feature = onehot_feature.split('_')[0]
        if original_feature in importance_dict:
            importance_dict[original_feature] += importance
        else:
            pass

    for feature, importance in zip(numerical_features, importances[:len(numerical_features)]):
        importance_dict[feature] += importance

    importance_df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])

    random_feature_importance = pd.DataFrame({'feature': ['random_feature'], 'importance': [importances[-1]]})
    importance_df = pd.concat([importance_df, random_feature_importance], ignore_index=True)

    intelligence_importance_value = importance_df[importance_df['feature'] == 'intelligence']['importance'].values[0]

    importance_df = importance_df[importance_df['importance'] > 0.005]
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    important_features = importance_df[importance_df['importance'] > 0.005]['feature'].tolist()

    plt.figure(figsize=(14, 10))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Ranking (Importance > random feature)')
    plt.xticks(rotation=45, ha='right')
    plt.gca().invert_yaxis()
    plt.show()

    return important_features


def prepare_train_test_data(path, important_features):
    data = pd.read_csv(path)
    important_features = [feature.split('_b\'')[0] for feature in important_features if '_b\'' in feature] + [feature
                                                                                                              for
                                                                                                              feature in
                                                                                                              important_features
                                                                                                              if
                                                                                                              '_b\'' not in feature]
    important_features_with_target = important_features + ['match']
    missing_features = [feature for feature in important_features_with_target if feature not in data.columns]
    filtered_data = preprocess(data)
    important_features.append('match')
    filtered_data = filtered_data[important_features].copy()

    labeled_data = filtered_data.dropna(subset=['match'])
    unlabeled_data = filtered_data[filtered_data['match'].isna()]
    labeled_train, labeled_test = train_test_split(labeled_data, test_size=0.5, random_state=42)

    labeled_train = pd.concat([labeled_train] * 7, ignore_index=True)
    labeled_test = pd.concat([labeled_test] * 7, ignore_index=True)

    training_data = pd.concat([labeled_train, unlabeled_data], ignore_index=True)
    testing_data = labeled_test

    X_train = training_data.drop(columns=['match'])
    y_train = training_data['match']
    X_test = testing_data.drop(columns=['match'])
    y_test = testing_data['match']

    return X_train, X_test, y_train, y_test


def train_spectral_model(spectral_model, X_train):
    train_clusters = spectral_model.fit(X_train)
    return spectral_model


def predict_spectral_model(spectral_model, X_test):
    predictions = spectral_model.fit_predict(X_test)
    return predictions


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split(r'path_to_csv\mixer_event_training.csv')
    xgb_model = train_xgb_classifier(X_train, y_train, X_test, y_test)
    important_features = plot_feature_importance(xgb_model, X_train, y_train)
    X_train, X_test, y_train, y_test = prepare_train_test_data(r'path_to_csv\mixer_event_training.csv',
                                                               important_features)
    spectral_model = SpectralClustering(n_clusters=2, random_state=42, affinity='nearest_neighbors')
    spectral_model = train_spectral_model(spectral_model, X_train)
    test_clusters = predict_spectral_model(spectral_model, X_test)
    accuracy = accuracy_score(y_test, test_clusters)
    f1 = f1_score(y_test, test_clusters)
    X_test_scaled = StandardScaler().fit_transform(X_test)  # Scale the data for GMM
    plot_clusters_with_ellipses(X_test_scaled, test_clusters)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-score: {f1:.2f}")
