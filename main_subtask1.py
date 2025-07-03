import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

torch.manual_seed(69)


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
    training_file_path = 'mixer_event_training.csv'
    training_data = pd.read_csv(training_file_path)

    # Preprocess the training data
    training_data = preprocess_data(training_data)

    # Ensure 'match' is numeric and drop rows with NaN 'match'
    training_data['match'] = pd.to_numeric(training_data['match'], errors='coerce')
    labeled_training_data = training_data.dropna(subset=['match'])

    # Drop 'unique_id' and any non-numeric columns
    X_full = labeled_training_data.drop(['match'], axis=1).values
    y_full = labeled_training_data['match'].values

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_full = imputer.fit_transform(X_full)

    # Split the data into train, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=69)

    # Balance the training data using SMOTE
    smote = SMOTE(random_state=69)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)


    # Define a simpler network architecture
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return self.softmax(x)


    # Initialize the model, optimizer, and loss function
    input_dim = X_train.shape[1]
    hidden_dim = 64  # Simplified hidden dimension
    output_dim = 2  # Binary classification (match/no match)
    model = SimpleNet(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with labeled data
    num_epochs = 100
    batch_size = 64
    labeled_dataset = TensorDataset(X_train, y_train)
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

    # Lists to store metrics for visualization
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in labeled_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_losses.append(np.mean(batch_losses))

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_preds = torch.max(val_outputs, 1)
            val_loss = criterion(val_outputs, y_val).item()
            val_accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
            val_f1 = f1_score(y_val.numpy(), val_preds.numpy())

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)

    # evaluate X_match.csv
    match_data = pd.read_csv(r'path_to_csv\X_match.csv')
    pred_df = match_data[['unique_id']]
    match_data = preprocess_data(match_data, False, False)
    match_data = imputer.transform(match_data)
    match_data = torch.tensor(match_data, dtype=torch.float32)
    match_preds = model(match_data)
    match_preds = torch.argmax(match_preds, dim=1)
    pred_df['match'] = match_preds.numpy()
    pred_df.to_csv('match_predictions.csv', index=False)
