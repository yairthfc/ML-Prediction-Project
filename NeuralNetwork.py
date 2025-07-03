import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from task_2.code.hackathon_code.preproccess import preprocess

torch.manual_seed(69)

# Load the training dataset
training_file_path = '../../../Datasets/mixer_event_training.csv'
training_data = pd.read_csv(training_file_path)


# Preprocess the training data
training_data = preprocess(training_data)

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
X_train, X_temp, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.4, random_state=69)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)

# Check label distribution before applying SMOTE
print("Label distribution before SMOTE:", np.bincount(y_train.astype(int)))

# Balance the training data using SMOTE
smote = SMOTE(random_state=69)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check label distribution after applying SMOTE
print("Label distribution after SMOTE:", np.bincount(y_train.astype(int)))

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

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

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}, Validation F1-score: {val_f1:.4f}')

# Plotting training progress
plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Metrics
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.plot(val_f1_scores, label='Validation F1-score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics')
plt.legend()

plt.tight_layout()
plt.savefig('training_progress.png')
plt.show()

# Evaluate the model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1)
    test_accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
    test_f1 = f1_score(y_test.numpy(), test_preds.numpy())
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1-score: {test_f1:.4f}")