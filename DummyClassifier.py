import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import plotly.graph_objects as go

# Load the training dataset
training_file_path = '../../../Datasets/mixer_event_training.csv'
training_data = pd.read_csv(training_file_path)

# Preprocess the training data
training_data = training_data.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

# Ensure 'match' is numeric and drop rows with NaN 'match'
training_data['match'] = pd.to_numeric(training_data['match'], errors='coerce')
labeled_training_data = training_data[training_data['match'].notnull()].dropna(subset=['match'])

# Split labeled data into training, validation, and test sets
X_train_full = labeled_training_data.drop('match', axis=1)
y_train_full = labeled_training_data['match'].astype(int)

# Assuming the dataset has a 'unique_id' column
unique_ids_train = X_train_full['unique_id']
X_train_full = X_train_full.drop('unique_id', axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the dummy classifier
dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
dummy_clf.fit(X_train, y_train)

# Evaluate the dummy classifier on the validation set
y_val_pred = dummy_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Evaluate the dummy classifier on the test set
y_test_pred = dummy_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Visualize the accuracy
fig = go.Figure(data=[
    go.Bar(name='Validation Accuracy', x=['Validation'], y=[val_accuracy]),
    go.Bar(name='Test Accuracy', x=['Test'], y=[test_accuracy])
])

fig.update_layout(
    title='Dummy Classifier Accuracy',
    xaxis_title='Dataset',
    yaxis_title='Accuracy',
    yaxis=dict(range=[0, 1]),
    barmode='group'
)

fig.write_image('accuracy.png')

# Visualize the F1-score
fig_f1 = go.Figure(data=[
    go.Bar(name='Validation F1-score', x=['Validation'], y=[val_f1]),
    go.Bar(name='Test F1-score', x=['Test'], y=[test_f1])
])

fig_f1.update_layout(
    title='Dummy Classifier F1-score',
    xaxis_title='Dataset',
    yaxis_title='F1-score',
    yaxis=dict(range=[0, 1]),
    barmode='group'
)

fig_f1.write_image('f1_score.png')

# Load the dataset for prediction
prediction_file_path = '../../../Datasets/X_match.csv'
prediction_data = pd.read_csv(prediction_file_path)

# Preprocess the prediction data
prediction_data = prediction_data.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))

# Ensure the dataset has a 'unique_id' column
unique_ids_unlabeled = prediction_data['unique_id']
X_unlabeled = prediction_data.drop('unique_id', axis=1)

# Use the dummy classifier to predict matches for the unlabeled data
y_unlabeled_pred = dummy_clf.predict(X_unlabeled)

# Create the match_predictions.csv file
match_predictions = pd.DataFrame({
    'unique_id': unique_ids_unlabeled,
    'match': y_unlabeled_pred
})

# notice that the slash is in the wrong direction if you are a windows user
output_file_path = r'path_to_csv/evaluation_scripts/predictions/match_predictions.csv'
match_predictions.to_csv(output_file_path, index=False)

