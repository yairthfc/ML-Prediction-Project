# Hackathon ML Prediction Project – Match & Importance Ratings

This repository contains our solution for a hackathon machine learning task aimed at **predicting compatibility ("match") and personality importance ratings** between participants, based on anonymized profile data.

We built, trained, and evaluated models for two subtasks:
1. Predicting whether two users are a "match" (`X_match.csv`)
2. Predicting how much users value "ambition" and "creativity" in a match (`X_importance_ratings.csv`)

---

## 🧠 Project Overview

Our solution includes:
- Data preprocessing and profiling using `pandas-profiling`
- Multiple model experiments: Decision Trees, SVMs, Neural Networks, Dummy baselines
- Final selected models for each subtask with output predictions
- PDF reports summarizing methodology, results, and suggestions

---

## 📁 File Structure & Descriptions

| Path | Description |
|------|-------------|
| `README.txt` | Text-based description of project files. |
| `project.pdf` | Project summary, problem definition, modeling approach, and evaluation results. |
| `task_2/predictions/match_predictions.csv` | Predictions for subtask 1: matching likelihoods. |
| `task_2/predictions/importance_ratings_predictions.csv` | Predictions for subtask 2: importance ratings (ambition, creativity). |
| `task_2/answers/conclusions_and_suggestions.pdf` | Final conclusions and suggestions based on analysis. |
| `task_2/code/requirements.txt` | Required Python libraries for running the project code. |

---

## 🧪 Code Breakdown

| File | Purpose |
|------|---------|
| `main_subtask1.py` | Loads data, preprocesses, trains a model, and predicts `match` labels. |
| `main_subtask2.py` | Same as above, but for `ambition_important` and `creativity_important`. |
| `preprocess.py` | Data cleaning, encoding, scaling utilities. |
| `q1.py` | Experimental semi-supervised learning approach for subtask 1. |
| `q2.py` | Older version of subtask 2 solution using multi-output classifier. |
| `svm_tree.py` | Alternate SVM and Decision Tree model versions (unused). |
| `DummyClassifier.py` | Baseline model implementation. |
| `NeuralNetwork.py` | Neural Network classifier (early version of subtask 1 model). |
| `testing_.py` | Experimental clustering-based solution for subtask 1. |
| `pandas_profiling.py` | Generates full profile report for the match dataset. |
| `amb_creat_profiling.py` | Profile report for the importance ratings dataset. |

---

## 🛠️ How to Run

1. Install dependencies:
   ```bash
   pip install -r task_2/code/requirements.txt

---

## 📈 Project Results

Detailed performance metrics and conclusions can be found in:
- `project.pdf`
- `task_2/answers/conclusions_and_suggestions.pdf`

We explored multiple modeling strategies before finalizing our submission, balancing accuracy and interpretability.

---

## 📬 Contact

For questions or collaboration, feel free to reach out:

- 📧 Email: yairthfc@gmail.com  
- 🔗 LinkedIn: [linkedin.com/in/yairmahfud](https://www.linkedin.com/in/yairmahfud)
