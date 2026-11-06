# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning classification algorithms to predict whether a patient has heart disease based on various medical and health-related features.

Three Machine Learning algorithms are trained and compared:

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)

The models are evaluated using Accuracy, Precision, Recall, and F1 Score to identify the best-performing algorithm.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze the heart disease dataset.
- Perform basic data preprocessing and exploratory data analysis.
- Split the dataset into training and testing sets.
- Train multiple Machine Learning classification models.
- Evaluate model performance using suitable metrics.
- Compare the performance of the models.
- Identify the best-performing Machine Learning algorithm.

---

## 📊 Dataset

The project uses the `heart.csv` dataset.

The dataset contains medical attributes of patients and a target variable indicating the presence or absence of heart disease.

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex of the patient |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Cholesterol level |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `target` | Heart disease target |

### Target

The `target` column is used as the output variable:

- `0` → No heart disease
- `1` → Heart disease

---

## 🤖 Machine Learning Algorithms

### 1. K-Nearest Neighbors (KNN)

KNN classifies a new data point based on the classes of its nearest neighboring data points.

Feature scaling using `StandardScaler` is applied before training the KNN model.

### 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

### 3. Support Vector Machine (SVM)

SVM finds an optimal boundary that separates different classes in the dataset.

Feature scaling is applied before training the SVM model.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Machine Learning Models
   ├── KNN
   ├── Random Forest
   └── SVM
   ↓
Model Evaluation
   ↓
Performance Comparison
   ↓
Best Model Selection
