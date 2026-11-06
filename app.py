import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("heart.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. DATA INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumns after cleaning:")
print(df.columns)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\nNumber of duplicate rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Shape after removing duplicates:")
print(df.shape)


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\nHeart Disease Values:")
print(df["target"].value_counts())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# Heart Disease Distribution

sns.countplot(data=df, x="target")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")
plt.show()


# Age Distribution

sns.histplot(data=df, x="age", kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()


# Age vs Heart Disease

sns.boxplot(
    data=df,
    x="target",
    y="age"
)

plt.title("Age vs Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Age")
plt.show()


# Sex vs Heart Disease

sns.countplot(
    data=df,
    x="sex",
    hue="target"
)

plt.title("Sex vs Heart Disease")
plt.xlabel("Sex")
plt.ylabel("Number of Patients")
plt.show()


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\nInput Features:")
print(X.columns)

print("\nNumber of Features:", X.shape[1])

print("\nTarget:")
print(y.name)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 9. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 10. K-NEAREST NEIGHBORS
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)

print("\n================ K-NEAREST NEIGHBORS ================")

print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("\n================ RANDOM FOREST ================")

print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)


# ============================================================
# 12. SUPPORT VECTOR MACHINE
# ============================================================

svm = SVC()

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print("\n================ SUPPORT VECTOR MACHINE ================")

print("Accuracy :", svm_accuracy)
print("Precision:", svm_precision)
print("Recall   :", svm_recall)
print("F1 Score :", svm_f1)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "K-Nearest Neighbors",
        "Random Forest",
        "Support Vector Machine"
    ],

    "Accuracy": [
        knn_accuracy,
        rf_accuracy,
        svm_accuracy
    ],

    "Precision": [
        knn_precision,
        rf_precision,
        svm_precision
    ],

    "Recall": [
        knn_recall,
        rf_recall,
        svm_recall
    ],

    "F1 Score": [
        knn_f1,
        rf_f1,
        svm_f1
    ]
})

print("\n================ MODEL COMPARISON ================")

print(results)


# ============================================================
# 14. ACCURACY COMPARISON
# ============================================================

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()


# ============================================================
# 15. ALL METRICS COMPARISON
# ============================================================

metrics = results.set_index("Model")

metrics.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# ============================================================
# 16. KNN CONFUSION MATRIX
# ============================================================

cm_knn = confusion_matrix(y_test, y_pred_knn)

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d"
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 17. RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 18. SVM CONFUSION MATRIX
# ============================================================

cm_svm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d"
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 19. CLASSIFICATION REPORTS
# ============================================================

print("\n================ KNN CLASSIFICATION REPORT ================")

print(classification_report(
    y_test,
    y_pred_knn
))


print("\n================ RANDOM FOREST CLASSIFICATION REPORT ================")

print(classification_report(
    y_test,
    y_pred_rf
))


print("\n================ SVM CLASSIFICATION REPORT ================")

print(classification_report(
    y_test,
    y_pred_svm
))


# ============================================================
# 20. FIND BEST MODEL
# ============================================================

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\n================ BEST MODEL ================")

print("Best Model:", best_model["Model"])
print("Accuracy :", best_model["Accuracy"])
print("Precision:", best_model["Precision"])
print("Recall   :", best_model["Recall"])
print("F1 Score :", best_model["F1 Score"])


# ============================================================
# 21. FINAL CONCLUSION
# ============================================================

print("\n================ CONCLUSION ================")

print(
    "The models were used to predict whether a patient "
    "has heart disease."
)

print(
    "K-Nearest Neighbors, Random Forest and Support Vector "
    "Machine were compared using Accuracy, Precision, "
    "Recall and F1 Score."
)

print(
    "The model with the highest accuracy is:",
    best_model["Model"]
)