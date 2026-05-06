# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 2. LOAD DATA
df = pd.read_csv("dataset/Healthcare_Dataset_Preprocessed.csv")

# 3. DATA CLEANING
# Fix negative age values
df["Age"] = df["Age"].apply(lambda x: abs(x))

# Separate columns
target = "Target"

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

num_cols.remove(target)

# Fill missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 4. FEATURE & TARGET
X = df.drop(columns=[target])
y = df[target]

# 5. PREPROCESSING PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

# 6. DECISION TREE MODEL
dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,              # controls overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. TRAIN MODEL
dt_model.fit(X_train, y_train)

# 9. PREDICTIONS
y_pred = dt_model.predict(X_test)

# 10. EVALUATION
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== DECISION TREE PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
