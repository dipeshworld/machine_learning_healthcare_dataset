# 🏥 Health Risk Classification for Insurance Pricing

## 📌 Overview

This project builds a **Logistic Regression model** to classify individuals as:

* ✅ **Healthy (0)**
* ⚠️ **Unhealthy (1)**

for a global insurance company (**Anova Insurance**).

The goal is to support:

* Insurance **eligibility decisions**
* **Premium pricing optimization** based on health risk

---

## 🎯 Business Objective

Insurance providers need to assess applicant risk efficiently.
This model helps:

* Identify high-risk individuals
* Enable **risk-based premium pricing**
* Improve underwriting decisions

---

## 📂 Dataset Description

The dataset contains **10,000 records** with **20 features**, including:

### 🔢 Numerical Features

* Age
* BMI
* Blood Pressure
* Cholesterol
* Glucose Level
* Heart Rate
* Sleep Hours
* Exercise Hours
* Water Intake
* Stress Level

### 🔤 Categorical Features

* Smoking
* Alcohol
* Diet
* MentalHealth
* Physical Activity
* Medical History
* Allergies
* Diet_Type (Vegetarian, Non-Vegetarian, Vegan)
* Blood_Group (A, B, AB, O)

### 🎯 Target Variable

* **Target**

  * `0` → Healthy
  * `1` → Unhealthy

---

## ⚙️ Data Challenges & Handling

| Challenge           | Solution                                  |
| ------------------- | ----------------------------------------- |
| Negative Age values | Converted to absolute values              |
| Missing values      | Median (numeric), Mode (categorical)      |
| Mixed data types    | One-hot encoding for categorical features |
| Scale sensitivity   | StandardScaler applied                    |

---

## 🧠 Model Used

### 🔹 Logistic Regression

Why Logistic Regression?

* Interpretable and fast
* Works well for binary classification
* Suitable baseline for healthcare risk modeling

---

## 🏗️ Pipeline Architecture

```id="arch1"
Raw Data → Cleaning → Encoding → Scaling → Logistic Regression → Predictions
```

---

## 🚀 Installation

Clone the repository:

```bash id="cmd1"
git clone https://github.com/your-username/health-risk-classification.git
cd health-risk-classification
```

Install dependencies:

```bash id="cmd2"
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the model:

```bash id="cmd3"
python main.py
```

---

## 📊 Evaluation Metrics

* **Accuracy**
* **F1 Score**
* **Confusion Matrix**
* **Classification Report**

---

## 📈 Sample Output

```id="out1"
===== MODEL PERFORMANCE =====
Accuracy: 0.87
F1 Score: 0.85
```

---

## 📁 Project Structure

```id="struct1"
health-risk-classification/
│
├── data/
│   └── Healthcare_Dataset_Preprocessed.csv
│
├── models/
│   └── health_insurance_logistic_model.pkl
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔍 Key Features

* Robust preprocessing pipeline
* Handles missing & inconsistent data
* End-to-end ML pipeline using `Pipeline`
* Model persistence using `joblib`

---

## 💡 Future Improvements

* 📊 Feature importance & explainability (SHAP)
* ⚖️ Handle class imbalance (SMOTE / class weights)
* 🔍 Hyperparameter tuning (GridSearchCV)
* 🌐 Deployment via FastAPI / Streamlit
* 🧠 Upgrade to advanced models (XGBoost, LightGBM)

---

## 🧪 Business Impact

* Faster underwriting decisions
* Reduced risk exposure
* Data-driven premium pricing
* Improved customer segmentation

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or collaboration, feel free to reach out.

---
