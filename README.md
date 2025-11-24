# Phonesense – Smartphone Addiction Prediction

[Link text](https://huggingface.co/spaces/Peekaboo2803/phonesense)

**Phonesense** is a machine learning project that predicts the **risk level of smartphone addiction** (Low / Moderate / High) based on user behavior and survey data. It helps in understanding phone usage patterns and identifying potential addiction risks.

---

## 📊 Dataset

- **Source:** Kaggle 
- Features may include:
  - Screen time
  - App usage
  - Phone usage during study/work hours
  - Demographics (age, gender, etc.)
  - Lifestyle and mental health indicators

> ⚠️ **Note:** Make sure to download the dataset and place it in the `data/` folder.

---

## 🛠 Technology Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn  
- **Optional:** Jupyter Notebook for exploratory data analysis (EDA)  

---

## 📁 Project Structure

```bash
Phonesense/
│
├── data/
│   └── phonesense_dataset.csv
│
├── notebooks/
│   └── EDA_and_Preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── Logistic_regression.py
│   ├── Random_forest.py
│   ├── XGBoost.py
│   ├── evaluate_model.py
│   └── visualize_results.py
│
├── reports/
│   └── analysis_report.pdf
└── requirements.txt
```

---

## 🚀 How to Use

Clone the repository
```bash
git clone https://github.com/yourusername/phonesense.git
cd phonesense
```
Install dependencies
```bash
pip install -r requirements.txt
```

- **Run individual scripts:**
  - Preprocessing: ```bash python src/preprocessing.py```
  - Train models: ```bash python src/train_models.py```
  - Evaluate models: ```bash python src/evaluate.py```
  - Visualize results: ```bash python src/visualize.py```

---

## 🧩 Pipeline Overview

Load Data – Load CSV dataset using pandas.  
Preprocessing – Handle missing values, encode categorical variables, normalize/standardize features.  
Feature Engineering – Examples: average screen time, number of social media apps, usage during study hours.  
Modeling – Train and test models: Logistic Regression, Random Forest, XGBoost.  
Evaluation – Metrics: accuracy, precision, recall, F1-score; confusion matrix.  
Visualization – Feature distributions, correlation heatmaps, model performance plots.  

---

## 📈 Goal

Predict the risk level of smartphone addiction and gain insights into user behavior for digital wellness awareness.

---

## 📚 Credits

- Dataset provided by Kaggle
- Python libraries used: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

---

## 📬 Contact

Created by Kirtan Sukhadiya, Abhijeet Singh
