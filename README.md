# 🧠 Adult Income Prediction Project  
### End-to-End Machine Learning + Streamlit Dashboard

🔗 **Live App**: https://aryamannain2005-adult-income-project.streamlit.app 
🔗 **GitHub Repo**: https://github.com/aryamannain2005/Adult-Income-Project

---

## 📌 Project Overview  
This project predicts whether a person earns **>50K or <=50K** annually using demographic features such as age, education, occupation, work hours, and more.

The goal is to demonstrate a full **end-to-end ML workflow**, including:

- Data cleaning & preprocessing  
- Exploratory data analysis  
- Feature engineering  
- Model building & comparison  
- Outlier handling  
- Custom prediction logic  
- Streamlit web deployment  

---

## 📊 Dataset  
**Source:** UCI Machine Learning Repository — *Adult Census Income Dataset*  
**Rows:** 48,842  
**Target Column:** `income` (<=50K or >50K)

---

## 🧹 Data Preprocessing Steps  
✔ Removed missing or incorrect entries  
✔ Removed whitespace & fixed inconsistent values  
✔ Handled `"?"` values  
✔ Outlier detection using IQR  
✔ Winsorization for extreme values  
✔ Label Encoding for categorical features  
✔ One-Hot Encoding (drop-first to avoid dummy trap)  
✔ Train-test split (80–20)

---

## 🔍 Exploratory Data Analysis  
- Age distribution  
- Income vs Education  
- Income vs Gender  
- Work hours distribution  
- Correlation matrix  
- Boxplots for outlier detection  

Visualizations are saved inside:  
📁 `visualizations/`

---

## 🤖 Machine Learning Models  
Models trained:

| Model | Accuracy |
|-------|----------|
| **Random Forest Classifier** | ~85% |
| **Logistic Regression (scaled)** | ~81% |

Random Forest performed best.

Models are saved inside:  
📁 `models/`

---

## 🎯 Custom Prediction Logic  
A custom scoring-based function predicts income based on:

- Age  
- Education  
- Weekly work hours  
- Gender  

Used for faster in-app predictions.

---

## 🌐 Streamlit Dashboard  
The app includes:

- Input panel for prediction  
- Automatically scaled visualizations  
- Model results summary  
- Outlier detection view  
- Dataset insight charts  

Main dashboard file:  
📄 `streamlit_dashboard.py`

---

## 📁 Project Structure
