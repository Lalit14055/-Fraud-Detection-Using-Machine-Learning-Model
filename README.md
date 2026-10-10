# 🚨 Fraud Detection in Financial Transactions
Machine Learning Project | Predicting Fraudulent Transactions

This project focuses on building a fraud detection model for a financial company using a large-scale dataset of 6.3M+ transactions. The objective is to identify fraudulent behavior, generate insights, and propose preventive strategies.

---

---

## 📊 1. Business Objective

Fraud causes massive financial and reputational damage.  
This project aims to:

- Build a fraud detection ML model  
- Clean & analyze transaction data  
- Identify key fraud indicators  
- Suggest fraud prevention strategies  
- Evaluate model and business impact  

Dataset Size: **6,362,620 rows · 10 columns**

---

## 📦 2. Dataset Description

| Column | Description |
|--------|-------------|
| step | Time step (1 = 1 hour) |
| type | Transaction type |
| amount | Transaction amount |
| nameOrig | Sender account |
| oldbalanceOrg | Sender balance before transaction |
| newbalanceOrig | Sender balance after transaction |
| nameDest | Receiver account |
| oldbalanceDest | Receiver balance before transaction |
| newbalanceDest | Receiver balance after transaction |
| isFraud | 1 = Fraud, 0 = Legit |
| isFlaggedFraud | Rule-based flag |

---

## 🧹 3. Data Cleaning

- Replaced missing merchant balances with 0  
- Handled outliers (kept because they represent fraud patterns)  
- Checked multicollinearity (drop redundant balance columns)  
- Cleaned categorical fields & extracted merchant indicators  

---

## 🛠️ 4. Feature Engineering

- `balanceDiff = oldbalanceOrg - newbalanceOrg`  
- `destinationDiff = oldbalanceDest - newbalanceDest`  
- `isMerchantOrig` (if nameOrig starts with “M”)  
- `isMerchantDest` (if nameDest starts with “M”)  
- One-Hot Encoding for transaction type  
- Removed high-cardinality ID fields: `nameOrig`, `nameDest`

---

## 🤖 5. Model Used

### ⭐ Chosen Model: **XGBoost Classifier**

Reasons:
- Works best for imbalanced datasets  
- Handles non-linear fraud patterns  
- High recall & precision  
- Fast training  

Pipeline Includes:
- Train-test split  
- SMOTE oversampling  
- XGBoost hyperparameter tuning  
- Evaluation on validation set  

---

## 📈 6. Model Performance

Metrics used:
- Accuracy  
- Precision  
- Recall (most important)  
- F1-score  
- ROC–AUC  
- Precision–Recall curve  

Target achieved:
- AUC: **0.95+**  
- Recall: **0.85+**  
- Precision: **0.90+**

---

## 🔍 7. Key Indicators of Fraud

Top fraud predictors:
- Transaction type = **TRANSFER / CASH-OUT**  
- Very high transaction amount  
- Sender balance drops to zero instantly  
- Receiver balance jumps abnormally  
- Receiver is a new/empty account  
- Non-merchant → merchant patterns  

These behaviors align with real financial fraud:  
fraudsters transfer all funds → send to mule accounts → quickly cash out.

---

## 🛡️ 8. Fraud Prevention Recommendations

**Technical Controls**
- Transaction velocity monitoring  
- Sudden balance-drop alert system  
- Device/IP fingerprinting  
- Behavioral pattern tracking  
- Two-factor authentication for high-value transfers  

**Process Controls**
- Cooling period for new accounts  
- Daily transfer limits  
- Auto-block suspicious mule accounts  

---

## 📉 9. Post-Implementation Evaluation

Measure success using:
- Reduction in total fraud cases  
- Reduction in false negatives  
- Increased detection rate  
- Lower financial losses  
- Fewer customer complaints  

A/B testing between old vs new systems can validate improvements.

---

## 👨‍💻 Author

**Lalit Patil**  
Machine Learning & Software Developer  
GitHub: https://github.com/Lalit14055  
LinkedIn: https://www.linkedin.com/in/lalit-patil-12858b263  

---

## 📜 License
This project is for educational and assessment purposes only.

