import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import seaborn as sns

# 1.Load Data
df = pd.read_csv("Fraud.csv")

# 2.Feature Engineering

# Log transform for skewed transaction amount
df['log_amount'] = np.log1p(df['amount'])

# Balance difference features
df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Transaction error indicators
df['orig_error'] = (df['balance_diff_orig'] != df['amount']).astype(int)
df['dest_error'] = (df['balance_diff_dest'] != df['amount']).astype(int)

# Encode transaction type (categorical feature)
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# 3. Define Features & Target
features = [
    'log_amount',
    'balance_diff_orig',
    'balance_diff_dest',
    'orig_error',
    'dest_error',
    'type_encoded'
]

X = df[features]
y = df['isFraud']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. Handle Class Imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 6. Model Training
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluation Metrics
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print("PR-AUC Score:", pr_auc)

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. Feature Importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()

feat_imp.plot(kind='barh')
plt.title("Feature Importance")
plt.show()


