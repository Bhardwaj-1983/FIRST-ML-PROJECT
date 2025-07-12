#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)


df = pd.read_csv("../data/adult.csv")
df.head()


# In[17]:


df.info()
print("\nMissing values per column:")
print(df.isnull().sum())

df.describe(include='all').T


# In[18]:


df.columns = df.columns.str.strip()
df.head()


# In[19]:


df.replace("?", np.nan, inplace=True)

print(df.isnull().sum())

df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)


# In[20]:


df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

num_cols = df.select_dtypes(include=['int64', 'float64'])

corr = num_cols.corr()

print(corr['income'].sort_values(ascending=False))


# In[64]:


if 'fnlwgt' in df.columns:
    df = df.drop('fnlwgt', axis=1)

df.head()


# In[72]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']
numerical_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[73]:


from sklearn.model_selection import train_test_split

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# In[74]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)


# In[75]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)


# In[76]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# In[77]:


from sklearn.linear_model import LogisticRegression

lr_balanced = LogisticRegression(
    max_iter=3000,
    C=0.1,  # use same best param
    solver='liblinear',
    class_weight='balanced'
)

lr_balanced.fit(X_train_scaled, y_train)
y_pred_lr_bal = lr_balanced.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Balanced LR Accuracy:", accuracy_score(y_test, y_pred_lr_bal))
print(confusion_matrix(y_test, y_pred_lr_bal))
print(classification_report(y_test, y_pred_lr_bal))


# In[78]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeClassifier(random_state=42)

grid_dt = GridSearchCV(dt, dt_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train_scaled, y_train)

print("Best Decision Tree Params:", grid_dt.best_params_)
print("Best Decision Tree Score (CV):", grid_dt.best_score_)

best_dt_model = grid_dt.best_estimator_
y_pred_dt_tuned = best_dt_model.predict(X_test_scaled)

print("Tuned DT Accuracy:", accuracy_score(y_test, y_pred_dt_tuned))
print(confusion_matrix(y_test, y_pred_dt_tuned))
print(classification_report(y_test, y_pred_dt_tuned))


# In[80]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                        'relationship', 'race', 'sex', 'native.country']
numerical_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

rf = RandomForestClassifier(random_state=42)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

rf_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': ['balanced']
}

grid_rf = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best RF Params:", grid_rf.best_params_)
print("Best RF Score (CV):", grid_rf.best_score_)

best_rf_model = grid_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[81]:


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

xgb_model = XGBClassifier(
    eval_metric='logloss',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

xgb_pipeline.fit(X_train, y_train)
y_pred_xgb_pipeline = xgb_pipeline.predict(X_test)

print("XGBoost (Pipeline) Accuracy:", accuracy_score(y_test, y_pred_xgb_pipeline))
print(confusion_matrix(y_test, y_pred_xgb_pipeline))
print(classification_report(y_test, y_pred_xgb_pipeline))


# In[82]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'min_child_weight': [1, 3, 5],
    'scale_pos_weight': [1, 2, 3]  # to balance the class
}


# In[85]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

xgb_clf = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.5, 0.7, 1.0],
    'classifier__colsample_bytree': [0.5, 0.7, 1.0],
    'classifier__gamma': [0, 1, 5],
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_final = random_search.predict(X_test)
print("Tuned Pipeline XGBoost Accuracy:", accuracy_score(y_test, y_pred_final))
print(confusion_matrix(y_test, y_pred_final))
print(classification_report(y_test, y_pred_final))


# In[41]:


print("Best XGBoost Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

best_xgb = random_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(confusion_matrix(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))


# In[63]:


from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Use your best params (copied from RandomizedSearchCV results)
best_params = {
    'subsample': 1.0,
    'scale_pos_weight': 1,
    'n_estimators': 500,
    'min_child_weight': 5,
    'max_depth': 7,
    'learning_rate': 0.2,
    'gamma': 0.1,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'random_state': 42
}

# Build pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # From cell 26
    ('classifier', XGBClassifier(**best_params))
])

# Fit pipeline
xgb_pipeline.fit(X_train, y_train)

# Predict
y_pred_pipeline = xgb_pipeline.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Pipeline XGBoost Accuracy:", accuracy_score(y_test, y_pred_pipeline))
print(confusion_matrix(y_test, y_pred_pipeline))
print(classification_report(y_test, y_pred_pipeline))


# In[44]:


feature_names = X.columns  

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance

importances = xgb.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [feature_names[int(f[1:])] for f in importances.keys()],
    'Importance': list(importances.values())
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
plt.title("Top 10 Feature Importances - XGBoost")
plt.tight_layout()
plt.show()


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
accuracies = [0.853, 0.856, 0.855, 0.874]
f1_scores = [0.68, 0.66, 0.72, 0.72]
recalls = [0.62, 0.56, 0.73, 0.67]
precisions = [0.75, 0.80, 0.70, 0.79]

metrics_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies,
    "F1 Score": f1_scores,
    "Recall": recalls,
    "Precision": precisions
})
plt.figure(figsize=(12, 6))
metrics_df.set_index("Model").plot(kind="bar", figsize=(12, 6), edgecolor="black")

plt.title("Model Performance Comparison", fontsize=16)
plt.ylim(0.5, 1.0)
plt.ylabel("Score", fontsize=12)
plt.xticks(rotation=0, fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.legend(
    loc='upper left', 
    bbox_to_anchor=(1.02, 1), 
    borderaxespad=0,
    title='Metrics', 
    title_fontsize=12, 
    fontsize=11
)

plt.tight_layout()
plt.show()


# In[49]:


import shap

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train_scaled)

shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)


# In[50]:


import joblib

joblib.dump(best_xgb, "xgboost_best_model.pkl")


# In[51]:


df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)


# In[52]:


df.head()


# In[55]:


print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Categorical columns:", X[categorical_features].nunique())


# In[54]:


X = df[selected_features]
y = df[target]  


# In[53]:


numerical_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

selected_features = numerical_features + categorical_features

target = 'income'


# In[57]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# In[59]:


print(X_train.columns)


# In[ ]:


X = df_cleaned.drop('income', axis=1)
y = df_cleaned['income']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

