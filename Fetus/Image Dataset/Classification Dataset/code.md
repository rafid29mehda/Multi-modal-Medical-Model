Here's a detailed explanation of the code step by step, including the purpose of each section:

---

### **1. Importing Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
```
- **Purpose**: Import necessary libraries for:
  - Data manipulation (`numpy`, `pandas`)
  - Visualization (`matplotlib`, `seaborn`)
  - Machine learning and evaluation (`sklearn`)
- These libraries provide tools for data analysis, model training, and evaluation.

---

### **2. Loading and Understanding the Dataset**
```python
data = pd.read_csv("/content/fetal_health.csv")
print("Dataset Info:")
print(data.info())
print("\nFirst few rows:")
print(data.head())
print("\nStatistical Summary:")
print(data.describe().T)
```
- **Loading the dataset**: Reads the dataset into a `pandas` DataFrame.
- **Dataset info**: Displays information about columns, data types, and missing values.
- **First few rows**: Shows the first five rows of the data to understand its structure.
- **Statistical summary**: Describes each column's statistical properties (mean, median, etc.).

---

### **3. Visualizing the Target Distribution**
```python
colours = ["#f7b2b0", "#8f7198", "#003f5c"]
sns.countplot(data=data, x="fetal_health", palette=colours)
plt.title("Target Distribution")
plt.show()
```
- **Why**: Visualizes how many records fall into each fetal_health class (1 = Normal, 2 = Suspect, 3 = Pathological).
- **How**: Uses a bar chart to detect class imbalance, which is crucial for training a balanced model.

---

### **4. Correlation Matrix**
```python
plt.figure(figsize=(15, 15))
corrmat = data.corr()
cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)
sns.heatmap(corrmat, annot=True, cmap=cmap, center=0)
plt.title("Correlation Matrix")
plt.show()
```
- **Why**: Displays relationships between features using a heatmap. Strong correlations indicate redundancy or important relationships for model training.
- **How**: Calculates correlations and visualizes them with annotations.

---

### **5. Splitting Features and Target**
```python
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]
```
- **Why**: Separates the input features (`X`) from the target variable (`y`), which we aim to predict.

---

### **6. Standardizing the Features**
```python
s_scaler = preprocessing.StandardScaler()
X_df = s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=X.columns)
```
- **Why**: Machine learning models often perform better when numerical features are standardized to have a mean of 0 and a standard deviation of 1.
- **How**: `StandardScaler` scales all columns of `X`.

---

### **7. Visualizing Standardized Features**
```python
plt.figure(figsize=(20, 10))
shades = ["#f7b2b0", "#c98ea6", "#8f7198", "#50587f", "#003f5c"]
sns.boxenplot(data=X_df, palette=shades)
plt.xticks(rotation=90)
plt.title("Standardized Features")
plt.show()
```
- **Why**: Checks the spread of the standardized data to ensure no extreme outliers are present.
- **How**: Boxen plots display data distributions for each feature.

---

### **8. One-Hot Encoding the Target**
```python
y_onehot = pd.get_dummies(y).astype(int).values
```
- **Why**: Converts the target labels (`y`) into a format usable by machine learning models. For example:
  - Class 1 becomes `[1, 0, 0]`
  - Class 2 becomes `[0, 1, 0]`
  - Class 3 becomes `[0, 0, 1]`

---

### **9. Splitting Data for Training and Testing**
```python
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)
```
- **Why**: Splits the data into:
  - **Training set**: Used to train the model (70% of data).
  - **Testing set**: Used to evaluate the model (30% of data).
- **How**: `train_test_split` ensures no data leakage between training and testing.

---

### **10. Defining Pipelines for Models**
```python
pipeline_lr = Pipeline([('lr_classifier', LogisticRegression(random_state=42))])
pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier(random_state=42))])
pipeline_rf = Pipeline([('rf_classifier', RandomForestClassifier())])
pipeline_svc = Pipeline([('sv_classifier', SVC())])
```
- **Why**: Sets up machine learning pipelines for various models (Logistic Regression, Decision Tree, Random Forest, and SVC).
- **How**: Each pipeline standardizes the input and fits the specified model.

---

### **11. Training Models**
```python
for pipe in pipelines:
    pipe.fit(X_train, y_train)
```
- **Why**: Trains each model pipeline on the training data.

---

### **12. Cross-Validation**
```python
cv_results_accuracy = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train, y_train, cv=10)
    cv_results_accuracy.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))
```
- **Why**: Measures model performance using cross-validation to ensure consistency across different splits of training data.
- **How**: Averages accuracy scores over 10 splits.

---

### **13. Evaluating Random Forest**
```python
pred_rfc = pipeline_rf.predict(X_test)
accuracy = accuracy_score(y_test, pred_rfc)
print("Random Forest Test Accuracy:", accuracy)
print(classification_report(y_test, pred_rfc))
```
- **Why**: Evaluates the Random Forest pipeline on the test set using metrics like accuracy, precision, and recall.

---

### **14. Hyperparameter Tuning**
```python
parameters = {
    'n_estimators': [100, 150, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 12],
    'criterion': ['gini', 'entropy'],
    'n_jobs': [-1]
}
CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5)
CV_rfc.fit(X_train, y_train)
```
- **Why**: Finds the best parameters for the Random Forest model using `GridSearchCV`.

---

### **15. Final Model and Metrics**
```python
RF_model = RandomForestClassifier(**CV_rfc.best_params_)
RF_model.fit(X_train, y_train)
predictions = RF_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average="weighted")
precision = precision_score(y_test, predictions, average="weighted")
f1 = f1_score(y_test, predictions, average="weighted")
```
- **Why**: Trains the best-tuned model and evaluates its performance on the test set.

---

### **16. Confusion Matrix**
```python
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix / np.sum(cf_matrix), cmap=cmap, annot=True, annot_kws={'size': 15}, fmt=".2%")
plt.title("Confusion Matrix (Normalized)")
plt.show()
```
- **Why**: Visualizes the model’s ability to correctly classify each class using a heatmap.

---

### **17. Saving the Model**
```python
joblib.dump(RF_model, ctg_model_path)
```
- **Why**: Saves the trained model for future use. The model can be reloaded without retraining.

---

This pipeline thoroughly processes the data, evaluates models, tunes them for optimal performance, and saves the best-performing model for deployment!
