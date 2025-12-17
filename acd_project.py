import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering

RANDOM_STATE = 42


df = pd.read_excel("acd_data.xlsx")
print("Original Dataset:\n", df.head())


# 3. RENAME COLUMNS

rename_map = {
    "How often do you feel stressed due to academic workload? \n": "acd_workload",
    "How would you rate your overall stress level this semester? ": "stress_level",
    "How would you rate your current emotional balance (ability to manage feelings like sadness, anger, or frustration)?": "emotional_balance",
    "How frequently do you engage in physical activity (exercise, sports, active hobbies) for at least 30 minutes?": "physical_activity",
    "Which of the following stress factors most significantly affect your well-being? (Select all that apply)": "stress_factor",
    "Rate your agreement with the following statements regarding your academic life: [I feel motivated to study and attend classes.]": "motivation_level",
    "Rate your agreement with the following statements regarding your academic life: [I feel prepared for my exams and assignments.]": "preparation_level",
    "Rate your agreement with the following statements regarding your academic life: [I believe my academic performance reflects my full potential.]": "performance_satisfaction",
    "How many hours of sleep do you usually get per night? ": "sleep_hours",
    "  How satisfied are you with your current mental well-being?  ": "mental_wellbeing",
    "  How regular are you in attending your classes or lectures?  ": "attendance",
    "  What was your last semester’s CGPA or percentage?  ": "cgpa",
    "How would you rate your overall academic performance so far? ": "academic_performance"
}

df = df.rename(columns=rename_map)
df.drop(columns=["Timestamp", "Email address"] ,errors="ignore",inplace=True)

# 4. PREPROCESSING

categorical_cols = [
    "acd_workload",
    "stress_level",
    "emotional_balance",
    "physical_activity",
    "stress_factor",
    "motivation_level",
    "preparation_level",
    "performance_satisfaction",
    "mental_wellbeing",
    "attendance",
    "academic_performance",
    "sleep_hours"
]

numeric_cols = ["cgpa"]


#reassigning datatypes to columns as some of the categorical columns with values (1,2,3,4,5) are treated as numeric

print("datatypes of all columns:")
for col in df.columns:
    print(f"{col}  -->  ",df[col].dtype)

df[categorical_cols] = df[categorical_cols].astype("object")

# Convert numeric columns properly
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce") #convert invalid values like "ten" to NaN

print("\n\nDatatypes after making changes")
for col in df.columns:
    print(f"{col}  -->  ",df[col].dtype)


# Fix CGPA values > 10 as some people filled percantage instead of cgpa
print("first 10 rows in cgpa:",df["cgpa"].head(10))

df.loc[df["cgpa"] > 10, "cgpa"] = df.loc[df["cgpa"] > 10, "cgpa"] / 10

# Enforce valid CGPA range
df["cgpa"] = df["cgpa"].clip(0, 10)


# Fill missing values
print("\nCheck null values:",df.isnull().sum())

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = (
            df[col]
            .fillna(df[col].mode()[0])
            .infer_objects(copy=False) #no force conversion of datatype, Avoids unnecessary memory copy
        )
    else:
        df[col] = df[col].fillna(df[col].mean())


print("\nMissing Values After Cleaning:\n", df.isnull().sum())

sleep_map = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "more than 8 hours": 4
}

attendance_map = {
    "very irregular": 1,
    "somewhat irregular": 2,
    "Regular": 3,
    "very regular": 4
}

df["sleep_hours_ord"] = df["sleep_hours"].map(sleep_map)
df["attendance_ord"] = df["attendance"].map(attendance_map)


# 5. LABEL ENCODING:

label_encoders = {} #store encoders for future use
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]) #convert categorical columns to numeric (0,1,2...)
        label_encoders[col] = le 


# 2. GENERATE SYNTHETIC DATA (SDV)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

model = GaussianCopulaSynthesizer(metadata)
model.fit(df)

data = model.sample(num_rows=5000)
print("\nSynthetic Data Sample:\n", data.head())


# 6. EXPLORATORY DATA ANALYSIS (PLOTS)

print("\nprint first 10 records of dataset: \n",data.head(10))
print("\nDataset Shape: ", data.shape)
print("\nNumerical Summary:")
print(data.describe())

# Distributions
for col in numeric_cols:
    print(data[col])
    plt.figure(figsize=(6,4))
    sns.histplot(data[col])
    plt.title(f"Distribution of {col}")
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=data[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
    
# 7. REGRESSION (Predict cgpa)

# REGRESSION: Predict CGPA

X = data[["performance_satisfaction"]]
y = data["cgpa"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("\n--- Linear Regression (CGPA Prediction) ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


# Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], linestyle="--")
plt.title("Actual vs Predicted CGPA")
plt.xlabel("Actual CGPA")
plt.ylabel("Predicted CGPA")

plt.show()

#the predicited cgpa values cluster around mean, indicating that the linear regression model underfits the data.
#This suggests weak linear relationships between survey-based features and CGPA, causing predictions to regress toward the average.

#knn on stress level
X_knn = data.drop(columns=["stress_level"])
y_knn = data["stress_level"]

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=RANDOM_STATE, stratify=y_knn
)

scaler = StandardScaler()
X_train_k_scaled = scaler.fit_transform(X_train_k)
X_test_k_scaled = scaler.transform(X_test_k)

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="distance"
)
knn.fit(X_train_k_scaled, y_train_k)
y_pred_k = knn.predict(X_test_k_scaled)

print("\n--- KNN Classification (Stress Level) ---")
print("Accuracy :", accuracy_score(y_test_k, y_pred_k))
print("Precision:", precision_score(y_test_k, y_pred_k, average="weighted"))
print("Recall   :", recall_score(y_test_k, y_pred_k, average="weighted"))
print("F1 Score :", f1_score(y_test_k, y_pred_k, average="weighted"))

plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test_k, y_pred_k),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted Stress Level")
plt.ylabel("Actual Stress Level")
plt.title("KNN Confusion Matrix (Stress Level)")
plt.show()

# 8. CLASSIFICATION (Emotional Balance)

X_clf = data.drop(columns=["emotional_balance"])
y_clf = data["emotional_balance"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

clf = DecisionTreeClassifier(
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             class_weight="balanced",
                             random_state=RANDOM_STATE
)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print("\n--- Classification Metrics ---")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Precision:", precision_score(y_test_c, y_pred_c, average="weighted"))
print("Recall:", recall_score(y_test_c, y_pred_c, average="weighted"))
print("F1:", f1_score(y_test_c, y_pred_c, average="weighted"))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

#visual 
plt.figure(figsize = (22,10))
plot_tree(
    clf,
    feature_names = X_clf.columns,
    class_names = ["Very Low(0)", "Low(1)","Moderate(2)", "High(3)","Very High(4)"],
    filled = True,
    rounded = True,
    fontsize = 8)

plt.title("Decision Tree")
plt.show()


# 11. CLUSTERING

cluster_features = ["cgpa", "sleep_hours_ord", "attendance_ord"]

X_cluster = data[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ---- KMeans ----
kmeans = KMeans(n_clusters=3,init = 'k-means++',
        max_iter = 500, random_state=RANDOM_STATE)
data["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=X_scaled[:,0],
    y=X_scaled[:,1],
    hue=data["kmeans_cluster"],palette="Set2",
    s=80
)
plt.xlabel("CGPA (Standardized)")
plt.ylabel("Sleep Hours (Ordinal)")
plt.title("KMeans Clustering of Students Based on Academic & Lifestyle Factors")
plt.legend(title="Cluster")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
plt.show()

# Hierarchical
hc = AgglomerativeClustering(n_clusters=3)
data["hc_cluster"] = hc.fit_predict(X_scaled)

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=X_scaled[:,0],
    y=X_scaled[:,1],
    hue=data["hc_cluster"]
)
plt.xlabel("CGPA (Standardized)")
plt.ylabel("Sleep Hours (Ordinal)")
plt.title("Hierarchical Clustering")
plt.show()


