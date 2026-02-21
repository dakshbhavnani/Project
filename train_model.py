import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("dataSet.csv")

# Drop unnecessary columns if exist
drop_cols = [c for c in df.columns if "Unnamed" in c]
df = df.drop(columns=drop_cols, errors="ignore")

# Target
target = "Suggested Job Role"
y = df[target]
X = df.drop(columns=[target])

# Encode categorical columns
encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# Encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(X, y)

# Save everything
joblib.dump(model, "career_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(le_y, "label_encoder.pkl")

print("✅ Model trained and saved")