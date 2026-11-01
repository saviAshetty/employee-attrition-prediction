import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

DATA_PATH = "Employee_Attrition.csv"
MODEL_PATH = "attrition_model.joblib"

def train_and_save():
    df = pd.read_csv(DATA_PATH)
    drop_cols = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train_p, y_train)

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_bal, y_bal)

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_columns": X.columns.tolist(),
        "metrics_note": "Retrained using the project notebook's preprocessing and Gradient Boosting settings."
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    return bundle

if __name__ == "__main__":
    train_and_save()
