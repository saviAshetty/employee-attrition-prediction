# Employee Attrition Prediction — Streamlit Deployment

## Files
- `app.py` — Streamlit web application
- `train_model.py` — trains the selected Gradient Boosting model and saves `attrition_model.joblib`
- `Employee_Attrition.csv` — dataset used by the project
- `requirements.txt` — Python dependencies

## Run on your computer
1. Install Python 3.10 or 3.11.
2. Extract this ZIP and open the extracted folder in VS Code.
3. Open a terminal in that folder and run:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Train and save the model:

```bash
python train_model.py
```

6. Start the app:

```bash
streamlit run app.py
```

The browser should open the local application.

## Deploy online with Streamlit Community Cloud
1. Create a GitHub account (if you do not already have one).
2. Create a new public GitHub repository, for example `employee-attrition-prediction`.
3. Upload `app.py`, `train_model.py`, `requirements.txt`, and `Employee_Attrition.csv` to the repository.
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select your repository, branch `main`, and main file path `app.py`.
6. Click **Deploy**. The app trains the model on first startup if the saved model file is not present.

## Important
- The app retrains Gradient Boosting using the same preprocessing and main model settings in your notebook. It does not load a model already saved by the notebook.
- Your report's test metrics are from the original notebook run. Retraining can produce the same or slightly different results depending on package versions.
- Employee data is sensitive. Do not upload real identifiable employee records to a public repository. Use anonymized/demo data and add privacy controls before real organizational use.
