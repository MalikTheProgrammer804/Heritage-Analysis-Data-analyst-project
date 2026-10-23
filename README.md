# Retail Sales Analytics Studio (Streamlit)

A responsive Streamlit web app for retail sales data cleaning, EDA, charts, business summaries, and CSV exports. The UI follows the provided reference: a light-grey left control panel and a clean white analytics area with headline metrics and tabs.

## 1. Requirements

- Python 3.10 or newer recommended
- A CSV, Excel (`.xlsx`/`.xls`) or Parquet dataset

## 2. Run locally (Windows)

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 3. How to use

1. Upload your retail CSV/Excel/Parquet file using the sidebar.
2. Choose cleaning controls. Settings rerun the app automatically.
3. Review the Overview, Data Cleaning and Data Quality tabs.
4. Explore Univariate, Bivariate, Multivariate and Retail Insights.
5. Download the cleaned CSV and reports from Export.

## 4. Cleaning notes

- Exact duplicate rows are removed automatically after basic text normalization.
- Category casing is standardized for low-cardinality text columns when enabled.
- Numeric-looking strings may be converted to numeric values.
- Date-like columns are parsed when enabled.
- Missing-value imputation is optional. Numeric fields use median, text fields use mode/`Unknown`, and dates are not imputed.
- Outliers are flagged with the 1.5×IQR rule by default. You can choose to remove rows flagged as outliers, but genuine high-value sales may be removed too.
- Generic invalid-value rules are only review flags. Confirm business rules (for example discount scale) for your dataset.

## 5. Deploy online

You can deploy this app on Streamlit Community Cloud:

1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io/ and connect GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Click Deploy.

Do not upload confidential customer data or secrets to a public repository.
