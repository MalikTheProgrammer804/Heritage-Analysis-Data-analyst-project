import re
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Retail Sales | Data Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS: light control panel + clean dashboard, inspired by the reference ----------
st.markdown("""
<style>
  .stApp { background: #ffffff; color: #303342; }
  [data-testid="stHeader"] { background: rgba(255,255,255,0.9); }
  [data-testid="stSidebar"] { background: #f0f2f6; border-right: 1px solid #e4e7ee; }
  [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
  .brand-mark {
    width: 34px; height: 34px; border-radius: 10px; background: #24263d;
    color: white; display:flex; align-items:center; justify-content:center;
    font-size: 18px; margin-bottom: 28px;
  }
  .hero-title { font-size: 2.35rem; font-weight: 780; line-height: 1.18; color:#303342; margin-bottom: .6rem; }
  .hero-copy { font-size: 1rem; color:#555b6e; line-height:1.7; max-width: 850px; }
  .section-heading { color:#303342; font-size:1.3rem; font-weight:750; margin:1.3rem 0 .65rem 0; }
  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e9ebf1; border-radius: 12px;
    padding: 13px 16px; box-shadow: 0 2px 8px rgba(24,32,56,.035);
  }
  div[data-testid="stMetricLabel"] { color:#5d6273; }
  div[data-testid="stMetricValue"] { color:#303342; }
  .sidebar-note { color:#555b6e; font-size:.86rem; line-height:1.65; }
  .status-pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#e8f7ee; color:#237447; font-size:.82rem; }
  div[data-testid="stDataFrame"] { border:1px solid #e9ebf1; border-radius:10px; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom:1px solid #e9ebf1; }
  .stTabs [data-baseweb="tab"] { padding:10px 14px; }
  @media (max-width: 800px) {
    .hero-title { font-size:1.8rem; }
  }
</style>
""", unsafe_allow_html=True)


PLACEHOLDERS = {"na", "n/a", "none", "null", "unknown", "not available",
                "not applicable", "-", "--", "?"}


def clean_names(columns):
    result, seen = [], {}
    for col in columns:
        name = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_") or "column"
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def read_uploaded_file(upload):
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(upload, low_memory=False)
        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin1", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(upload)
    if suffix == ".parquet":
        return pd.read_parquet(upload)
    raise ValueError("Upload a CSV, Excel, or Parquet file.")


def convert_numeric_text(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        original = frame[col]
        cleaned = (original.astype("string")
                   .str.replace(r"[$£€,]", "", regex=True)
                   .str.replace("%", "", regex=False).str.strip())
        numeric = pd.to_numeric(cleaned, errors="coerce")
        valid = int(original.notna().sum())
        if valid and numeric.notna().sum() / valid >= .90:
            frame[col] = numeric
    return frame


def quality_report(frame):
    return pd.DataFrame({
        "Column": frame.columns,
        "Data type": [str(frame[c].dtype) for c in frame.columns],
        "Missing values": [int(frame[c].isna().sum()) for c in frame.columns],
        "Missing (%)": [round(float(frame[c].isna().mean() * 100), 2) for c in frame.columns],
        "Unique values": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
    }).sort_values(["Missing (%)", "Unique values"], ascending=[False, True])


def outlier_report(frame, columns):
    rows = []
    for col in columns:
        s = frame[col].dropna()
        if s.nunique() < 4:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        low, high = (q1 - 1.5 * iqr, q3 + 1.5 * iqr) if iqr else (q1, q3)
        count = int(((frame[col] < low) | (frame[col] > high)).sum())
        rows.append({"Column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                     "Lower bound": low, "Upper bound": high,
                     "Outlier count": count,
                     "Outlier (%)": round(count / max(len(frame), 1) * 100, 2)})
    return pd.DataFrame(rows).sort_values("Outlier count", ascending=False) if rows else pd.DataFrame()


def find_col(columns, words):
    return next((c for c in columns if any(w in c.lower() for w in words)), None)


def style_fig(fig, height=420):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=60, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#45495a"),
        title_font=dict(size=17, color="#303342"),
    )
    return fig


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown('<div class="brand-mark">✿</div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Control Panel")
    st.markdown(
        '<p class="sidebar-note"><b>Notice:</b> Adjusting any widget below reruns the app with updated values. '
        'Upload your sales dataset to start the report.</p>',
        unsafe_allow_html=True,
    )
    upload = st.file_uploader("Upload sales dataset", type=["csv", "xlsx", "xls", "parquet"])
    st.markdown("---")
    st.markdown("#### Cleaning controls")
    standardize = st.checkbox("Standardize category text", True)
    numeric_conversion = st.checkbox("Convert numeric-looking text", True)
    parse_dates = st.checkbox("Parse date columns", True)
    fill_missing = st.checkbox("Impute missing values", False)
    outlier_action = st.selectbox("Outlier handling", ["Flag only (recommended)", "Remove flagged rows"])
    st.markdown("---")
    st.markdown("#### Chart controls")
    max_categories = st.slider("Categories to show", 5, 30, 12)
    chart_sample = st.slider("Max rows for scatter/pair charts", 500, 5000, 2000, step=500)
    show_raw = st.checkbox("Show original raw data", False)
    st.markdown("---")
    st.caption("Formats: CSV · XLSX · XLS · Parquet")

# ---------- Landing state ----------
if upload is None:
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="hero-title">📊 Retail Sales Analytics<br>Cleaning, Insights & Reports</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-copy">An interactive data analyst workspace to clean retail data, '
            'review data quality, explore distributions, compare performance, and export a report.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.info("Use the **Control Panel** on the left to upload your dataset.")
        st.markdown("#### Included in the report")
        st.markdown(
            "- Duplicate detection and removal\n"
            "- Missing-value and invalid-value review\n"
            "- Category standardization and data-type checks\n"
            "- IQR outlier detection\n"
            "- Univariate, bivariate and multivariate charts\n"
            "- Sales/profit summaries and downloadable cleaned data"
        )
    with right:
        st.markdown("### Workflow")
        st.markdown("**01 — Upload**  \nAdd your CSV or Excel file.")
        st.markdown("**02 — Clean**  \nReview duplicates, missing values and suspicious records.")
        st.markdown("**03 — Analyze**  \nUse interactive charts and business summaries.")
        st.markdown("**04 — Export**  \nDownload cleaned data and analysis reports.")
        st.warning("Extreme values are flagged by default rather than deleted. Some outliers may be genuine large orders.")
    st.stop()

# ---------- Load and clean ----------
try:
    raw = read_uploaded_file(upload)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded file has no records.")
    st.stop()

original_rows, original_cols = raw.shape
raw_duplicate_count = int(raw.duplicated().sum())
raw_missing_count = int(raw.isna().sum().sum())
df = raw.copy()
df.columns = clean_names(df.columns)

# Normalize text and common missing placeholders
for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    df[col] = df[col].astype("string").str.strip()
    lowered = df[col].str.lower()
    df.loc[lowered.isin(PLACEHOLDERS), col] = pd.NA
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

if standardize:
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if 1 < df[col].nunique(dropna=True) <= 50:
            df[col] = (df[col].astype("string").str.replace(r"\s+", " ", regex=True)
                       .str.strip().str.title())

if numeric_conversion:
    df = convert_numeric_text(df)

date_candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "timestamp", "datetime"])] if parse_dates else []
date_failures = []
for col in date_candidates:
    before = int(df[col].notna().sum())
    converted = pd.to_datetime(df[col], errors="coerce")
    date_failures.append({"Column": col, "Unparseable values": max(0, before - int(converted.notna().sum()))})
    df[col] = converted

duplicate_rows = df.loc[df.duplicated(keep=False)].copy()
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
rows_after_dedup = len(df)

imputation_log = []
if fill_missing:
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if not missing:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            action = "Left missing; dates not imputed"
        elif pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            if pd.notna(value):
                df[col] = df[col].fillna(value)
                action = f"Filled with median ({value})"
            else:
                action = "No median available; left missing"
        else:
            modes = df[col].mode(dropna=True)
            if len(modes):
                value = modes.iloc[0]
                df[col] = df[col].fillna(value)
                action = f"Filled with mode ({value})"
            else:
                df[col] = df[col].fillna("Unknown")
                action = "Filled with Unknown"
        imputation_log.append({"Column": col, "Missing before": missing, "Action": action})

# Generic review flags; business meaning should be confirmed before deleting records
invalid_rows = []
for col in df.select_dtypes(include=np.number).columns:
    name = col.lower()
    checks = []
    if "quantity" in name or re.search(r"(^|_)qty($|_)", name):
        checks.append(("Quantity <= 0", df[col].notna() & (df[col] <= 0)))
    if "price" in name and "discount" not in name:
        checks.append(("Price < 0", df[col].notna() & (df[col] < 0)))
    if "discount" in name:
        checks.append(("Discount outside 0–100", df[col].notna() & ~df[col].between(0, 100)))
    if "age" in name:
        checks.append(("Age outside 0–120", df[col].notna() & ~df[col].between(0, 120)))
    for rule, mask in checks:
        if mask.any():
            invalid_rows.append({"Column": col, "Rule": rule, "Flagged rows": int(mask.sum())})

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
outliers = outlier_report(df, numeric_cols)
outlier_mask = pd.Series(False, index=df.index)
if not outliers.empty:
    for _, row in outliers.iterrows():
        c = row["Column"]
        outlier_mask |= df[c].notna() & ((df[c] < row["Lower bound"]) | (df[c] > row["Upper bound"]))
outlier_records = df.loc[outlier_mask].copy()
if outlier_action == "Remove flagged rows" and not outliers.empty:
    df = df.loc[~outlier_mask].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

quality_before = pd.DataFrame({
    "Column": raw.columns.astype(str),
    "Data type": [str(raw[c].dtype) for c in raw.columns],
    "Missing values": [int(raw[c].isna().sum()) for c in raw.columns],
    "Missing (%)": [round(float(raw[c].isna().mean() * 100), 2) for c in raw.columns],
    "Unique values": [int(raw[c].nunique(dropna=True)) for c in raw.columns],
}).sort_values("Missing (%)", ascending=False)
quality_after = quality_report(df)

sales_col = find_col(df.columns, ["sales", "revenue", "total_amount", "totalamount", "amount", "net_sales"])
profit_col = find_col(df.columns, ["profit"])
quantity_col = find_col(df.columns, ["quantity", "qty"])
category_col = find_col(df.columns, ["category", "product_type", "department", "product_line"])
region_col = find_col(df.columns, ["region", "state", "city", "store", "location"])
date_col = date_cols[0] if date_cols else None

st.caption(f"Dataset: **{upload.name}** · {original_rows:,} original records · Updated when controls change")
tabs = st.tabs(["Overview", "Data Cleaning", "Data Quality", "Univariate", "Bivariate", "Multivariate", "Retail Insights", "Export"])

# ---------- Overview ----------
with tabs[0]:
    st.markdown('<div class="hero-title">Understanding Your Retail Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">A live summary of your data, cleaning process, and analytical readiness.</div>',
        unsafe_allow_html=True,
    )
    a, b, c, d = st.columns(4)
    a.metric("Total Records", f"{original_rows:,}")
    b.metric("Features", f"{original_cols:,}")
    c.metric("Duplicate Rows", f"{raw_duplicate_count:,}")
    d.metric("Missing Cells", f"{raw_missing_count:,}")
    st.markdown('<div class="section-heading">After cleaning</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Rows in Analysis", f"{len(df):,}", delta=f"{len(df)-original_rows:,} vs original")
    b.metric("Numeric Features", f"{len(numeric_cols):,}")
    c.metric("Remaining Missing Cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Outlier Rows Flagged", f"{int(outlier_mask.sum()):,}")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Cleaned data preview")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Feature type distribution")
        dtype_df = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        st.plotly_chart(style_fig(px.pie(dtype_df, names="Data type", values="Columns", hole=.58,
                                         title="Column types"), 340), use_container_width=True)
    if numeric_cols:
        st.markdown("#### Numeric summary")
        st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)

# ---------- Data cleaning ----------
with tabs[1]:
    st.markdown('<div class="section-heading">Data cleaning report</div>', unsafe_allow_html=True)
    metrics = pd.DataFrame([
        ["Original rows", original_rows],
        ["Exact duplicate rows removed", duplicates_removed],
        ["Rows after duplicate removal", rows_after_dedup],
        ["Rows in final analysis", len(df)],
        ["Columns in final analysis", df.shape[1]],
        ["Original missing cells", raw_missing_count],
        ["Remaining missing cells", int(df.isna().sum().sum())],
        ["IQR outlier rows flagged", int(outlier_mask.sum())],
    ], columns=["Metric", "Value"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.markdown("#### Duplicate records (before removal)")
    if duplicate_rows.empty:
        st.success("No exact duplicate rows were detected after basic text normalization.")
    else:
        st.write(f"Rows belonging to duplicate groups: **{len(duplicate_rows):,}** (includes first occurrences).")
        st.dataframe(duplicate_rows.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download duplicate records", csv_download(duplicate_rows),
                           "duplicate_records_review.csv", "text/csv")
    st.markdown("#### Potential invalid values — review before deleting")
    if invalid_rows:
        st.dataframe(pd.DataFrame(invalid_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No generic invalid-value rules were triggered. Business-specific checks may still be needed.")
    st.markdown("#### Outlier report (IQR method)")
    if outliers.empty:
        st.info("No suitable numeric columns were available for IQR outlier detection.")
    else:
        st.dataframe(outliers.round(3), use_container_width=True, hide_index=True)
    if imputation_log:
        st.markdown("#### Missing-value imputation log")
        st.dataframe(pd.DataFrame(imputation_log), use_container_width=True, hide_index=True)
    else:
        st.caption("Missing-value imputation is off. Turn it on in the sidebar if median/mode filling suits your use case.")

# ---------- Data quality ----------
with tabs[2]:
    st.markdown('<div class="section-heading">Data quality dashboard</div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        st.markdown("#### Before cleaning")
        st.dataframe(quality_before, use_container_width=True, hide_index=True)
    with r:
        st.markdown("#### After cleaning")
        st.dataframe(quality_after, use_container_width=True, hide_index=True)
    remaining = quality_after[quality_after["Missing values"] > 0]
    if not remaining.empty:
        st.plotly_chart(style_fig(px.bar(remaining, x="Missing (%)", y="Column", orientation="h",
                                         title="Remaining missing values (%)", color="Missing (%)",
                                         color_continuous_scale="Blues"), 430), use_container_width=True)
    else:
        st.success("No missing cells remain in the final analysis dataset.")
    if date_failures:
        st.markdown("#### Date parsing review")
        st.dataframe(pd.DataFrame(date_failures), use_container_width=True, hide_index=True)

# ---------- Univariate ----------
with tabs[3]:
    st.markdown('<div class="section-heading">Univariate analysis</div>', unsafe_allow_html=True)
    available = numeric_cols + categorical_cols + date_cols
    if not available:
        st.info("No analyzable columns found.")
    else:
        selected = st.selectbox("Select one variable", available, key="uni_variable")
        s = df[selected]
        if selected in numeric_cols:
            a, b, c, d = st.columns(4)
            a.metric("Mean", f"{s.mean():,.3f}" if s.notna().any() else "N/A")
            b.metric("Median", f"{s.median():,.3f}" if s.notna().any() else "N/A")
            c.metric("Std. deviation", f"{s.std():,.3f}" if s.notna().any() else "N/A")
            d.metric("Unique values", f"{s.nunique():,}")
            l, r = st.columns(2)
            with l:
                st.plotly_chart(style_fig(px.histogram(df, x=selected, marginal="box", nbins=40,
                                                       title=f"Distribution of {selected}"), 420),
                                use_container_width=True)
            with r:
                st.plotly_chart(style_fig(px.box(df, y=selected, points="outliers",
                                                 title=f"Boxplot of {selected}"), 420),
                                use_container_width=True)
            st.dataframe(s.describe().to_frame("Statistics"), use_container_width=True)
        elif selected in date_cols:
            dates = s.dropna()
            if dates.empty:
                st.info("No valid dates in this column.")
            else:
                monthly = dates.dt.to_period("M").astype(str).value_counts().sort_index().rename_axis("Month").reset_index(name="Records")
                st.plotly_chart(style_fig(px.line(monthly, x="Month", y="Records", markers=True,
                                                  title=f"Records over time — {selected}"), 420),
                                use_container_width=True)
        else:
            counts = s.astype("string").fillna("Missing").value_counts().head(max_categories).rename_axis("Category").reset_index(name="Count")
            st.plotly_chart(style_fig(px.bar(counts, x="Count", y="Category", orientation="h",
                                             title=f"Most frequent values — {selected}"), 470),
                            use_container_width=True)
            st.dataframe(counts, use_container_width=True, hide_index=True)

# ---------- Bivariate ----------
with tabs[4]:
    st.markdown('<div class="section-heading">Bivariate analysis</div>', unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        l, r = st.columns(2)
        with l:
            x_col = st.selectbox("X-axis (numeric)", numeric_cols, key="biv_x")
        with r:
            y_options = [c for c in numeric_cols if c != x_col] or numeric_cols
            y_col = st.selectbox("Y-axis (numeric)", y_options, key="biv_y")
        sample = df[[x_col, y_col]].dropna()
        if len(sample) > chart_sample:
            sample = sample.sample(chart_sample, random_state=42)
        st.plotly_chart(style_fig(px.scatter(sample, x=x_col, y=y_col, opacity=.65,
                                             trendline="ols", title=f"{y_col} vs {x_col}"), 470),
                        use_container_width=True)
        corr = df[[x_col, y_col]].corr(numeric_only=True).iloc[0, 1]
        st.metric("Pearson correlation", f"{corr:.3f}" if pd.notna(corr) else "N/A")
    else:
        st.info("At least two numeric columns are needed for numeric bivariate analysis.")
    if numeric_cols and categorical_cols:
        l, r = st.columns(2)
        with l:
            cat = st.selectbox("Category", [c for c in categorical_cols if df[c].nunique(dropna=True) <= 50] or categorical_cols, key="biv_cat")
        with r:
            num = st.selectbox("Numeric measure", numeric_cols, key="biv_num")
        chart_df = df[[cat, num]].dropna()
        if not chart_df.empty:
            top_cats = chart_df[cat].value_counts().head(max_categories).index
            chart_df = chart_df[chart_df[cat].isin(top_cats)]
            st.plotly_chart(style_fig(px.box(chart_df, x=cat, y=num, title=f"{num} by {cat}"), 480),
                            use_container_width=True)
            st.dataframe(chart_df.groupby(cat)[num].agg(["count", "mean", "median", "sum"]).sort_values("sum", ascending=False).round(3),
                         use_container_width=True)

# ---------- Multivariate ----------
with tabs[5]:
    st.markdown('<div class="section-heading">Multivariate analysis</div>', unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        st.plotly_chart(style_fig(px.imshow(corr, text_auto=".2f", aspect="auto",
                                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                            title="Correlation heatmap"), max(440, min(750, 35*len(numeric_cols)+180)),
                        use_container_width=True)
        chosen = st.multiselect("Variables for scatter matrix", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        if len(chosen) >= 2:
            pair = df[chosen].dropna()
            if len(pair) > chart_sample:
                pair = pair.sample(chart_sample, random_state=42)
            st.plotly_chart(style_fig(px.scatter_matrix(pair, dimensions=chosen,
                                                        title="Multivariate scatter matrix"), 650),
                            use_container_width=True)
    else:
        st.info("At least two numeric columns are needed for multivariate analysis.")
    if date_col and sales_col:
        trend_df = df.dropna(subset=[date_col]).copy()
        trend_df["Month"] = trend_df[date_col].dt.to_period("M").dt.to_timestamp()
        trend = trend_df.groupby("Month", as_index=False)[sales_col].sum()
        st.plotly_chart(style_fig(px.line(trend, x="Month", y=sales_col, markers=True,
                                          title=f"Monthly {sales_col} trend"), 420),
                        use_container_width=True)

# ---------- Retail insights ----------
with tabs[6]:
    st.markdown('<div class="section-heading">Retail performance insights</div>', unsafe_allow_html=True)
    if sales_col:
        st.metric(f"Total {sales_col}", f"{df[sales_col].sum():,.2f}")
        st.metric(f"Average {sales_col}", f"{df[sales_col].mean():,.2f}")
    if profit_col:
        st.metric(f"Total {profit_col}", f"{df[profit_col].sum():,.2f}")
    if quantity_col:
        st.metric(f"Total {quantity_col}", f"{df[quantity_col].sum():,.2f}")
    if sales_col and category_col:
        summary = df.groupby(category_col, dropna=False)[sales_col].agg(Orders_or_Rows="count", Total_Sales="sum", Average_Sales="mean").sort_values("Total_Sales", ascending=False).head(max_categories)
        st.markdown(f"#### Sales by {category_col}")
        st.plotly_chart(style_fig(px.bar(summary.reset_index(), x=category_col, y="Total_Sales",
                                         title=f"Total sales by {category_col}"), 440),
                        use_container_width=True)
        st.dataframe(summary.round(2), use_container_width=True)
    if sales_col and region_col:
        summary_region = df.groupby(region_col, dropna=False)[sales_col].agg(Rows="count", Total_Sales="sum", Average_Sales="mean").sort_values("Total_Sales", ascending=False).head(max_categories)
        st.markdown(f"#### Sales by {region_col}")
        st.plotly_chart(style_fig(px.bar(summary_region.reset_index(), x=region_col, y="Total_Sales",
                                         title=f"Total sales by {region_col}"), 440),
                        use_container_width=True)
        st.dataframe(summary_region.round(2), use_container_width=True)
    if profit_col and category_col:
        prof = df.groupby(category_col, dropna=False)[profit_col].sum().sort_values(ascending=False).head(max_categories).reset_index()
        st.plotly_chart(style_fig(px.bar(prof, x=category_col, y=profit_col,
                                         title=f"Profit by {category_col}"), 420),
                        use_container_width=True)
    if not any([sales_col, profit_col, quantity_col]):
        st.info("No sales, profit, or quantity columns were recognized automatically. Use the other tabs to analyze available fields.")

# ---------- Export ----------
with tabs[7]:
    st.markdown('<div class="section-heading">Download your analysis outputs</div>', unsafe_allow_html=True)
    st.markdown("Export the cleaned dataset and supporting quality reports for your assignment.")
    st.download_button("⬇️ Download cleaned dataset (CSV)", csv_download(df),
                       "retail_sales_cleaned.csv", "text/csv", use_container_width=True)
    st.download_button("⬇️ Download data quality report", csv_download(quality_after),
                       "data_quality_report.csv", "text/csv", use_container_width=True)
    st.download_button("⬇️ Download IQR outlier report", csv_download(outliers),
                       "outlier_report.csv", "text/csv", use_container_width=True)
    st.download_button("⬇️ Download duplicate review", csv_download(duplicate_rows),
                       "duplicate_records_review.csv", "text/csv", use_container_width=True)
    summary = pd.DataFrame([{
        "file_name": upload.name,
        "original_rows": original_rows,
        "original_columns": original_cols,
        "original_duplicate_rows": raw_duplicate_count,
        "duplicate_rows_removed": duplicates_removed,
        "rows_after_deduplication": rows_after_dedup,
        "final_rows": len(df),
        "final_columns": df.shape[1],
        "original_missing_cells": raw_missing_count,
        "final_missing_cells": int(df.isna().sum().sum()),
        "outlier_rows_flagged": int(outlier_mask.sum()),
        "outlier_action": outlier_action,
    }])
    st.download_button("⬇️ Download cleaning summary", csv_download(summary),
                       "cleaning_summary.csv", "text/csv", use_container_width=True)
    if show_raw:
        st.markdown("#### Original uploaded data")
        st.dataframe(raw.head(1000), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Retail Sales Analytics Studio · Review business rules before removing invalid values or outliers.")
