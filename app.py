from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MegaMart BI Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_CLEAN_DIR = r"C:\Users\anthony\Desktop\MEGAMART_ETL\data\CLEANED_OUTPUT"


def normalize_date_key(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    parsed = pd.to_datetime(series, errors="coerce")
    return pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")


def build_datasets(date_df: pd.DataFrame, store_df: pd.DataFrame, supplier_df: pd.DataFrame, sales_df: pd.DataFrame):
    for df in (date_df, store_df, supplier_df, sales_df):
        df.columns = df.columns.str.strip()

    s_work = sales_df.copy()
    if "date_key" in s_work.columns:
        s_work["date_key_norm"] = normalize_date_key(s_work["date_key"])
    else:
        s_work["date_key_norm"] = pd.Series(pd.NA, index=s_work.index, dtype="Int64")

    d_work = date_df.copy()
    if "date_key" in d_work.columns:
        d_work["date_key_norm"] = normalize_date_key(d_work["date_key"])
    elif "full_date" in d_work.columns:
        d_work["date_key_norm"] = normalize_date_key(d_work["full_date"])
    else:
        d_work["date_key_norm"] = pd.Series(pd.NA, index=d_work.index, dtype="Int64")

    st_work = store_df.copy().reset_index(drop=True)
    st_work["_store_key_1based"] = st_work.index + 1

    merged_sales = s_work.merge(d_work, on="date_key_norm", how="left", suffixes=("", "_date"))
    if "store_key" in s_work.columns:
        merged_sales = merged_sales.merge(
            st_work,
            left_on="store_key",
            right_on="_store_key_1based",
            how="left",
            suffixes=("", "_store"),
        )

    if "sales_amount" in merged_sales.columns and "gross_profit" in merged_sales.columns:
        merged_sales["margin_pct"] = np.where(
            merged_sales["sales_amount"] > 0,
            (merged_sales["gross_profit"] / merged_sales["sales_amount"]) * 100,
            0.0,
        )

    return {
        "date": date_df,
        "store": store_df,
        "supplier": supplier_df,
        "sales": sales_df,
        "merged_sales": merged_sales,
    }


@st.cache_data
def load_from_folder(folder_path: str):
    base = Path(folder_path)
    required = {
        "date": "Dim_Date_Final.csv",
        "store": "Dim_Store_Final.csv",
        "supplier": "Dim_Supplier_Final.csv",
        "sales": "Fact_Sales_Final.csv",
    }
    missing = [fname for fname in required.values() if not (base / fname).exists()]
    if missing:
        return None, missing

    return build_datasets(
        pd.read_csv(base / required["date"]),
        pd.read_csv(base / required["store"]),
        pd.read_csv(base / required["supplier"]),
        pd.read_csv(base / required["sales"]),
    ), []


st.sidebar.title("MegaMart BI Controls")
load_mode = st.sidebar.radio("How do you want to load CSVs?", ["Folder Path", "Upload 4 CSVs"])

data = None
if load_mode == "Folder Path":
    data_dir = st.sidebar.text_input("Cleaned CSV Folder", value=DEFAULT_CLEAN_DIR)
    data, missing_files = load_from_folder(data_dir)
    if missing_files:
        st.warning(f"Waiting for files in `{data_dir}` — missing: {', '.join(missing_files)}")
        st.info("Switch to **Upload 4 CSVs** in the left sidebar or update the folder path above.")
        st.stop()
else:
    f_date = st.sidebar.file_uploader("Dim_Date_Final.csv", type="csv")
    f_store = st.sidebar.file_uploader("Dim_Store_Final.csv", type="csv")
    f_sup = st.sidebar.file_uploader("Dim_Supplier_Final.csv", type="csv")
    f_sales = st.sidebar.file_uploader("Fact_Sales_Final.csv", type="csv")
    if not all([f_date, f_store, f_sup, f_sales]):
        st.info("Upload all four CSV files in the left sidebar to populate the dashboard.")
        st.stop()
    data = build_datasets(
        pd.read_csv(f_date),
        pd.read_csv(f_store),
        pd.read_csv(f_sup),
        pd.read_csv(f_sales),
    )

merged_sales = data["merged_sales"]
store_df = data["store"]
supplier_df = data["supplier"]

filtered_sales = merged_sales.copy()
if "is_hurricane_season" in filtered_sales.columns:
    hur_vals = sorted(filtered_sales["is_hurricane_season"].dropna().unique().tolist())
    if hur_vals:
        selected_hur = st.sidebar.multiselect(
            "Hurricane Season Filter",
            options=hur_vals,
            default=hur_vals,
        )
        if selected_hur:
            filtered_sales = filtered_sales[filtered_sales["is_hurricane_season"].isin(selected_hur)]

st.title("MegaMart Business Intelligence Dashboard")
st.caption("Built from cleaned ETL outputs: Fact_Sales_Final, Dim_Date_Final, Dim_Store_Final, and Dim_Supplier_Final")

total_sales = filtered_sales["sales_amount"].sum() if "sales_amount" in filtered_sales.columns else 0.0
total_gp = filtered_sales["gross_profit"].sum() if "gross_profit" in filtered_sales.columns else 0.0
total_gct = filtered_sales["gct_amount"].sum() if "gct_amount" in filtered_sales.columns else 0.0
avg_margin = (total_gp / total_sales * 100) if total_sales > 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"{total_sales:,.2f}")
k2.metric("Total Gross Profit", f"{total_gp:,.2f}")
k3.metric("Total GCT (15%)", f"{total_gct:,.2f}")
k4.metric("Gross Margin %", f"{avg_margin:.2f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Sales, GP & 15% GCT",
    "2. Hurricane & Monthly Seasonality",
    "3. Supplier Lead Time & Type",
    "4. Store Footprint & Margins",
])

with tab1:
    st.subheader("Sales, Gross Profit & 15% GCT Performance")
    st.info(
        "**Justification:** Tracks top-line revenue, gross profit, and 15% GCT across store/parish context so "
        "leadership can monitor profitability and tax obligations."
    )
    group_cols = [c for c in ["store_code", "store_name", "parish", "region"] if c in filtered_sales.columns]
    if group_cols and {"sales_amount", "gross_profit", "gct_amount"}.issubset(filtered_sales.columns):
        store_perf = (
            filtered_sales.groupby(group_cols, dropna=False)
            .agg(
                line_items=("sales_amount", "count"),
                total_sales=("sales_amount", "sum"),
                total_gross_profit=("gross_profit", "sum"),
                total_gct=("gct_amount", "sum"),
            )
            .round(2)
        )
        st.dataframe(store_perf, use_container_width=True)

    chart_df = pd.DataFrame(
        {"Amount": [total_sales, total_gp, total_gct]},
        index=["Total Sales", "Gross Profit", "GCT Amount"],
    )
    st.bar_chart(chart_df)

with tab2:
    st.subheader("Hurricane Season & Calendar Seasonality")
    st.info(
        "**Justification:** Compares sales performance across hurricane season flags and months so MegaMart can "
        "plan inventory buffers, promotions, and staffing ahead of seasonal demand shifts."
    )
    hur_col = next((c for c in filtered_sales.columns if "hurricane" in c.lower()), None)
    if hur_col and "sales_amount" in filtered_sales.columns:
        hur_summary = (
            filtered_sales.groupby(hur_col, dropna=False)
            .agg(
                line_items=("sales_amount", "count"),
                total_sales=("sales_amount", "sum"),
                avg_sales=("sales_amount", "mean"),
                total_gross_profit=("gross_profit", "sum") if "gross_profit" in filtered_sales.columns else ("sales_amount", "count"),
            )
            .round(2)
        )
        st.dataframe(hur_summary, use_container_width=True)
        st.bar_chart(hur_summary[["avg_sales", "total_sales"]])

    month_col = next((c for c in ["month_name", "month_name_date"] if c in filtered_sales.columns), None)
    if month_col and "sales_amount" in filtered_sales.columns:
        month_summary = (
            filtered_sales.groupby(month_col, dropna=False)
            .agg(
                line_items=("sales_amount", "count"),
                total_sales=("sales_amount", "sum"),
                avg_sales=("sales_amount", "mean"),
            )
            .sort_values("total_sales", ascending=False)
            .round(2)
        )
        st.dataframe(month_summary, use_container_width=True)
        st.bar_chart(month_summary["total_sales"])

with tab3:
    st.subheader("Supplier Lead Time, Rating & Category Breakdown")
    st.info(
        "**Justification:** Highlights longer-lead-time suppliers that need earlier reorder points and compares "
        "vendor ratings across supplier types and countries."
    )
    sup = supplier_df.copy()
    type_col = next((c for c in ["supplier_type", "category", "type"] if c in sup.columns), None)
    lead_col = next((c for c in ["lead_time_days", "lead_days"] if c in sup.columns), None)
    rating_col = next((c for c in ["supplier_rating", "rating"] if c in sup.columns), None)

    if type_col and lead_col:
        agg_kwargs = {
            "supplier_count": (type_col, "count"),
            "avg_lead_days": (lead_col, "mean"),
            "max_lead_days": (lead_col, "max"),
        }
        if rating_col:
            agg_kwargs["avg_rating"] = (rating_col, "mean")

        sup_summary = (
            sup.groupby(type_col, dropna=False)
            .agg(**agg_kwargs)
            .sort_values("avg_lead_days", ascending=False)
            .round(2)
        )
        st.dataframe(sup_summary, use_container_width=True)
        st.bar_chart(sup_summary["avg_lead_days"])

    if "country" in sup.columns:
        st.write("**Suppliers by Country**")
        st.dataframe(sup["country"].value_counts(dropna=False).to_frame("supplier_count"), use_container_width=True)

with tab4:
    st.subheader("Store Footprint & Transaction Margin Profile")
    st.info(
        "**Justification:** Shows store distribution across Jamaican parishes/formats alongside transaction-level "
        "gross margin and discount behaviour."
    )
    col_left, col_right = st.columns(2)
    with col_left:
        if "parish" in store_df.columns:
            parish_dist = (
                store_df.groupby("parish", dropna=False)
                .agg(store_count=("store_code", "count") if "store_code" in store_df.columns else ("parish", "count"))
                .sort_values("store_count", ascending=False)
            )
            st.write("**Stores by Parish**")
            st.dataframe(parish_dist, use_container_width=True)
            st.bar_chart(parish_dist["store_count"])

    with col_right:
        if "discount_amount" in filtered_sales.columns and "sales_amount" in filtered_sales.columns:
            s_disc = filtered_sales.copy()
            s_disc["pricing_tier"] = np.where(s_disc["discount_amount"] > 0, "Discounted", "Full Price")
            disc_summary = (
                s_disc.groupby("pricing_tier", dropna=False)
                .agg(
                    transactions=("sales_amount", "count"),
                    total_sales=("sales_amount", "sum"),
                    total_discount=("discount_amount", "sum"),
                    avg_margin_pct=("margin_pct", "mean") if "margin_pct" in s_disc.columns else ("sales_amount", "mean"),
                )
                .round(2)
            )
            st.write("**Discounted vs. Full-Price Sales**")
            st.dataframe(disc_summary, use_container_width=True)
        elif "margin_pct" in filtered_sales.columns:
            st.write("**Margin % Summary**")
            st.dataframe(filtered_sales[["sales_amount", "gross_profit", "margin_pct"]].describe().round(2), use_container_width=True)