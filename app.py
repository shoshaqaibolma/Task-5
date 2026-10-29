import io
import os
import re
import datetime as dt
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st

import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)




DEFAULT_SPREADSHEET_ID = os.getenv(
    "SPREADSHEET_ID",
    "1NTN25CN-PfUbPKnS1cwcBHhEqlqLqodzuTHLQvltLXY",
)
DEFAULT_SHEET_NAME = os.getenv("SHEET_NAME", "Final")




def extract_sheet_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()

    if not url_or_id:
        return ""

    if "/d/" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0]

    if "id=" in url_or_id:
        return url_or_id.split("id=")[1].split("&")[0]

    return url_or_id


def fetch_csv(sheet_id: str, sheet_name: str) -> str:


    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    text = resp.text.strip()

    if text.startswith("<") or "accounts.google.com" in resp.url:
        raise PermissionError(
            "The Google Sheet looks private. "
            "Please share it as: Anyone with the link -> Viewer."
        )

    return text


def _is_date_str(x) -> bool:
    x = str(x).strip()

    if not x:
        return False

    if not re.search(r"\d", x):
        return False

    common_formats = [
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in common_formats:
        try:
            dt.datetime.strptime(x, fmt)
            return True
        except Exception:
            pass

    try:
        pd.to_datetime(x, dayfirst=True)
        return True
    except Exception:
        return False


def _normalize_number_str(x) -> str:

    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in ("", "nan", "none"):
        return ""


    s = re.sub(r"[\s   ']", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") == 1:
        s = s.replace(",", ".")
    elif s.count(",") > 1:
        s = s.replace(",", "")

    return s


def _to_number(x) -> float:
    s = _normalize_number_str(x)

    if not s:
        return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def _is_number_str(x) -> bool:
    return np.isfinite(_to_number(x))


def parse_sheet(csv_text: str) -> pd.DataFrame:


    raw = pd.read_csv(
        io.StringIO(csv_text),
        header=None,
        dtype=str,
    )

    raw = raw.dropna(axis=1, how="all")
    raw = raw.replace(r"^\s*$", np.nan, regex=True)
    raw = raw.dropna(axis=0, how="all")

    if raw.empty:
        return pd.DataFrame()

    first = raw.iloc[0].fillna("").astype(str)

    first_date_ok = _is_date_str(first.iloc[0])
    first_nums_ok = all(
        _is_number_str(v)
        for v in first.iloc[1:]
        if str(v).strip() != ""
    )

    has_header = not (
        first_date_ok and first_nums_ok and len(first) > 1
    )

    header = 0 if has_header else None


    df = pd.read_csv(
        io.StringIO(csv_text),
        header=header,
        dtype=str,
    )

    df = df.dropna(axis=1, how="all")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all")

    if df.empty:
        return df

    if header is None:
        ncols = df.shape[1]
        names = ["Date"] + [f"Mine {i}" for i in range(1, ncols)]
        df.columns = names[:ncols]

    date_col = df.columns[0]

    if str(date_col).startswith("Unnamed") or str(date_col).strip() == "":
        df = df.rename(columns={date_col: "Date"})
        date_col = "Date"

    df[date_col] = pd.to_datetime(
        df[date_col],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    numeric_cols = []

    for c in df.columns[1:]:

        df[c] = df[c].map(_to_number).astype(float)

        if df[c].notna().any():
            numeric_cols.append(c)

    df = df[[date_col] + numeric_cols]


    if len(numeric_cols) >= 2:
        last_col = numeric_cols[-1]
        other_cols = numeric_cols[:-1]

        if not re.search("total", str(last_col), re.IGNORECASE):
            calculated_sum = df[other_cols].sum(axis=1, min_count=1)

            close = np.isclose(
                calculated_sum,
                df[last_col],
                rtol=1e-3,
                atol=1e-2,
                equal_nan=True,
            )

            if close.mean() > 0.98:
                df = df.rename(columns={last_col: "Total"})

    return df


@st.cache_data(ttl=300, show_spinner="Loading Google Sheet...")
def load_dataframe(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    csv_text = fetch_csv(sheet_id, sheet_name)
    return parse_sheet(csv_text)


def split_columns(df: pd.DataFrame):
    date_col = df.columns[0]

    numeric_cols = [
        c
        for c in df.columns[1:]
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    total_col = next(
        (
            c
            for c in numeric_cols
            if re.search("total", str(c), re.IGNORECASE)
        ),
        None,
    )

    mine_cols = [c for c in numeric_cols if c != total_col]

    if total_col is None and len(mine_cols) >= 1:
        df["Total"] = df[mine_cols].sum(axis=1, min_count=1)
        total_col = "Total"

    return date_col, mine_cols, total_col




def series_stats(s: pd.Series, iqr_k: float) -> dict:
    s = s.dropna()

    if s.empty:
        return {
            "Mean": np.nan,
            "Std Dev": np.nan,
            "Median": np.nan,
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "IQR Lower": np.nan,
            "IQR Upper": np.nan,
            "Count": 0,
        }

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    return {
        "Mean": s.mean(),
        "Std Dev": s.std(),
        "Median": s.median(),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "IQR Lower": q1 - iqr_k * iqr,
        "IQR Upper": q3 + iqr_k * iqr,
        "Count": len(s),
    }



def grubbs_indices(s: pd.Series, alpha: float, max_outliers: int):


    current = s.dropna().copy()
    found = []

    for _ in range(max_outliers):
        n = len(current)

        if n < 4:
            break

        mean = current.mean()
        std = current.std(ddof=1)

        if not np.isfinite(std) or std == 0:
            break

        abs_dev = (current - mean).abs()
        idx = abs_dev.idxmax()

        G = abs_dev.loc[idx] / std

        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)

        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(
            t_crit**2 / (n - 2 + t_crit**2)
        )

        if G > G_crit:
            found.append(idx)
            current = current.drop(idx)
        else:
            break

    return found


def detect_anomalies(
    df: pd.DataFrame,
    date_col: str,
    series: str,
    params: dict,
) -> pd.DataFrame:

    s = df[series]
    valid = s.notna()

    if valid.sum() < 10:
        return pd.DataFrame()

    mean = s[valid].mean()
    std = s[valid].std()



    z = pd.Series(np.nan, index=df.index)

    if std and np.isfinite(std) and std > 0:
        z.loc[valid] = (s[valid] - mean) / std

    z_mask = pd.Series(False, index=df.index)
    z_mask.loc[valid] = z[valid].abs() > params["z_threshold"]



    q1 = s[valid].quantile(0.25)
    q3 = s[valid].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - params["iqr_k"] * iqr
    upper = q3 + params["iqr_k"] * iqr

    iqr_mask = pd.Series(False, index=df.index)
    iqr_mask.loc[valid] = (s[valid] < lower) | (s[valid] > upper)


    ma = s.rolling(
        window=params["ma_window"],
        center=True,
        min_periods=1,
    ).mean()

    pct = pd.Series(np.nan, index=df.index)
    denom = ma.replace(0, np.nan)

    pct.loc[valid] = ((s[valid] - ma[valid]) / denom[valid]) * 100

    ma_mask = pd.Series(False, index=df.index)
    ma_mask.loc[valid] = pct[valid].abs() > params["ma_pct"]


    if params["max_grubbs"] > 0:
        g_idx = grubbs_indices(
            s[valid],
            alpha=params["grubbs_alpha"],
            max_outliers=params["max_grubbs"],
        )
    else:
        g_idx = []

    g_mask = pd.Series(False, index=df.index)

    if g_idx:
        g_mask.loc[g_idx] = True

    any_mask = valid & (z_mask | iqr_mask | ma_mask | g_mask)

    rows = []

    for idx in df.index[any_mask]:
        tests = []

        if iqr_mask.loc[idx]:
            tests.append("IQR")

        if z_mask.loc[idx]:
            tests.append("Z-score")

        if ma_mask.loc[idx]:
            tests.append("Moving Avg %")

        if g_mask.loc[idx]:
            tests.append("Grubbs")

        value = s.loc[idx]
        moving = ma.loc[idx]

        if pd.notna(moving):
            direction = "Spike" if value >= moving else "Drop"
        else:
            direction = "Spike" if value >= mean else "Drop"

        rows.append(
            {
                "Date": df.at[idx, date_col],
                "Series": series,
                "Value": value,
                "Direction": direction,
                "Tests": ", ".join(tests),
                "Z": z.loc[idx],
                "Moving Average": moving,
                "MA Deviation %": pct.loc[idx],
                "IQR Lower": lower,
                "IQR Upper": upper,
            }
        )

    return pd.DataFrame(rows)


def detect_all_anomalies(
    df: pd.DataFrame,
    date_col: str,
    series_list: list,
    params: dict,
) -> pd.DataFrame:

    frames = []

    for series in series_list:
        frame = detect_anomalies(df, date_col, series, params)

        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "Series",
                "Value",
                "Direction",
                "Tests",
                "Z",
                "Moving Average",
                "MA Deviation %",
                "IQR Lower",
                "IQR Upper",
            ]
        )

    anomalies = pd.concat(frames, ignore_index=True)
    anomalies["Date"] = pd.to_datetime(anomalies["Date"])

    anomalies = anomalies.sort_values(
        ["Date", "Series"]
    ).reset_index(drop=True)

    return anomalies




def poly_trend(
    dates: pd.Series,
    values: pd.Series,
    degree: int,
    anomaly_dates=None,
):
    dates = pd.to_datetime(dates).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)

    x = ((dates - dates.min()).dt.total_seconds() / 86400.0).values
    y = pd.to_numeric(values, errors="coerce").values.astype(float)

    valid = np.isfinite(y)

    if anomaly_dates is not None and len(anomaly_dates) > 0:
        valid &= ~dates.isin(pd.to_datetime(anomaly_dates)).values

    if valid.sum() <= degree:
        valid = np.isfinite(y)

    if valid.sum() == 0:
        return np.full(len(x), np.nan)

    coeffs = np.polyfit(x[valid], y[valid], degree)
    trend = np.polyval(coeffs, x)

    return trend



def plotly_series(
    df: pd.DataFrame,
    date_col: str,
    series: str,
    chart_type: str,
    degree: int,
    anomalies_df: pd.DataFrame,
):
    d = df[[date_col, series]].dropna().sort_values(date_col)

    fig = go.Figure()

    if chart_type == "Bar":
        fig.add_trace(
            go.Bar(
                x=d[date_col],
                y=d[series],
                name=series,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=d[date_col],
                y=d[series],
                mode="lines",
                name=series,
            )
        )

    if not anomalies_df.empty:
        anom = anomalies_df[anomalies_df["Series"] == series]
    else:
        anom = pd.DataFrame()

    anomaly_dates = (
        pd.to_datetime(anom["Date"]) if not anom.empty else []
    )

    trend = poly_trend(
        d[date_col],
        d[series],
        degree=degree,
        anomaly_dates=anomaly_dates,
    )

    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=trend,
            mode="lines",
            name=f"Trend degree {degree}",
            line=dict(dash="dash", color="rgba(0,0,0,0.75)"),
        )
    )

    if not anom.empty:
        fig.add_trace(
            go.Scatter(
                x=anom["Date"],
                y=anom["Value"],
                mode="markers",
                name="Anomalies",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="x",
                ),
            )
        )

    fig.update_layout(
        title=series,
        height=330,
        margin=dict(l=30, r=10, t=45, b=30),
        xaxis_title="",
        yaxis_title="Output",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def plotly_stacked(
    df: pd.DataFrame,
    date_col: str,
    mine_cols: list,
    total_col: str,
    degree: int,
    anomalies_df: pd.DataFrame,
):
    cols = [date_col] + mine_cols

    d = df[cols].dropna(subset=[date_col]).sort_values(date_col)

    fig = go.Figure()

    for mine in mine_cols:
        fig.add_trace(
            go.Bar(
                x=d[date_col],
                y=d[mine],
                name=mine,
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Stacked mine output",
        height=430,
        xaxis_title="",
        yaxis_title="Output",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    if total_col and total_col in df.columns:
        total = (
            df[[date_col, total_col]]
            .dropna()
            .set_index(date_col)[total_col]
            .reindex(d[date_col])
        )
    else:
        total = d[mine_cols].sum(axis=1)

    if not anomalies_df.empty:
        anomaly_dates = (
            pd.to_datetime(anomalies_df["Date"])
            .drop_duplicates()
        )
    else:
        anomaly_dates = pd.Series(dtype="datetime64[ns]")

    trend = poly_trend(
        d[date_col],
        total,
        degree=degree,
        anomaly_dates=anomaly_dates,
    )

    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=trend,
            mode="lines",
            name=f"Total trend degree {degree}",
            line=dict(dash="dash", color="black"),
        )
    )

    if not anomaly_dates.empty:
        total_series = pd.Series(
            total.values,
            index=pd.to_datetime(d[date_col]).values,
        )

        y = total_series.reindex(
            pd.to_datetime(anomaly_dates).values
        ).values

        keep = ~np.isnan(y)

        if keep.any():
            fig.add_trace(
                go.Scatter(
                    x=anomaly_dates[keep],
                    y=y[keep],
                    mode="markers",
                    name="Anomaly dates",
                    marker=dict(
                        color="red",
                        size=10,
                        symbol="x",
                    ),
                )
            )

    return fig




def mpl_series(
    df: pd.DataFrame,
    date_col: str,
    series: str,
    chart_type: str,
    degree: int,
    anomalies_df: pd.DataFrame,
):
    d = df[[date_col, series]].dropna().sort_values(date_col)

    fig, ax = plt.subplots(figsize=(10, 3.2))

    if chart_type == "Bar":
        ax.bar(
            d[date_col],
            d[series],
            label=series,
            width=0.8,
        )
    else:
        ax.plot(
            d[date_col],
            d[series],
            label=series,
            linewidth=1.3,
        )

    if not anomalies_df.empty:
        anom = anomalies_df[anomalies_df["Series"] == series]
    else:
        anom = pd.DataFrame()

    anomaly_dates = (
        pd.to_datetime(anom["Date"]) if not anom.empty else []
    )

    trend = poly_trend(
        d[date_col],
        d[series],
        degree=degree,
        anomaly_dates=anomaly_dates,
    )

    ax.plot(
        d[date_col],
        trend,
        "--",
        color="black",
        label=f"Trend degree {degree}",
    )

    if not anom.empty:
        ax.scatter(
            anom["Date"],
            anom["Value"],
            color="red",
            s=35,
            zorder=5,
            label="Anomalies",
        )

    ax.set_title(series, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    return fig


def mpl_stacked(
    df: pd.DataFrame,
    date_col: str,
    mine_cols: list,
    total_col: str,
    degree: int,
    anomalies_df: pd.DataFrame,
):
    cols = [date_col] + mine_cols

    d = df[cols].dropna(subset=[date_col]).sort_values(date_col)

    fig, ax = plt.subplots(figsize=(10, 3.6))

    bottom = np.zeros(len(d))

    for mine in mine_cols:
        vals = d[mine].fillna(0).values

        ax.bar(
            d[date_col],
            vals,
            bottom=bottom,
            label=mine,
            width=0.8,
        )

        bottom += vals

    if total_col and total_col in df.columns:
        total = (
            df[[date_col, total_col]]
            .dropna()
            .set_index(date_col)[total_col]
            .reindex(d[date_col])
        )
    else:
        total = d[mine_cols].sum(axis=1)

    if not anomalies_df.empty:
        anomaly_dates = (
            pd.to_datetime(anomalies_df["Date"])
            .drop_duplicates()
        )
    else:
        anomaly_dates = pd.Series(dtype="datetime64[ns]")

    trend = poly_trend(
        d[date_col],
        total,
        degree=degree,
        anomaly_dates=anomaly_dates,
    )

    ax.plot(
        d[date_col],
        trend,
        "--",
        color="black",
        label=f"Total trend degree {degree}",
    )

    if not anomaly_dates.empty:
        total_series = pd.Series(
            total.values,
            index=pd.to_datetime(d[date_col]).values,
        )

        y = total_series.reindex(
            pd.to_datetime(anomaly_dates).values
        ).values

        keep = ~np.isnan(y)

        if keep.any():
            ax.scatter(
                anomaly_dates[keep],
                y[keep],
                color="red",
                s=30,
                label="Anomaly dates",
            )

    ax.set_title("Stacked mine output", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    return fig



def fmt(x, nd: int = 2) -> str:
    if x is None:
        return "-"

    if isinstance(x, float) and not np.isfinite(x):
        return "-"

    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def fig_to_reportlab_image(fig, width=175 * mm, height=58 * mm):
    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buf.seek(0)

    return Image(buf, width=width, height=height)


def make_reportlab_table(data, header=True):
    t = Table(data, repeatRows=1 if header else 0)

    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ]

    if header:
        style += [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#2c3e50"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
        ]

        style.append(
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f2f2f2")],
            )
        )

    t.setStyle(TableStyle(style))

    return t


def build_pdf_report(
    df: pd.DataFrame,
    date_col: str,
    mine_cols: list,
    total_col: str,
    stats_by_series: dict,
    anomalies_df: pd.DataFrame,
    params: dict,
    chart_type: str,
    degree: int,
) -> bytes:

    all_series = mine_cols + ([total_col] if total_col else [])

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph("Mining Output Analysis Report", styles["Title"])
    )

    story.append(
        Paragraph(
            f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M}",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 8))



    story.append(
        Paragraph("Analysis parameters", styles["Heading2"])
    )

    param_rows = [
        ["Parameter", "Value"],
        ["Series", ", ".join(all_series)],
        ["Chart type", chart_type],
        ["Polynomial trendline degree", str(degree)],
        ["IQR multiplier", fmt(params["iqr_k"], 1)],
        ["Z-score threshold", fmt(params["z_threshold"], 1)],
        ["Moving average window", str(params["ma_window"])],
        ["Moving average threshold %", fmt(params["ma_pct"], 1)],
        ["Grubbs alpha", fmt(params["grubbs_alpha"], 2)],
        ["Max Grubbs outliers", str(params["max_grubbs"])],
    ]

    story.append(make_reportlab_table(param_rows))
    story.append(Spacer(1, 10))



    story.append(
        Paragraph("Summary statistics", styles["Heading2"])
    )

    if not anomalies_df.empty:
        anomaly_counts = anomalies_df.groupby("Series").size().to_dict()
    else:
        anomaly_counts = {}

    stats_rows = [
        [
            "Series",
            "Mean",
            "Std Dev",
            "Median",
            "IQR",
            "IQR Lower",
            "IQR Upper",
            "Anomalies",
        ]
    ]

    for s in all_series:
        stt = stats_by_series[s]

        stats_rows.append(
            [
                s,
                fmt(stt["Mean"]),
                fmt(stt["Std Dev"]),
                fmt(stt["Median"]),
                fmt(stt["IQR"]),
                fmt(stt["IQR Lower"]),
                fmt(stt["IQR Upper"]),
                str(anomaly_counts.get(s, 0)),
            ]
        )

    story.append(make_reportlab_table(stats_rows))
    story.append(PageBreak())



    story.append(Paragraph("Charts", styles["Heading2"]))

    if chart_type == "Stacked":
        fig = mpl_stacked(
            df,
            date_col,
            mine_cols,
            total_col,
            degree,
            anomalies_df,
        )

        story.append(fig_to_reportlab_image(fig))
        story.append(Spacer(1, 6))

        if total_col:
            fig_total = mpl_series(
                df,
                date_col,
                total_col,
                "Line",
                degree,
                anomalies_df,
            )

            story.append(fig_to_reportlab_image(fig_total))

    else:
        for s in all_series:
            fig = mpl_series(
                df,
                date_col,
                s,
                chart_type,
                degree,
                anomalies_df,
            )

            story.append(fig_to_reportlab_image(fig))
            story.append(Spacer(1, 6))

    story.append(PageBreak())



    story.append(Paragraph("Anomaly details", styles["Heading2"]))

    if anomalies_df.empty:
        story.append(
            Paragraph(
                "No anomalies detected with current parameters.",
                styles["Normal"],
            )
        )
    else:
        summary_rows = [
            ["Series", "Spike", "Drop", "Total"]
        ]

        for s in all_series:
            sub = anomalies_df[anomalies_df["Series"] == s]

            spike = int((sub["Direction"] == "Spike").sum())
            drop = int((sub["Direction"] == "Drop").sum())

            summary_rows.append(
                [s, str(spike), str(drop), str(len(sub))]
            )

        story.append(make_reportlab_table(summary_rows))
        story.append(Spacer(1, 10))

        for i, row in anomalies_df.iterrows():
            elems = []

            date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")

            title = (
                f"Anomaly {i + 1}: {date_str} — "
                f"{row['Series']} ({row['Direction']})"
            )

            elems.append(
                Paragraph(title, styles["Heading3"])
            )

            details = (
                f"<b>Value:</b> {fmt(row['Value'])}<br/>"
                f"<b>Detected by:</b> {row['Tests']}<br/>"
                f"<b>Z-score:</b> {fmt(row['Z'])}<br/>"
                f"<b>Moving average:</b> {fmt(row['Moving Average'])}<br/>"
                f"<b>Moving avg deviation:</b> {fmt(row['MA Deviation %'])}%<br/>"
                f"<b>IQR bounds:</b> "
                f"[{fmt(row['IQR Lower'])}, {fmt(row['IQR Upper'])}]"
            )

            elems.append(
                Paragraph(details, styles["Normal"])
            )

            elems.append(Spacer(1, 8))

            story.append(KeepTogether(elems))

    doc.build(story)

    pdf_bytes = buf.getvalue()
    buf.close()

    return pdf_bytes




st.set_page_config(
    page_title="Mine Output Dashboard",
    layout="wide",
)

st.title("Weyland-Yutani Mine Output Dashboard")

st.caption(
    "Online analysis dashboard for mine output data stored in Google Sheets."
)



st.sidebar.header("Data source")

url_input = st.sidebar.text_input(
    "Google Sheet URL or ID",
    value=DEFAULT_SPREADSHEET_ID,
)

sheet_name = st.sidebar.text_input(
    "Sheet name",
    value=DEFAULT_SHEET_NAME,
)

st.sidebar.markdown("---")

st.sidebar.header("Visualization")

chart_type = st.sidebar.radio(
    "Chart type",
    options=["Line", "Bar", "Stacked"],
    horizontal=True,
)

degree = st.sidebar.select_slider(
    "Polynomial trendline degree",
    options=[1, 2, 3, 4],
    value=1,
)

st.sidebar.markdown("---")

st.sidebar.header("Anomaly detection")

with st.sidebar.expander("Parameters", expanded=True):
    iqr_k = st.slider(
        "IQR multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )

    z_threshold = st.slider(
        "Z-score threshold",
        min_value=2.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
    )

    ma_window = st.slider(
        "Moving average window, days",
        min_value=3,
        max_value=30,
        value=7,
        step=1,
    )

    ma_pct = st.slider(
        "Moving average deviation threshold, %",
        min_value=5.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )

    grubbs_alpha = st.select_slider(
        "Grubbs alpha",
        options=[0.01, 0.05, 0.10],
        value=0.05,
    )

    max_grubbs = st.number_input(
        "Max Grubbs outliers per series",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
    )



sheet_id = extract_sheet_id(url_input)

if not sheet_id:
    st.info(
        "Paste your public Google Sheet URL or ID into the sidebar."
    )

    st.markdown(
        """
        ### Expected sheet format

        Use a sheet like `Final`:

        ```text
        Date | Mine 1 | Mine 2 | Mine 3 | Total
        ```

        The `Total` column is optional. If it is missing, it will be calculated.
        """
    )

    st.stop()


try:
    df = load_dataframe(sheet_id, sheet_name)
except Exception as e:
    st.error(f"Could not load Google Sheet: {e}")
    st.stop()


if df.empty:
    st.warning("No data found in the sheet.")
    st.stop()


date_col, mine_cols, total_col = split_columns(df)

all_series = mine_cols + ([total_col] if total_col else [])

if not all_series:
    st.warning("No numeric data columns found.")
    st.stop()


params = {
    "iqr_k": float(iqr_k),
    "z_threshold": float(z_threshold),
    "ma_window": int(ma_window),
    "ma_pct": float(ma_pct),
    "grubbs_alpha": float(grubbs_alpha),
    "max_grubbs": int(max_grubbs),
}



stats_by_series = {
    s: series_stats(df[s], params["iqr_k"])
    for s in all_series
}

anomalies_df = detect_all_anomalies(
    df,
    date_col,
    all_series,
    params,
)



st.subheader("Overview")

if total_col:
    total_stats = stats_by_series[total_col]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total mean daily output",
        fmt(total_stats["Mean"]),
    )

    c2.metric(
        "Total median",
        fmt(total_stats["Median"]),
    )

    c3.metric(
        "Total std deviation",
        fmt(total_stats["Std Dev"]),
    )

    c4.metric(
        "Anomalies detected",
        len(anomalies_df),
    )
else:
    c1, c2, c3 = st.columns(3)

    first_series = all_series[0]
    first_stats = stats_by_series[first_series]

    c1.metric(
        f"{first_series} mean",
        fmt(first_stats["Mean"]),
    )

    c2.metric(
        f"{first_series} median",
        fmt(first_stats["Median"]),
    )

    c3.metric(
        "Anomalies detected",
        len(anomalies_df),
    )



tab_stats, tab_charts, tab_anomalies, tab_data = st.tabs(
    [
        "Statistics",
        "Charts",
        "Anomalies",
        "Raw data",
    ]
)



with tab_stats:
    st.subheader("Statistics by mine and total")

    stats_df = pd.DataFrame(stats_by_series).T
    stats_df.index.name = "Series"

    st.dataframe(
        stats_df.style.format(lambda x: fmt(x, 2)),
        width="stretch",
    )



with tab_charts:
    st.subheader("Output charts")

    if chart_type == "Stacked":
        st.plotly_chart(
            plotly_stacked(
                df,
                date_col,
                mine_cols,
                total_col,
                degree,
                anomalies_df,
            ),
            width="stretch",
        )

        if total_col:
            st.plotly_chart(
                plotly_series(
                    df,
                    date_col,
                    total_col,
                    "Line",
                    degree,
                    anomalies_df,
                ),
                width="stretch",
            )

    else:
        for s in all_series:
            st.plotly_chart(
                plotly_series(
                    df,
                    date_col,
                    s,
                    chart_type,
                    degree,
                    anomalies_df,
                ),
                width="stretch",
            )



with tab_anomalies:
    st.subheader("Detected anomalies")

    if anomalies_df.empty:
        st.success("No anomalies detected with current parameters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            series_filter = st.multiselect(
                "Filter by series",
                options=anomalies_df["Series"].unique(),
                default=anomalies_df["Series"].unique(),
            )

        with col2:
            direction_filter = st.multiselect(
                "Filter by direction",
                options=["Spike", "Drop"],
                default=["Spike", "Drop"],
            )

        filtered_anomalies = anomalies_df[
            anomalies_df["Series"].isin(series_filter)
            & anomalies_df["Direction"].isin(direction_filter)
        ]

        st.dataframe(
            filtered_anomalies,
            width="stretch",
        )

        st.caption(
            "Anomalies are detected using IQR, Z-score, "
            "moving-average deviation, and Grubbs test."
        )



with tab_data:
    st.subheader("Loaded data")

    st.dataframe(
        df,
        width="stretch",
    )



st.sidebar.markdown("---")

if st.sidebar.button("Build PDF report"):
    with st.spinner("Building PDF report..."):
        pdf_bytes = build_pdf_report(
            df=df,
            date_col=date_col,
            mine_cols=mine_cols,
            total_col=total_col,
            stats_by_series=stats_by_series,
            anomalies_df=anomalies_df,
            params=params,
            chart_type=chart_type,
            degree=degree,
        )

    st.session_state["pdf_bytes"] = pdf_bytes


if "pdf_bytes" in st.session_state:
    st.sidebar.download_button(
        "Download PDF report",
        data=st.session_state["pdf_bytes"],
        file_name=f"mine_report_{dt.date.today():%Y%m%d}.pdf",
        mime="application/pdf",
    )