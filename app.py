"""Gradio interface for the DWDM KPI Linear Regression Trend Forecast.

Wraps the forecast logic of Trend.ipynb in a web UI: the user uploads an
Optical Power Report (.xlsx), types the Receiver Optical Power Sensitivity and
gets back the fitted trend, the crossing date and the forecast graphs.
"""

import os
from datetime import datetime, timedelta

import gradio as gr
import matplotlib
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")  # No GUI backend: figures are handed to Gradio, not shown
import matplotlib.pyplot as plt

FMT = "%m/%d/%Y %H:%M:%S"
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE = os.path.join(HERE, "Historical_Performance_Template.xlsx")

# Title line of the NMS Optical Power Report, with the column widths it exports.
TEMPLATE_HEADER = (("Monitored Object", 55.0),
                   ("Performance Event", 20.0),
                   ("Monitor Period", 15.0),
                   ("End Time", 20.0),
                   ("Value", 10.0))


def write_template(path=TEMPLATE_FILE):
    """Write an empty Optical Power Report: the NMS title line and no samples.

    Handed to the user through the Download Template button so the uploaded
    report always carries the columns read_report() looks for.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for i, (name, width) in enumerate(TEMPLATE_HEADER, start=1):
        cell = ws.cell(row=1, column=i, value=name)
        cell.font = Font(name="Arial", size=11, bold=True)
        ws.column_dimensions[cell.column_letter].width = width
    wb.save(path)
    return path


def read_report(dataset):
    """Read an Optical Power Report and return its timestamps and values sorted in time.

    Expects the columns "End Time" (m/d/Y H:M:S) and "Value" produced by the NMS.
    Raises gr.Error with a readable reason when the file cannot be parsed.
    """
    try:
        f = pd.read_excel(dataset)
    except Exception as e:
        raise gr.Error(f"The Excel File could not be read because: {e!r}")

    for column in ("End Time", "Value"):
        if column not in f.columns:
            raise gr.Error(
                f'The Excel File has no "{column}" column. '
                f"Columns found: {', '.join(map(str, f.columns))}"
            )

    try:
        timestamps = [datetime.strptime(str(t).strip(), FMT) for t in f["End Time"]]
    except ValueError:
        # The NMS export is usually plain text, but pandas may already have
        # typed the column as datetime, in which case FMT no longer matches.
        try:
            timestamps = pd.to_datetime(f["End Time"]).dt.to_pydatetime().tolist()
        except Exception as e:
            raise gr.Error(f'The "End Time" column could not be parsed because: {e!r}')

    try:
        values = pd.to_numeric(f["Value"]).tolist()
    except Exception as e:
        raise gr.Error(f'The "Value" column could not be parsed because: {e!r}')

    if len(timestamps) < 2:
        raise gr.Error("At least 2 samples are needed to fit a trend.")

    order = np.argsort(timestamps)  # the report is not guaranteed to be chronological
    timestamps = [timestamps[i] for i in order]
    values = [values[i] for i in order]
    return timestamps, values


def observed_figure(timestamps, values):
    """Scatter plot of the raw KPI samples."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(timestamps, values, label="Observed")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Observed KPI")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fit_figure(timestamps, values, x, w, b):
    """Scatter plot of the samples with the fitted regression line on top."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(timestamps, values, label="Observed")
    ax.plot([timestamps[0], timestamps[-1]],
            [x[0][0] * w + b, x[-1][0] * w + b],
            color="tab:orange", label="Fit")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def forecast_figure(timestamps, values, x, w, b, sensitivity, t0,
                    x_at_threshold, predicted_date):
    """Fit extended up to the threshold crossing, with the crossing point marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if x_at_threshold is None:
        x_start, x_end = x[0][0], x[-1][0]
    else:
        x_start = min(x[0][0], x_at_threshold)
        x_end = max(x[-1][0], x_at_threshold)
    lx = np.linspace(x_start, x_end, 100)
    ld = [t0 + timedelta(days=float(d)) for d in lx]
    ax.scatter(timestamps, values, label="Observed")
    ax.plot(ld, w * lx + b, color="tab:orange", label="Fit")
    ax.axhline(sensitivity, linestyle="--", color="tab:red", label="Threshold")
    if predicted_date is not None:
        ax.scatter([predicted_date], [sensitivity], marker="X", s=120,
                   color="red", zorder=5, label="Crossing")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Trend Forecast")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def forecast(dataset, sensitivity):
    """Fit y = wx + b over the report and predict when the KPI reaches the sensitivity.

    Returns the summary in Markdown plus the observed, fit and forecast figures.
    """
    if dataset is None:
        raise gr.Error("Please upload an Optical Power Report (.xlsx) first.")
    if sensitivity is None:
        raise gr.Error("Please enter the Receiver Optical Power Sensitivity.")

    plt.close("all")
    timestamps, values = read_report(dataset)

    t0 = timestamps[0]
    x = np.array([(t - t0).total_seconds() / 86400.0
                  for t in timestamps]).reshape(-1, 1)
    y = np.array(values)

    lr = LinearRegression()
    lr.fit(x, y)
    w = lr.coef_[0]
    b = lr.intercept_
    r2 = lr.score(x, y)

    x_at_threshold = None
    predicted_date = None
    if w == 0:
        verdict = f"The trend is flat, the KPI will never reach **{sensitivity}**."
    else:
        x_at_threshold = (sensitivity - b) / w
        predicted_date = t0 + timedelta(days=x_at_threshold)
        if predicted_date < timestamps[-1]:
            verdict = (f"The KPI already crossed **{sensitivity}** on "
                       f"**{predicted_date.strftime(FMT)}** "
                       f"(the crossing is inside the observed window).")
        else:
            days_ahead = (predicted_date - timestamps[-1]).total_seconds() / 86400.0
            verdict = (f"The KPI will reach **{sensitivity}** at "
                       f"**{predicted_date.strftime(FMT)}** "
                       f"({days_ahead:,.2f} days after the last sample).")

    summary = "\n".join([
        f"- **Samples:** {len(values)}",
        f"- **Observed window:** {timestamps[0].strftime(FMT)} "
        f"to {timestamps[-1].strftime(FMT)}",
        f"- **Slope (w):** {w:.6f} units/day",
        f"- **Bias (b):** {b:.6f}",
        f"- **R²:** {r2:.4f}",
        "",
        verdict,
    ])

    return (summary,
            observed_figure(timestamps, y),
            fit_figure(timestamps, y, x, w, b),
            forecast_figure(timestamps, y, x, w, b, sensitivity, t0,
                            x_at_threshold, predicted_date))


if not os.path.exists(TEMPLATE_FILE):  # keep the download button always served
    write_template()

with gr.Blocks(title="DWDM KPI Trend Forecast") as demo:
    gr.Markdown(
        "# Linear Regression Trend Forecast for DWDM KPI's\n"
        "Upload an Optical Power Report (.xlsx with the columns **End Time** and "
        "**Value**), enter the Receiver Optical Power Sensitivity and get the "
        "forecast of when the Optical Power will reach it.\n\n"
        "No report at hand? Download the empty template below and fill it in."
    )

    with gr.Row():
        with gr.Column(scale=1):
            dataset = gr.File(
                label="Optical Power Report (.xlsx)",
                file_types=[".xlsx", ".xls"],
                type="filepath",
            )
            gr.DownloadButton(
                label="Download Template (Historical_Performance.xlsx)",
                value=TEMPLATE_FILE,
            )
            sensitivity = gr.Number(
                label="Receiver Optical Power Sensitivity",
                value=-16.0,
                step=0.1,
            )
            run = gr.Button("Forecast", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown(label="Result")

    with gr.Tabs():
        with gr.Tab("Forecast"):
            forecast_plot = gr.Plot(label="Trend Forecast")
        with gr.Tab("Fit"):
            fit_plot = gr.Plot(label="Linear Regression Fit")
        with gr.Tab("Observed"):
            observed_plot = gr.Plot(label="Observed KPI")

    run.click(
        fn=forecast,
        inputs=[dataset, sensitivity],
        outputs=[summary, observed_plot, fit_plot, forecast_plot],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)  # open the default browser on the app URL
