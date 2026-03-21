import io
import os
import base64
import tempfile
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dash import Dash, html, dcc, dash_table, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
from flask import send_file

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit

from circuits import circuit_options
from plotting_utilities import plot_impedance_results_zoomable


# =========================================================
# APP SETUP
# =========================================================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


# =========================================================
# DATA / DEFAULTS
# =========================================================
sorted_circuit_options = OrderedDict(
    sorted(circuit_options.items(), key=lambda item: len(item[0]))
)

default_circuit = (
    "R0-p(R1,CPE1)-CPE2"
    if "R0-p(R1,CPE1)-CPE2" in sorted_circuit_options
    else list(sorted_circuit_options.keys())[0]
)


# =========================================================
# STYLES
# =========================================================
COLORS = {
    "bg": "#f8fafc",
    "card": "#ffffff",
    "border": "#e2e8f0",
    "text": "#0f172a",
    "muted": "#64748b",
    "blue": "#2563eb",
    "green": "#10b981",
    "amber": "#f59e0b",
    "purple": "#8b5cf6",
    "hero1": "#0f172a",
    "hero2": "#1e293b",
}

PAGE_STYLE = {
    "backgroundColor": COLORS["bg"],
    "fontFamily": '"Inter", "Segoe UI", sans-serif',
    "padding": "20px",
    "maxWidth": "1250px",
    "margin": "0 auto",
}

CARD_STYLE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "18px",
    "padding": "18px",
    "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
    "marginBottom": "16px",
}

METRIC_CARD_BASE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "16px",
    "padding": "14px 16px",
    "boxShadow": "0 4px 14px rgba(15, 23, 42, 0.04)",
    "minHeight": "92px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "10px 12px",
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "fontSize": "14px",
    "boxSizing": "border-box",
}

LABEL_STYLE = {
    "fontWeight": "600",
    "fontSize": "14px",
    "marginBottom": "6px",
    "display": "block",
    "color": COLORS["text"],
}

BUTTON_STYLE = {
    "width": "100%",
    "padding": "12px 14px",
    "borderRadius": "12px",
    "border": "none",
    "background": "linear-gradient(135deg, #2563eb, #1d4ed8)",
    "color": "white",
    "fontWeight": "700",
    "fontSize": "14px",
    "cursor": "pointer",
}

DOWNLOAD_BUTTON_STYLE = {
    "width": "100%",
    "padding": "12px 14px",
    "borderRadius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "background": "#ffffff",
    "color": COLORS["text"],
    "fontWeight": "600",
    "fontSize": "14px",
    "cursor": "pointer",
}

TAB_STYLE = {
    "padding": "12px 18px",
    "border": f"1px solid {COLORS['border']}",
    "backgroundColor": "#f8fafc",
    "fontWeight": "600",
    "borderTopLeftRadius": "12px",
    "borderTopRightRadius": "12px",
}

TAB_SELECTED_STYLE = {
    "padding": "12px 18px",
    "border": f"1px solid {COLORS['border']}",
    "borderBottom": "none",
    "backgroundColor": "#ffffff",
    "fontWeight": "700",
    "color": COLORS["blue"],
    "borderTopLeftRadius": "12px",
    "borderTopRightRadius": "12px",
}


# =========================================================
# HELPERS
# =========================================================
def metric_card(title, value, color):
    return html.Div(
        [
            html.Div(title, style={
                "color": COLORS["muted"],
                "fontSize": "13px",
                "marginBottom": "6px",
            }),
            html.Div(str(value), style={
                "color": COLORS["text"],
                "fontWeight": "700",
                "fontSize": "17px",
                "lineHeight": "1.25",
                "wordBreak": "break-word",
            }),
        ],
        style={
            **METRIC_CARD_BASE,
            "borderLeft": f"6px solid {color}",
        }
    )


def pill(text):
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "4px 10px",
            "borderRadius": "999px",
            "background": "#eff6ff",
            "color": "#1d4ed8",
            "fontSize": "12px",
            "fontWeight": "600",
            "marginRight": "8px",
            "marginBottom": "4px",
            "border": "1px solid #bfdbfe",
        }
    )


def get_param_names_and_units(circuit):
    names = []
    units = []

    try:
        result = circuit.get_param_names()

        if isinstance(result, tuple) and len(result) == 2:
            names, units = result
        elif isinstance(result, list):
            names = result
            units = [""] * len(names)
        else:
            names = [f"p{i+1}" for i in range(len(circuit.parameters_))]
            units = [""] * len(names)

    except Exception:
        names = [f"p{i+1}" for i in range(len(circuit.parameters_))]
        units = [""] * len(names)

    if len(units) != len(names):
        units = [""] * len(names)

    return names, units


def extract_fit_table(circuit):
    params = np.array(circuit.parameters_, dtype=float)
    names, units = get_param_names_and_units(circuit)

    errors = None

    if hasattr(circuit, "conf_") and circuit.conf_ is not None:
        try:
            errors = np.array(circuit.conf_, dtype=float)
        except Exception:
            errors = None

    if errors is None and hasattr(circuit, "covariance_") and circuit.covariance_ is not None:
        try:
            errors = np.sqrt(np.diag(np.array(circuit.covariance_, dtype=float)))
        except Exception:
            errors = None

    if errors is None:
        errors = np.array([np.nan] * len(params), dtype=float)

    rows = []
    for i, (name, value, error, unit) in enumerate(zip(names, params, errors, units), start=1):
        if value != 0 and not np.isnan(error):
            rel_error = abs(error / value) * 100
        else:
            rel_error = np.nan

        rows.append({
            "Index": i,
            "Parameter": name,
            "Value": value,
            "Error": error,
            "Relative Error (%)": rel_error,
            "Unit": unit
        })

    return pd.DataFrame(rows)


def make_fit_plot(frequencies, Z, Z_fit):
    fig = plot_impedance_results_zoomable(frequencies, Z, Z_fit)
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_fitted_data_excel(frequencies, Z, Z_fit):
    freq = np.asarray(frequencies, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    Z_fit = np.asarray(Z_fit, dtype=complex)

    nyquist_df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "Zreal_exp (ohm)": Z.real,
        "-Zimag_exp (ohm)": -Z.imag,
        "Zreal_fit (ohm)": Z_fit.real,
        "-Zimag_fit (ohm)": -Z_fit.imag,
    })

    bode_mag_df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "|Z|_exp (ohm)": np.abs(Z),
        "|Z|_fit (ohm)": np.abs(Z_fit),
    })

    bode_phase_df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "-Phase_exp (deg)": -np.angle(Z, deg=True),
        "-Phase_fit (deg)": -np.angle(Z_fit, deg=True),
    })

    residual_df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "Residual_Zreal (ohm)": (Z - Z_fit).real,
        "Residual_-Zimag (ohm)": -(Z - Z_fit).imag,
    })

    combined_df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "Zreal_exp (ohm)": Z.real,
        "Zimag_exp (ohm)": Z.imag,
        "-Zimag_exp (ohm)": -Z.imag,
        "Zreal_fit (ohm)": Z_fit.real,
        "Zimag_fit (ohm)": Z_fit.imag,
        "-Zimag_fit (ohm)": -Z_fit.imag,
        "|Z|_exp (ohm)": np.abs(Z),
        "|Z|_fit (ohm)": np.abs(Z_fit),
        "-Phase_exp (deg)": -np.angle(Z, deg=True),
        "-Phase_fit (deg)": -np.angle(Z_fit, deg=True),
        "Residual_Zreal (ohm)": (Z - Z_fit).real,
        "Residual_-Zimag (ohm)": -(Z - Z_fit).imag,
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        nyquist_df.to_excel(writer, sheet_name="Nyquist", index=False)
        bode_mag_df.to_excel(writer, sheet_name="Bode_Magnitude", index=False)
        bode_phase_df.to_excel(writer, sheet_name="Bode_Phase", index=False)
        residual_df.to_excel(writer, sheet_name="Residuals", index=False)
        combined_df.to_excel(writer, sheet_name="Combined_Data", index=False)

    output.seek(0)
    return output.getvalue()


def parse_uploaded_contents(contents, filename):
    if contents is None:
        raise ValueError("Please upload a CSV file first.")

    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    suffix = os.path.splitext(filename)[1] if filename else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(decoded)
        tmp_path = tmp.name

    try:
        frequencies, Z = preprocessing.readCSV(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return frequencies, Z


def bytes_to_data_url(data, mime_type):
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_circuit_image_url(circuit_name):
    image_path = os.path.join("circuit_images", f"{circuit_name}.png")
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return bytes_to_data_url(f.read(), "image/png")
    return None


def empty_results():
    return {
        "analysis_done": False,
        "fit_report": "",
        "fit_table": [],
        "plot_png_b64": None,
        "excel_b64": None,
        "csv_text": None,
        "last_points_used": None,
        "last_file_name": None,
        "last_circuit": None,
        "message": None,
        "message_type": None,
    }


# =========================================================
# LAYOUT
# =========================================================
app.layout = html.Div(
    [
        dcc.Store(id="analysis-store", data=empty_results()),
        dcc.Download(id="download-fit-csv"),
        dcc.Download(id="download-plot-png"),
        dcc.Download(id="download-fit-excel"),

        html.Div(
            [
                html.H1("📈 ECM - EIS Analyzer", style={
                    "margin": "0",
                    "fontSize": "2rem",
                    "fontWeight": "800",
                    "letterSpacing": "-0.02em",
                }),
                html.P(
                    "Upload EIS data, select an equivalent circuit, fit the model, and export polished "
                    "results for Nyquist, Bode magnitude, Bode phase, and residual analysis.",
                    style={
                        "margin": "8px 0 0 0",
                        "fontSize": "1rem",
                        "color": "#dbeafe",
                    }
                ),
            ],
            style={
                "padding": "1.45rem 1.6rem",
                "borderRadius": "20px",
                "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #2563eb 100%)",
                "color": "white",
                "marginBottom": "1.2rem",
                "boxShadow": "0 8px 28px rgba(15, 23, 42, 0.18)",
            }
        ),

        html.Div(
            [pill("Online GUI"), pill("Equivalent Circuit Fitting"), pill("CSV + Excel Export")],
            style={"marginBottom": "14px"}
        ),

        html.Div(
            [
                html.H3("Analysis setup", style={"marginTop": "0"}),
                html.Div(
                    "Choose your data file, circuit, frequency window, and fitting settings.",
                    style={"color": COLORS["muted"], "fontSize": "0.93rem", "marginBottom": "16px"}
                ),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Upload CSV file", style=LABEL_STYLE),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and drop or ", html.Span("select a CSV file", style={"fontWeight": "700"})]
                                    ),
                                    style={
                                        **INPUT_STYLE,
                                        "padding": "18px 12px",
                                        "textAlign": "center",
                                        "borderStyle": "dashed",
                                        "cursor": "pointer",
                                        "backgroundColor": "#fbfdff",
                                    },
                                    multiple=False,
                                ),
                            ],
                            style={"flex": "1.2"}
                        ),
                        html.Div(
                            [
                                html.Label("Circuit", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="selected-circuit",
                                    options=[{"label": k, "value": k} for k in sorted_circuit_options.keys()],
                                    value=default_circuit,
                                    clearable=False,
                                    style={"fontSize": "14px"},
                                ),
                            ],
                            style={"flex": "1.3"}
                        ),
                        html.Div(
                            [
                                html.Label("Maximum frequency (Hz)", style=LABEL_STYLE),
                                dcc.Input(
                                    id="max-freq",
                                    type="number",
                                    value=1000000.0,
                                    step=1000.0,
                                    style={**INPUT_STYLE, "marginBottom": "10px"},
                                ),
                                html.Label("Minimum frequency (Hz)", style=LABEL_STYLE),
                                dcc.Input(
                                    id="min-freq",
                                    type="number",
                                    value=1.0,
                                    step=1.0,
                                    style=INPUT_STYLE,
                                ),
                            ],
                            style={"flex": "1.0"}
                        ),
                        html.Div(
                            [
                                html.Label("Fit iterations", style=LABEL_STYLE),
                                dcc.Input(
                                    id="num-iterations",
                                    type="number",
                                    value=1,
                                    min=1,
                                    step=1,
                                    style={**INPUT_STYLE, "marginBottom": "28px"},
                                ),
                                html.Button("Analyze", id="analyze-button", n_clicks=0, style=BUTTON_STYLE),
                            ],
                            style={"flex": "0.8"}
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "alignItems": "flex-start", "flexWrap": "wrap"}
                ),
            ],
            style=CARD_STYLE
        ),

        html.Div(
            id="summary-metrics",
            style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "16px"}
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3("Circuit preview", style={"marginTop": "0"}),
                        html.Div(id="circuit-preview-content"),
                    ],
                    style={**CARD_STYLE, "flex": "1.05"}
                ),
                html.Div(
                    [
                        html.H3("Current settings", style={"marginTop": "0"}),
                        html.Div(id="settings-panel-content"),
                    ],
                    style={**CARD_STYLE, "flex": "1.0"}
                ),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "stretch", "flexWrap": "wrap"}
        ),

        html.Div(id="status-message"),

        html.Div(id="results-container"),

        html.Div(
            "Built for online electrochemical impedance analysis · NEVORA Toolbox",
            style={
                "textAlign": "center",
                "color": COLORS["muted"],
                "fontSize": "0.9rem",
                "marginTop": "1.5rem",
            }
        ),
    ],
    style=PAGE_STYLE
)


# =========================================================
# STATIC / LIVE PANELS
# =========================================================
@app.callback(
    Output("summary-metrics", "children"),
    Output("circuit-preview-content", "children"),
    Output("settings-panel-content", "children"),
    Input("selected-circuit", "value"),
    Input("upload-data", "filename"),
    Input("min-freq", "value"),
    Input("max-freq", "value"),
    Input("num-iterations", "value"),
)
def update_live_panels(selected_circuit, filename, min_freq, max_freq, num_iterations):
    metrics = [
        metric_card("Selected circuit", selected_circuit, COLORS["blue"]),
        metric_card("Number of parameters", sorted_circuit_options[selected_circuit], COLORS["green"]),
        metric_card("File status", "Loaded" if filename else "No file", COLORS["amber"]),
        metric_card("Uploaded file", filename if filename else "None", COLORS["purple"]),
    ]

    image_url = get_circuit_image_url(selected_circuit)
    if image_url:
        preview = html.Img(
            src=image_url,
            style={"width": "100%", "borderRadius": "12px", "display": "block"}
        )
    else:
        preview = html.Div(
            "No circuit image found for this circuit string.",
            style={"color": COLORS["muted"]}
        )

    settings = html.Div(
        [
            html.P(f"Frequency range: {min_freq:g} Hz to {max_freq:g} Hz", style={"margin": "0 0 10px 0"}),
            html.P(f"Iterations: {int(num_iterations) if num_iterations else 1}", style={"margin": "0 0 10px 0"}),
            html.P("Nyquist ticks: Automatic", style={"margin": "0 0 10px 0"}),
            html.P(f"Uploaded file: {filename if filename else 'None'}", style={"margin": "0"}),
        ],
        style={"color": COLORS["text"]}
    )

    return metrics, preview, settings


# =========================================================
# ANALYSIS CALLBACK
# =========================================================
@app.callback(
    Output("analysis-store", "data"),
    Output("status-message", "children"),
    Input("analyze-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("selected-circuit", "value"),
    State("min-freq", "value"),
    State("max-freq", "value"),
    State("num-iterations", "value"),
    prevent_initial_call=True,
)
def run_analysis_callback(n_clicks, contents, filename, selected_circuit, min_freq, max_freq, num_iterations):
    if not n_clicks:
        raise PreventUpdate

    try:
        if contents is None:
            raise ValueError("Please upload a CSV file first.")

        frequencies, Z = parse_uploaded_contents(contents, filename)
        frequencies, Z = preprocessing.cropFrequencies(
            frequencies, Z, freqmin=min_freq, freqmax=max_freq
        )

        if len(frequencies) == 0:
            raise ValueError("No data points remain after applying the selected frequency window.")

        num_params = sorted_circuit_options[selected_circuit]

        circuit = CustomCircuit(
            selected_circuit,
            initial_guess=[1] * num_params
        )
        circuit.fit(frequencies, Z)

        iterations = int(num_iterations) if num_iterations else 1
        for _ in range(iterations - 1):
            initial_guess = circuit.parameters_
            circuit = CustomCircuit(circuit.circuit, initial_guess=initial_guess)
            circuit.fit(frequencies, Z)

        Z_fit = circuit.predict(frequencies)

        fit_report = str(circuit)
        fit_table = extract_fit_table(circuit)
        fitted_data_excel = make_fitted_data_excel(frequencies, Z, Z_fit)

        fig = make_fit_plot(frequencies, Z, Z_fit)
        plot_bytes = fig_to_png_bytes(fig)
        plt.close(fig)

        result = {
            "analysis_done": True,
            "fit_report": fit_report,
            "fit_table": fit_table.to_dict("records"),
            "plot_png_b64": base64.b64encode(plot_bytes).decode("utf-8"),
            "excel_b64": base64.b64encode(fitted_data_excel).decode("utf-8"),
            "csv_text": fit_table.to_csv(index=False),
            "last_points_used": int(len(frequencies)),
            "last_file_name": filename,
            "last_circuit": getattr(circuit, "circuit", selected_circuit),
            "message": "Analysis completed successfully.",
            "message_type": "success",
        }

        msg = html.Div(
            result["message"],
            style={
                **CARD_STYLE,
                "color": "#166534",
                "backgroundColor": "#f0fdf4",
                "border": "1px solid #bbf7d0",
            }
        )
        return result, msg

    except Exception as e:
        result = empty_results()
        result["message"] = f"An error occurred during analysis: {e}"
        result["message_type"] = "error"

        msg = html.Div(
            result["message"],
            style={
                **CARD_STYLE,
                "color": "#991b1b",
                "backgroundColor": "#fef2f2",
                "border": "1px solid #fecaca",
            }
        )
        return result, msg


# =========================================================
# RESULTS RENDERING
# =========================================================
@app.callback(
    Output("results-container", "children"),
    Input("analysis-store", "data"),
)
def render_results(data):
    if not data or not data.get("analysis_done"):
        return html.Div(
            "Upload a file and click Analyze to generate the fit, plot, and parameter table.",
            style={
                **CARD_STYLE,
                "color": "#1e3a8a",
                "backgroundColor": "#eff6ff",
                "border": "1px solid #bfdbfe",
            }
        )

    plot_src = f"data:image/png;base64,{data['plot_png_b64']}" if data.get("plot_png_b64") else None

    summary = html.Div(
        [
            html.H3("Fit summary", style={"marginTop": "0"}),
            html.Div(
                [
                    metric_card("Last analyzed file", data.get("last_file_name"), COLORS["blue"]),
                    metric_card("Points used", data.get("last_points_used"), COLORS["green"]),
                    metric_card("Fitted circuit", data.get("last_circuit"), COLORS["purple"]),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px"}
            ),
        ],
        style=CARD_STYLE
    )

    tabs = html.Div(
        [
            dcc.Tabs(
                value="tab-plot",
                children=[
                    dcc.Tab(label="📈 Plot", value="tab-plot", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="📋 Parameters", value="tab-params", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="🧾 Fit report", value="tab-report", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="⬇️ Export", value="tab-export", style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                ],
                id="results-tabs"
            ),
            html.Div(id="tab-content-container")
        ]
    )

    return html.Div([summary, tabs])


@app.callback(
    Output("tab-content-container", "children"),
    Input("results-tabs", "value"),
    State("analysis-store", "data"),
    prevent_initial_call=False,
)
def render_tab_content(tab, data):
    if not data or not data.get("analysis_done"):
        return html.Div()

    if tab == "tab-plot":
        return html.Div(
            [
                html.H3("Impedance plot", style={"marginTop": "0"}),
                html.Img(
                    src=f"data:image/png;base64,{data['plot_png_b64']}",
                    style={"width": "100%", "borderRadius": "12px"}
                ),
            ],
            style=CARD_STYLE
        )

    if tab == "tab-params":
        columns = [
            {"name": "Index", "id": "Index"},
            {"name": "Parameter", "id": "Parameter"},
            {"name": "Value", "id": "Value"},
            {"name": "Error", "id": "Error"},
            {"name": "Relative Error (%)", "id": "Relative Error (%)"},
            {"name": "Unit", "id": "Unit"},
        ]
        return html.Div(
            [
                html.H3("Fitted parameters", style={"marginTop": "0"}),
                dash_table.DataTable(
                    data=data["fit_table"],
                    columns=columns,
                    page_size=15,
                    sort_action="native",
                    style_table={"overflowX": "auto", "borderRadius": "14px", "overflow": "hidden"},
                    style_header={
                        "backgroundColor": "#f8fafc",
                        "fontWeight": "700",
                        "border": f"1px solid {COLORS['border']}",
                    },
                    style_cell={
                        "padding": "10px",
                        "textAlign": "left",
                        "border": f"1px solid {COLORS['border']}",
                        "fontFamily": '"Inter", "Segoe UI", sans-serif',
                        "fontSize": "14px",
                    },
                    style_data={"backgroundColor": "#ffffff"},
                ),
            ],
            style=CARD_STYLE
        )

    if tab == "tab-report":
        return html.Div(
            [
                html.H3("Fit report", style={"marginTop": "0"}),
                dcc.Textarea(
                    value=data["fit_report"],
                    style={
                        "width": "100%",
                        "height": "420px",
                        "borderRadius": "12px",
                        "border": f"1px solid {COLORS['border']}",
                        "padding": "12px",
                        "fontFamily": "monospace",
                        "fontSize": "13px",
                        "boxSizing": "border-box",
                    },
                    readOnly=True,
                ),
            ],
            style=CARD_STYLE
        )

    if tab == "tab-export":
        return html.Div(
            [
                html.H3("Export results", style={"marginTop": "0"}),
                html.Div(
                    "Download the fitted parameter table, plot, and fitted impedance workbook.",
                    style={"color": COLORS["muted"], "fontSize": "0.93rem", "marginBottom": "16px"}
                ),
                html.Div(
                    [
                        html.Button("Download fit CSV", id="btn-download-csv", n_clicks=0, style=DOWNLOAD_BUTTON_STYLE),
                        html.Button("Download plot PNG", id="btn-download-png", n_clicks=0, style=DOWNLOAD_BUTTON_STYLE),
                        html.Button("Download fitted data Excel", id="btn-download-excel", n_clicks=0, style=DOWNLOAD_BUTTON_STYLE),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "16px"}
                ),
            ],
            style=CARD_STYLE
        )

    return html.Div()


# =========================================================
# DOWNLOAD CALLBACKS
# =========================================================
@app.callback(
    Output("download-fit-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, data):
    if not n_clicks or not data or not data.get("csv_text"):
        raise PreventUpdate
    return dict(content=data["csv_text"], filename="fit_parameters.csv")


@app.callback(
    Output("download-plot-png", "data"),
    Input("btn-download-png", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_png(n_clicks, data):
    if not n_clicks or not data or not data.get("plot_png_b64"):
        raise PreventUpdate
    png_bytes = base64.b64decode(data["plot_png_b64"])
    return dcc.send_bytes(png_bytes, "impedance_plot.png")


@app.callback(
    Output("download-fit-excel", "data"),
    Input("btn-download-excel", "n_clicks"),
    State("analysis-store", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, data):
    if not n_clicks or not data or not data.get("excel_b64"):
        raise PreventUpdate
    excel_bytes = base64.b64decode(data["excel_b64"])
    return dcc.send_bytes(excel_bytes, "fitted_impedance_data.xlsx")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
