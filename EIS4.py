import io
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit

from circuits import circuit_options
from plotting_utilities import plot_impedance_results_zoomable


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ECM - EIS Analyzer",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .main {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.45rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #2563eb 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .hero p {
        margin: 0.45rem 0 0 0;
        font-size: 1rem;
        color: #dbeafe;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.15rem 1.15rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }

    .subtle-note {
        color: #64748b;
        font-size: 0.93rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        min-height: 92px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.08rem;
        line-height: 1.25;
        word-break: break-word;
    }

    .metric-bar-blue {
        border-left: 6px solid #2563eb;
    }

    .metric-bar-green {
        border-left: 6px solid #10b981;
    }

    .metric-bar-amber {
        border-left: 6px solid #f59e0b;
    }

    .metric-bar-purple {
        border-left: 6px solid #8b5cf6;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.7rem 1rem;
        border: none;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
    }

    div.stDownloadButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.65rem 0.9rem;
        width: 100%;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        overflow: hidden;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
        border: 1px solid #bfdbfe;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPERS
# =========================================================
def metric_card(title, value, bar_class="metric-bar-blue"):
    st.markdown(
        f"""
        <div class="metric-card {bar_class}">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
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


def read_uploaded_csv_temp(uploaded_file):
    temp_path = "temp_uploaded_file.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


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


def format_float_for_display(x, digits=6):
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


# =========================================================
# SESSION STATE
# =========================================================
if "last_circuit" not in st.session_state:
    st.session_state.last_circuit = None
if "last_fit_report" not in st.session_state:
    st.session_state.last_fit_report = ""
if "last_fit_table" not in st.session_state:
    st.session_state.last_fit_table = None
if "last_plot_bytes" not in st.session_state:
    st.session_state.last_plot_bytes = None
if "last_fitted_data_excel" not in st.session_state:
    st.session_state.last_fitted_data_excel = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "last_points_used" not in st.session_state:
    st.session_state.last_points_used = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


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
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>📈 ECM - EIS Analyzer</h1>
        <p>
            Upload EIS data, select an equivalent circuit, fit the model, and export polished results
            for Nyquist, Bode magnitude, Bode phase, and residual analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="pill">Online GUI</span>
    <span class="pill">Equivalent Circuit Fitting</span>
    <span class="pill">CSV + Excel Export</span>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MAIN INPUT AREA
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Analysis setup")
st.markdown(
    '<div class="subtle-note">Choose your data file, circuit, frequency window, and fitting settings.</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns([1.2, 1.3, 1.0, 0.8])

with c1:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

with c2:
    selected_circuit = st.selectbox(
        "Circuit",
        options=list(sorted_circuit_options.keys()),
        index=list(sorted_circuit_options.keys()).index(default_circuit)
    )

with c3:
    max_freq = st.number_input(
        "Maximum frequency (Hz)",
        value=1000000.0,
        step=1000.0,
        format="%.6g"
    )
    min_freq = st.number_input(
        "Minimum frequency (Hz)",
        value=1.0,
        step=1.0,
        format="%.6g"
    )

with c4:
    num_iterations = st.number_input(
        "Fit iterations",
        value=1,
        step=1,
        min_value=1
    )
    st.write("")
    st.write("")
    run_analysis = st.button("Analyze", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# SUMMARY METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card("Selected circuit", selected_circuit, "metric-bar-blue")

with m2:
    metric_card("Number of parameters", sorted_circuit_options[selected_circuit], "metric-bar-green")

with m3:
    metric_card("File status", "Loaded" if uploaded_file is not None else "No file", "metric-bar-amber")

with m4:
    current_file_name = uploaded_file.name if uploaded_file is not None else "None"
    metric_card("Uploaded file", current_file_name, "metric-bar-purple")


# =========================================================
# PREVIEW / SETTINGS PANEL
# =========================================================
left, right = st.columns([1.05, 1.0])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Circuit preview")
    image_path = os.path.join("circuit_images", f"{selected_circuit}.png")
    if os.path.exists(image_path):
        st.image(image_path, caption=selected_circuit, use_container_width=True)
    else:
        st.info("No circuit image found for this circuit string.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Current settings")
    st.write(f"**Frequency range:** {min_freq:g} Hz to {max_freq:g} Hz")
    st.write(f"**Iterations:** {num_iterations}")
    st.write("**Nyquist ticks:** Automatic")
    st.write(f"**Uploaded file:** {uploaded_file.name if uploaded_file is not None else 'None'}")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ANALYSIS
# =========================================================
if run_analysis:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        try:
            with st.spinner("Running fit..."):
                temp_path = read_uploaded_csv_temp(uploaded_file)

                frequencies, Z = preprocessing.readCSV(temp_path)
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

                for _ in range(num_iterations - 1):
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

                st.session_state.last_circuit = circuit
                st.session_state.last_fit_report = fit_report
                st.session_state.last_fit_table = fit_table
                st.session_state.last_plot_bytes = plot_bytes
                st.session_state.last_fitted_data_excel = fitted_data_excel
                st.session_state.analysis_done = True
                st.session_state.last_points_used = len(frequencies)
                st.session_state.last_file_name = uploaded_file.name

            st.success("Analysis completed successfully.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")


# =========================================================
# RESULTS
# =========================================================
if st.session_state.analysis_done:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Fit summary")

    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Last analyzed file", st.session_state.last_file_name, "metric-bar-blue")
    with s2:
        metric_card("Points used", st.session_state.last_points_used, "metric-bar-green")
    with s3:
        metric_card(
            "Fitted circuit",
            getattr(st.session_state.last_circuit, "circuit", selected_circuit),
            "metric-bar-purple"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Plot", "📋 Parameters", "🧾 Fit report", "⬇️ Export"]
    )

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Impedance plot")
        st.image(st.session_state.last_plot_bytes, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Fitted parameters")
        st.dataframe(
            st.session_state.last_fit_table,
            use_container_width=True,
            height=420
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Fit report")
        st.text_area(
            "Report",
            st.session_state.last_fit_report,
            height=420,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Export results")
        st.markdown(
            '<div class="subtle-note">Download the fitted parameter table, plot, and fitted impedance workbook.</div>',
            unsafe_allow_html=True
        )

        csv_data = st.session_state.last_fit_table.to_csv(index=False).encode("utf-8")

        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                label="Download fit CSV",
                data=csv_data,
                file_name="fit_parameters.csv",
                mime="text/csv",
                use_container_width=True
            )

        with d2:
            st.download_button(
                label="Download plot PNG",
                data=st.session_state.last_plot_bytes,
                file_name="impedance_plot.png",
                mime="image/png",
                use_container_width=True
            )

        with d3:
            st.download_button(
                label="Download fitted data Excel",
                data=st.session_state.last_fitted_data_excel,
                file_name="fitted_impedance_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.info("Upload a file and click Analyze to generate the fit, plot, and parameter table.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer-note">
        Built for online electrochemical impedance analysis · NEVORA Toolbox
    </div>
    """,
    unsafe_allow_html=True
)
