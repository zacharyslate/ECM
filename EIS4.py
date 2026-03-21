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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Equivalent Circuit Model Analysis",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Optional light styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Helper functions
# -----------------------------
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


# -----------------------------
# App state
# -----------------------------
if "last_circuit" not in st.session_state:
    st.session_state.last_circuit = None
if "last_fit_report" not in st.session_state:
    st.session_state.last_fit_report = ""
if "last_fit_table" not in st.session_state:
    st.session_state.last_fit_table = None
if "last_plot_bytes" not in st.session_state:
    st.session_state.last_plot_bytes = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# -----------------------------
# Header
# -----------------------------
st.title("📈 Impedance Analyzer")
st.markdown(
    '<div class="small-note">Upload a CSV file, choose a circuit, define the frequency window, and run an equivalent-circuit fit.</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------
sorted_circuit_options = OrderedDict(
    sorted(circuit_options.items(), key=lambda item: len(item[0]))
)

with st.sidebar:
    st.header("Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    selected_circuit = st.selectbox(
        "Circuit",
        options=list(sorted_circuit_options.keys()),
        index=list(sorted_circuit_options.keys()).index("R0-p(R1,CPE1)-CPE2")
        if "R0-p(R1,CPE1)-CPE2" in sorted_circuit_options
        else 0
    )

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

    num_iterations = st.number_input(
        "Fit iterations",
        value=1,
        step=1,
        min_value=1
    )

    run_analysis = st.button("Analyze", use_container_width=True)


# -----------------------------
# Top info row
# -----------------------------
info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Selected circuit", selected_circuit)

with info2:
    st.metric("Number of parameters", sorted_circuit_options[selected_circuit])

with info3:
    if uploaded_file is None:
        st.metric("File status", "No file")
    else:
        st.metric("File status", "Loaded")


# -----------------------------
# Circuit preview row
# -----------------------------
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Circuit preview")
    image_path = os.path.join("circuit_images", f"{selected_circuit}.png")
    if os.path.exists(image_path):
        st.image(image_path, caption=selected_circuit, use_container_width=True)
    else:
        st.info("No circuit image found for this circuit string.")

with right:
    st.subheader("Current analysis settings")
    st.write(f"**Frequency range:** {min_freq:g} Hz to {max_freq:g} Hz")
    st.write(f"**Iterations:** {num_iterations}")
    st.write(f"**Nyquist ticks:** Automatic")
    st.write(f"**Uploaded file:** {uploaded_file.name if uploaded_file is not None else 'None'}")


st.divider()


# -----------------------------
# Analysis
# -----------------------------
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

                num_params = sorted_circuit_options[selected_circuit]

                circuit = CustomCircuit(selected_circuit, initial_guess=[1] * num_params)
                circuit.fit(frequencies, Z)

                for _ in range(num_iterations - 1):
                    initial_guess = circuit.parameters_
                    circuit = CustomCircuit(circuit.circuit, initial_guess=initial_guess)
                    circuit.fit(frequencies, Z)

                Z_fit = circuit.predict(frequencies)

                fit_report = str(circuit)
                fit_table = extract_fit_table(circuit)

                fig = make_fit_plot(frequencies, Z, Z_fit)
                plot_bytes = fig_to_png_bytes(fig)
                plt.close(fig)

                st.session_state.last_circuit = circuit
                st.session_state.last_fit_report = fit_report
                st.session_state.last_fit_table = fit_table
                st.session_state.last_plot_bytes = plot_bytes
                st.session_state.analysis_done = True

            st.success("Analysis completed successfully.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")


# -----------------------------
# Results
# -----------------------------
if st.session_state.analysis_done:
    st.subheader("Impedance plot")

    plot_col_left, plot_col_center, plot_col_right = st.columns([0.12, 0.76, 0.12])
    with plot_col_center:
        st.image(st.session_state.last_plot_bytes)

    st.divider()

    table_col, report_col = st.columns([1.15, 0.85])

    with table_col:
        st.subheader("Fitted parameters")
        st.dataframe(st.session_state.last_fit_table, use_container_width=True, height=350)

        csv_data = st.session_state.last_fit_table.to_csv(index=False).encode("utf-8")

        d1, d2 = st.columns(2)
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

    with report_col:
        st.subheader("Fit report")
        with st.expander("Show report", expanded=True):
            st.text_area(
                "Report",
                st.session_state.last_fit_report,
                height=350,
                label_visibility="collapsed"
            )
else:
    st.info("Upload a file and click Analyze to generate the fit, plot, and parameter table.")
