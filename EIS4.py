import io
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit

from circuits import circuit_options
from plotting_utilities import plot_impedance_results_zoomable


st.set_page_config(page_title="Impedance Analyzer", layout="wide")


# -----------------------------
# Helper functions
# -----------------------------
def get_param_names_and_units(circuit):
    """
    Robustly obtain parameter names and units from impedance.py.
    Handles different possible return formats from get_param_names().
    """
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
    """
    Return a DataFrame of fitted parameters.
    Includes parameter names, fitted values, uncertainties,
    relative errors, and units.
    """
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
            "Relative_Error_%": rel_error,
            "Unit": unit
        })

    return pd.DataFrame(rows)


def read_uploaded_csv_temp(uploaded_file):
    """
    impedance.preprocessing.readCSV expects a filepath,
    so we save the uploaded file temporarily.
    """
    temp_path = "temp_uploaded_file.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path


def make_fit_plot(frequencies, Z, Z_fit, major_ticks):
    """
    Uses your external plotting utility.
    """
    fig = plot_impedance_results_zoomable(frequencies, Z, Z_fit, major_ticks)
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


# -----------------------------
# Sidebar controls
# -----------------------------
st.title("Impedance Analyzer")

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
        "Circuit String",
        options=list(sorted_circuit_options.keys()),
        index=list(sorted_circuit_options.keys()).index("R0-p(R1,CPE1)-CPE2")
        if "R0-p(R1,CPE1)-CPE2" in sorted_circuit_options
        else 0
    )

    max_freq = st.number_input("Max Frequency (Hz)", value=1000000.0, step=1000.0, format="%.6g")
    min_freq = st.number_input("Min Frequency (Hz)", value=1.0, step=1.0, format="%.6g")
    major_ticks = st.number_input("Major Ticks (Nyquist)", value=500, step=50)
    num_iterations = st.number_input("Number of Iterations", value=1, step=1, min_value=1)

    run_analysis = st.button("Analyze")


# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Fit Results")

    if st.session_state.last_fit_report:
        st.text_area(
            "Report",
            st.session_state.last_fit_report,
            height=400
        )
    else:
        st.info("Run an analysis to see the fit report.")

with col2:
    st.subheader("Circuit Info")
    st.write(f"**Selected circuit:** `{selected_circuit}`")
    st.write(f"**Number of parameters:** {sorted_circuit_options[selected_circuit]}")

    image_path = os.path.join("circuit_images", f"{selected_circuit}.png")
    if os.path.exists(image_path):
        st.image(image_path, caption=selected_circuit, use_container_width=True)
    else:
        st.warning("Circuit image not found.")


# -----------------------------
# Analysis
# -----------------------------
if run_analysis:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        try:
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
            fig = make_fit_plot(frequencies, Z, Z_fit, major_ticks)
            plot_bytes = fig_to_png_bytes(fig)

            st.session_state.last_circuit = circuit
            st.session_state.last_fit_report = fit_report
            st.session_state.last_fit_table = fit_table
            st.session_state.last_plot_bytes = plot_bytes

            st.success("Analysis completed successfully.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")


# -----------------------------
# Show outputs
# -----------------------------
if st.session_state.last_plot_bytes is not None:
    st.subheader("Impedance Plot")
    st.image(st.session_state.last_plot_bytes)

if st.session_state.last_fit_table is not None:
    st.subheader("Fitted Parameters")
    st.dataframe(st.session_state.last_fit_table, use_container_width=True)

    csv_data = st.session_state.last_fit_table.to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download Fit CSV",
            data=csv_data,
            file_name="fit_parameters.csv",
            mime="text/csv"
        )

    with c2:
        st.download_button(
            label="Download Plot PNG",
            data=st.session_state.last_plot_bytes,
            file_name="impedance_plot.png",
            mime="image/png"
        )
