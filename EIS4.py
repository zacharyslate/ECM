import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from impedance import preprocessing
from circuits import circuit_options
from plotting_utilities import plot_impedance_results_zoomable, plot_impedance_results
from impedance.models.circuits import CustomCircuit
from PIL import Image, ImageTk
from collections import OrderedDict


class ImpedanceAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Impedance Analyzer")

        # Ensure script exits when window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # File Path Variable
        self.file_path = None

        # Store latest fit objects/results
        self.last_circuit = None
        self.last_frequencies = None
        self.last_Z = None
        self.last_Z_fit = None
        self.fit_report = ""

        # Predefined Circuits and Parameters
        self.circuit_options = OrderedDict(
            sorted(circuit_options.items(), key=lambda item: len(item[0]))
        )

        # Selected Circuit Variable
        self.selected_circuit = tk.StringVar(value="R0-p(R1,CPE1)-CPE2")

        # Input Parameters
        self.param_vars = {
            "Max Frequency": tk.IntVar(value=1000000),
            "Min Frequency": tk.IntVar(value=1),
            "Major Ticks (Nyquist)": tk.IntVar(value=500),
            "Number of Iterations": tk.IntVar(value=1)
        }

        # Create GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # Configure the grid
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_columnconfigure(5, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(6, weight=1)

        # File Load Button
        load_btn = tk.Button(self.root, text="Load File", command=self.load_file)
        load_btn.grid(row=0, column=0, padx=10, pady=10)

        # Analyze Button
        analyze_btn = tk.Button(self.root, text="Analyze", command=self.analyze)
        analyze_btn.grid(row=0, column=1, padx=10, pady=10)

        # Save Button
        save_btn = tk.Button(self.root, text="Save Results", command=self.save_results)
        save_btn.grid(row=0, column=2, padx=10, pady=10)

        # Circuit Dropdown
        tk.Label(self.root, text="Circuit String (B2)").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )

        def on_circuit_change(selected_circuit):
            self.update_parameters(selected_circuit)
            self.update_circuit_image()

        circuit_menu = tk.OptionMenu(
            self.root,
            self.selected_circuit,
            *self.circuit_options.keys(),
            command=on_circuit_change
        )
        circuit_menu.grid(row=1, column=1, padx=10, pady=5)

        # Parameter Input Fields
        for idx, (label, var) in enumerate(self.param_vars.items()):
            tk.Label(self.root, text=label).grid(
                row=2 + idx, column=0, padx=10, pady=5, sticky="w"
            )
            tk.Entry(self.root, textvariable=var).grid(
                row=2 + idx, column=1, padx=10, pady=5
            )

        # Results Display
        tk.Label(self.root, text="Fit Results:").grid(
            row=4, column=3, padx=10, pady=5, sticky="nw"
        )
        self.results_text = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=60, height=18
        )
        self.results_text.grid(
            row=4, column=5, columnspan=2, padx=10, pady=5, sticky="nsew"
        )

        # Placeholder for Matplotlib Figure
        self.figure = None

        # Image Placeholder
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=1, column=3, columnspan=3, padx=10, pady=10)

        # Initial Update of Circuit Image
        self.update_circuit_image()

    def update_circuit_image(self, *args):
        """Update the image displayed below based on the selected circuit."""
        circuit_name = self.selected_circuit.get()
        image_path = os.path.join("circuit_images", f"{circuit_name}.png")

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image.thumbnail((800, 200))
                self.circuit_image = ImageTk.PhotoImage(image)
                self.image_label.config(image=self.circuit_image, text="")
            except Exception as e:
                print(f"Error loading image: {e}")
                self.image_label.config(image="", text="Error loading image")
        else:
            self.image_label.config(image="", text="Image not found")

    def load_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

    def update_parameters(self, selected_circuit):
        num_params = self.circuit_options.get(selected_circuit, 0)
        # Optional: show/update number of parameters somewhere if desired

    def analyze(self):
        if not self.file_path:
            messagebox.showerror("Error", "No file loaded. Please load a file first.")
            return

        try:
            plt.close("all")

            # Extract Parameters
            circuit_str = self.selected_circuit.get()
            num_params = self.circuit_options[circuit_str]
            max_freq = self.param_vars["Max Frequency"].get()
            min_freq = self.param_vars["Min Frequency"].get()
            major_ticks = self.param_vars["Major Ticks (Nyquist)"].get()
            min_iterations = self.param_vars["Number of Iterations"].get()

            # Load and preprocess data
            frequencies, Z = preprocessing.readCSV(self.file_path)
            frequencies, Z = preprocessing.cropFrequencies(
                frequencies, Z, freqmin=min_freq, freqmax=max_freq
            )

            # Define and fit the circuit
            circuit = CustomCircuit(circuit_str, initial_guess=[1] * num_params)
            circuit.fit(frequencies, Z)

            # Optional repeated refinement
            for _ in range(min_iterations):
                initial_guess = circuit.parameters_
                circuit = CustomCircuit(circuit.circuit, initial_guess=initial_guess)
                circuit.fit(frequencies, Z)

            Z_fit = circuit.predict(frequencies)

            # Store latest results
            self.last_circuit = circuit
            self.last_frequencies = frequencies
            self.last_Z = Z
            self.last_Z_fit = Z_fit

            # Display fit results
            self.display_fit_results(circuit)
            print(circuit)

            # Plot results
            self.plot_results(frequencies, Z, Z_fit, major_ticks)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis:\n{e}")

    def display_fit_results(self, circuit):
        self.results_text.delete(1.0, tk.END)
        self.fit_report = str(circuit)
        self.results_text.insert(tk.END, self.fit_report)

    def plot_results(self, frequencies, Z, Z_fit, major_ticks):
        if self.figure:
            try:
                self.figure.get_tk_widget().destroy()
            except Exception:
                pass

        # External plotting function
        fig = plot_impedance_results_zoomable(frequencies, Z, Z_fit, major_ticks)

        # If you later want embedded plotting, uncomment:
        # self.figure = FigureCanvasTkAgg(fig, master=self.root)
        # self.figure.get_tk_widget().grid(row=6, column=0, columnspan=3, pady=10)
        # fig.tight_layout()

    def get_param_names_and_units(self, circuit):
        """
        Robustly obtain parameter names and units from impedance.py.
        Handles different possible return formats from get_param_names().
        """
        names = []
        units = []

        try:
            result = circuit.get_param_names()

            # Case 1: returns (names, units)
            if isinstance(result, tuple) and len(result) == 2:
                names, units = result

            # Case 2: returns just names
            elif isinstance(result, list):
                names = result
                units = [""] * len(names)

            else:
                names = [f"p{i+1}" for i in range(len(circuit.parameters_))]
                units = [""] * len(names)

        except Exception:
            names = [f"p{i+1}" for i in range(len(circuit.parameters_))]
            units = [""] * len(names)

        # Safety
        if len(units) != len(names):
            units = [""] * len(names)

        return names, units

    def extract_fit_table(self, circuit):
        """
        Return a list of dictionaries for CSV export.
        Includes parameter names, fitted values, uncertainties, relative errors, and units.
        """
        params = np.array(circuit.parameters_, dtype=float)
        names, units = self.get_param_names_and_units(circuit)

        # Try to retrieve fit uncertainties/confidence values
        errors = None

        # impedance.py often stores these as conf_
        if hasattr(circuit, "conf_") and circuit.conf_ is not None:
            try:
                errors = np.array(circuit.conf_, dtype=float)
            except Exception:
                errors = None

        # Fallback: covariance matrix -> sqrt(diag(cov))
        if errors is None and hasattr(circuit, "covariance_") and circuit.covariance_ is not None:
            try:
                errors = np.sqrt(np.diag(np.array(circuit.covariance_, dtype=float)))
            except Exception:
                errors = None

        # Final fallback
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

        return rows

    def save_fit_csv(self):
        """Save fitted parameters in CSV table format."""
        if self.last_circuit is None:
            messagebox.showerror("Error", "No fit available. Please analyze data first.")
            return

        csv_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            title="Save Fit Parameters Table"
        )

        if not csv_path:
            messagebox.showwarning("Save Cancelled", "CSV parameter table not saved.")
            return

        try:
            rows = self.extract_fit_table(self.last_circuit)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Index",
                        "Parameter",
                        "Value",
                        "Error",
                        "Relative_Error_%",
                        "Unit"
                    ]
                )
                writer.writeheader()
                writer.writerows(rows)

            messagebox.showinfo("File Saved", f"CSV fit table saved as:\n{csv_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving CSV:\n{e}")

    def save_text_report(self):
        """Save the full text fit report exactly as shown by str(circuit)."""
        if not self.fit_report:
            messagebox.showerror("Error", "No fit report available. Please analyze data first.")
            return

        results_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Fitting Results Report"
        )

        if not results_path:
            messagebox.showwarning("Save Cancelled", "Fitting results report not saved.")
            return

        try:
            with open(results_path, "w", encoding="utf-8") as results_file:
                results_file.write(self.fit_report)
            messagebox.showinfo("File Saved", f"Fitting results saved as:\n{results_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving report:\n{e}")

    def save_plot(self):
        """Save the current matplotlib plot."""
        plot_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
            title="Save Impedance Plot"
        )

        if not plot_path:
            messagebox.showwarning("Save Cancelled", "Plot not saved.")
            return

        try:
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("File Saved", f"Plot saved as:\n{plot_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving plot:\n{e}")

    def save_circuit_image(self):
        """Save the displayed circuit image."""
        circuit_image_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
            title="Save Circuit Image"
        )

        if not circuit_image_path:
            messagebox.showwarning("Save Cancelled", "Circuit image not saved.")
            return

        try:
            circuit_name = self.selected_circuit.get()
            image_path = os.path.join("circuit_images", f"{circuit_name}.png")

            if os.path.exists(image_path):
                circuit_image = Image.open(image_path)
                circuit_image.save(circuit_image_path)
                messagebox.showinfo("File Saved", f"Circuit image saved as:\n{circuit_image_path}")
            else:
                messagebox.showwarning("Image Not Found", f"No image found for circuit:\n{circuit_name}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving circuit image:\n{e}")

    def save_results(self):
        if self.last_circuit is None:
            messagebox.showerror("Error", "No analysis performed. Please analyze data first.")
            return

        try:
            self.save_plot()
            self.save_fit_csv()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving:\n{e}")

    def on_close(self):
        self.root.destroy()
        plt.close("all")
        os._exit(0)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImpedanceAnalyzerApp(root)
    root.mainloop()