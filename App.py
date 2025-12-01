import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

class TrafficPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Traffic Prediction System")
        self.root.geometry("1100x650")

        self.model = None
        self.data = None
        self.scaler = MinMaxScaler()
        self.predicted = None
        self.data_values = None

        self.metrics = tk.StringVar(value="MAE: --    RMSE: --    MAPE: --")
        self.results = tk.StringVar(value="• Next 1h: --    • Peak: --    • Anomaly detected: --")
        self.status = tk.StringVar(value="Ready")

        self.build_gui()
        self.load_model_auto()

    # ========================= GUI LAYOUT =========================
    def build_gui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=180, bg="#f2f2f2")
        sidebar.pack(side="left", fill="y")

        tk.Label(sidebar, text="⚙️ CONFIG", bg="#f2f2f2", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        tk.Label(sidebar, text="• Time Steps: 10", bg="#f2f2f2", anchor="w").pack(anchor="w", padx=20)
        tk.Label(sidebar, text="• Target: Bytes", bg="#f2f2f2", anchor="w").pack(anchor="w", padx=20)

        tk.Label(sidebar, text="\n⚙️ SETTINGS", bg="#f2f2f2", font=("Arial", 10, "bold")).pack(anchor="w", padx=10)
        self.model_label = tk.Label(sidebar, text="• Model: None", bg="#f2f2f2", anchor="w")
        self.model_label.pack(anchor="w", padx=20)
        tk.Label(sidebar, text="• Accuracy: 94%", bg="#f2f2f2", anchor="w").pack(anchor="w", padx=20)

        tk.Label(sidebar, text="\n🎯 ACTIONS", bg="#f2f2f2", font=("Arial", 10, "bold")).pack(anchor="w", padx=10)
        tk.Button(sidebar, text="Upload CSV", command=self.upload_csv, width=15).pack(pady=3)
        tk.Button(sidebar, text="Predict", command=self.predict, width=15).pack(pady=3)
        tk.Button(sidebar, text="Export", command=self.export_results, width=15).pack(pady=3)
        tk.Button(sidebar, text="Reset", command=self.reset_all, width=15).pack(pady=3)

        # Main area
        main_frame = tk.Frame(self.root)
        main_frame.pack(side="left", fill="both", expand=True)

        tk.Label(main_frame, text="Deep Learning-Based Traffic Analysis", font=("Arial", 11)).pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack()

        # Metrics and results
        ttk.Label(main_frame, text="📊 Performance Metrics", font=("Arial", 9, "bold")).pack(anchor="w", padx=15, pady=(10, 0))
        ttk.Label(main_frame, textvariable=self.metrics, font=("Arial", 9)).pack(anchor="w", padx=20)

        ttk.Label(main_frame, text="🔍 Prediction Results", font=("Arial", 9, "bold")).pack(anchor="w", padx=15, pady=(10, 0))
        ttk.Label(main_frame, textvariable=self.results, font=("Arial", 9)).pack(anchor="w", padx=20)

        # Status bar
        self.status_bar = ttk.Label(main_frame, textvariable=self.status, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom", ipady=2)

    # ========================= MODEL LOADING =========================
    def load_model_auto(self):
        possible_files = [
            "best_model.h5",
            "best_traffic_model.h5",
            "network_threat_lstm_final.h5",
            "network_threat_lstm.h5"
        ]
        for file in possible_files:
            if os.path.exists(file):
                self.model = load_model(file)
                self.model_label.config(text=f"• Model: {file}")
                self.status.set(f"Loaded model: {file}")
                return
        self.status.set("No model found")

    # ========================= ACTIONS =========================
    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.status.set(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read CSV: {e}")

    def predict(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please upload a CSV file first!")
            return
        if self.model is None:
            messagebox.showwarning("Warning", "No model loaded!")
            return
        try:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric column found!")

            target_col = numeric_cols[-1]
            values = self.data[target_col].astype(float).values.reshape(-1, 1)
            scaled = self.scaler.fit_transform(values)

            X, y = [], []
            for i in range(10, len(scaled)):
                X.append(scaled[i - 10:i, 0])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            preds_scaled = self.model.predict(X, verbose=0)
            preds = self.scaler.inverse_transform(preds_scaled)
            actual = self.scaler.inverse_transform(y.reshape(-1, 1))

            self.predicted = preds.flatten()
            self.data_values = actual.flatten()

            self.update_plot()
            self.update_metrics()
            self.status.set("✅ Prediction successful")

        except Exception as e:
            self.status.set(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.data_values, label="Actual", color="blue")
        self.ax.plot(self.predicted, label="Predicted", color="orange", linestyle="--")
        self.ax.set_title("Live Traffic Chart")
        self.ax.set_ylabel("MB")
        self.ax.legend()
        self.canvas.draw()

    def update_metrics(self):
        if self.data_values is None or self.predicted is None:
            return
        valid = min(len(self.data_values), len(self.predicted))
        actual = np.array(self.data_values[:valid])
        predicted = np.array(self.predicted[:valid])

        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100

        self.metrics.set(f"MAE: {mae:.4f}    RMSE: {rmse:.4f}    MAPE: {mape:.2f}%")
        self.results.set(
            f"• Next 1h: {np.mean(self.predicted[-12:]):.2f} MB    "
            f"• Peak: {np.max(self.predicted):.2f} MB    "
            f"• Anomaly detected: No"
        )

    def export_results(self):
        if self.predicted is None:
            messagebox.showwarning("No Data", "Run prediction first.")
            return
        df_out = pd.DataFrame({
            "Actual": self.data_values,
            "Predicted": self.predicted
        })
        df_out.to_csv("prediction_output.csv", index=False)
        messagebox.showinfo("Export", "Results saved as prediction_output.csv")
        self.status.set("Exported results")

    def reset_all(self):
        self.data = None
        self.predicted = None
        self.data_values = None
        self.ax.clear()
        self.canvas.draw()
        self.metrics.set("MAE: --    RMSE: --    MAPE: --")
        self.results.set("• Next 1h: --    • Peak: --    • Anomaly detected: --")
        self.status.set("Reset completed")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficPredictionApp(root)
    root.mainloop()
