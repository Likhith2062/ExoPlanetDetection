"""Tkinter GUI (colourful).  Run:  python gui.py

Tab 1 - Train & Test : run the full pipeline.
Tab 2 - Predict      : enter feature values (ranges shown) and get a
                       prediction with probability and confidence.
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

from pipeline import (
    DATASETS_DIR, FEATURES, FEATURE_INFO, PROJECT_DIR,
    format_summary, list_models, load_bundle, predict_single, run_pipeline,
)

# ---------------- colour palette ----------------
BG = "#f1f5f9"
CARD = "#ffffff"
NAVY = "#1e1b4b"
PURPLE = "#6366f1"
PURPLE_D = "#4f46e5"
GREEN = "#10b981"
GREEN_D = "#059669"
RED = "#ef4444"
RED_D = "#dc2626"
AMBER = "#f59e0b"
AMBER_BG = "#fef3c7"
CYAN = "#06b6d4"
PINK = "#ec4899"
TEXT = "#0f172a"
MUTED = "#64748b"
LOG_BG = "#0b1220"
LOG_FG = "#e2e8f0"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NASA KOI Exoplanet Classifier")
        self.geometry("1000x800")
        self.minsize(900, 700)
        self.configure(bg=BG)
        self.messages = queue.Queue()
        self.bundle = None
        self._last_model = None
        self._style()
        self._header()

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        train_tab = ttk.Frame(self.tabs, style="Card.TFrame")
        predict_tab = ttk.Frame(self.tabs, style="Card.TFrame")
        self.tabs.add(train_tab, text="  Train & Test  ")
        self.tabs.add(predict_tab, text="  Predict  ")
        self._build_train(train_tab)
        self._build_predict(predict_tab)

        self.after(100, self._poll)

    # =====================================================
    # Styling / header
    # =====================================================
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=CARD, foreground=TEXT)
        s.configure("Muted.TLabel", foreground=MUTED)
        s.configure("TCheckbutton", background=CARD, foreground=TEXT)
        s.map("TCheckbutton", background=[("active", CARD)])
        s.configure("TLabelframe", background=CARD, bordercolor=PURPLE)
        s.configure("TLabelframe.Label", background=CARD, foreground=PURPLE_D,
                    font=("Segoe UI", 10, "bold"))

        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background="#c7d2fe", foreground=NAVY,
                    padding=(14, 8), font=("Segoe UI", 11, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", PURPLE)],
              foreground=[("selected", "white")])

        def button(name, bg, active):
            s.configure(name, background=bg, foreground="white",
                        font=("Segoe UI", 10, "bold"), padding=(14, 7),
                        borderwidth=0, focusthickness=0)
            s.map(name, background=[("active", active), ("disabled", "#cbd5e1")],
                  foreground=[("disabled", "#f8fafc")])

        button("Primary.TButton", PURPLE, PURPLE_D)
        button("Success.TButton", GREEN, GREEN_D)
        button("Info.TButton", CYAN, "#0891b2")
        button("Warn.TButton", AMBER, "#d97706")

        s.configure("Green.Horizontal.TProgressbar", background=GREEN,
                    troughcolor="#e2e8f0", thickness=22, borderwidth=0)
        s.configure("Red.Horizontal.TProgressbar", background=RED,
                    troughcolor="#e2e8f0", thickness=22, borderwidth=0)
        s.configure("Busy.Horizontal.TProgressbar", background=PINK,
                    troughcolor="#e2e8f0", thickness=10, borderwidth=0)

    def _header(self):
        head = tk.Frame(self, bg=NAVY)
        head.pack(fill="x")
        tk.Label(head, text="NASA KOI Exoplanet Classifier", bg=NAVY, fg="white",
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=18,
                                                      pady=(12, 0))
        tk.Label(head, text="Random Forest  |  Train, evaluate and predict "
                            "Kepler Objects of Interest",
                 bg=NAVY, fg="#a5b4fc",
                 font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(0, 10))
        strip = tk.Frame(self, height=5)
        strip.pack(fill="x", pady=(0, 10))
        for colour in (PURPLE, CYAN, GREEN, AMBER, PINK):
            tk.Frame(strip, bg=colour, height=5).pack(side="left", fill="x",
                                                      expand=True)

    # =====================================================
    # TAB 1 - Train & Test
    # =====================================================
    def _build_train(self, parent):
        pad = {"padx": 8, "pady": 5}
        form = ttk.LabelFrame(parent, text="Settings")
        form.pack(fill="x", padx=12, pady=12)
        form.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar(
            value=os.path.join(DATASETS_DIR, "RawData.csv"))
        self.processed_var = tk.StringVar(value="Processed")
        self.model_var = tk.StringVar(value="rf_model")
        self.threshold_var = tk.StringVar()
        self.preprocess_var = tk.BooleanVar(value=True)
        self.auto_var = tk.BooleanVar(value=True)

        ttk.Label(form, text="Input CSV").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(form, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", **pad)
        ttk.Button(form, text="Browse...", style="Info.TButton",
                   command=self._browse).grid(row=0, column=2, **pad)

        ttk.Checkbutton(form, text="Preprocess raw data first",
                        variable=self.preprocess_var, command=self._toggle
                        ).grid(row=1, column=0, columnspan=3, sticky="w", **pad)

        ttk.Label(form, text="Processed file name").grid(
            row=2, column=0, sticky="w", **pad)
        self.processed_entry = ttk.Entry(form, textvariable=self.processed_var)
        self.processed_entry.grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(form, text="Model name").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(form, textvariable=self.model_var).grid(
            row=3, column=1, sticky="ew", **pad)

        ttk.Checkbutton(form, text="Tune threshold automatically "
                                   "(validation set)",
                        variable=self.auto_var, command=self._toggle
                        ).grid(row=4, column=0, columnspan=3, sticky="w", **pad)

        ttk.Label(form, text="Manual threshold (0-1)").grid(
            row=5, column=0, sticky="w", **pad)
        self.threshold_entry = ttk.Entry(form, textvariable=self.threshold_var,
                                         width=10)
        self.threshold_entry.grid(row=5, column=1, sticky="w", **pad)

        buttons = ttk.Frame(parent, style="Card.TFrame")
        buttons.pack(fill="x", padx=12)
        self.run_btn = ttk.Button(buttons, text="Run Training & Testing",
                                  style="Primary.TButton", command=self._run)
        self.run_btn.pack(side="left")
        ttk.Button(buttons, text="Open Project Folder", style="Warn.TButton",
                   command=self._open_folder).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(
            buttons, mode="indeterminate",
            style="Busy.Horizontal.TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        out = ttk.LabelFrame(parent, text="Log / Results")
        out.pack(fill="both", expand=True, padx=12, pady=12)
        self.text = tk.Text(out, wrap="none", font=("Consolas", 10),
                            state="disabled", bg=LOG_BG, fg=LOG_FG,
                            insertbackground="white", relief="flat")
        scroll = ttk.Scrollbar(out, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("err", foreground="#f87171")
        self.text.tag_configure("ok", foreground="#34d399")
        self.text.tag_configure("head", foreground="#a5b4fc")
        self.text.tag_configure("info", foreground="#67e8f9")
        self._toggle()

    def _toggle(self):
        self.processed_entry.configure(
            state="normal" if self.preprocess_var.get() else "disabled")
        self.threshold_entry.configure(
            state="disabled" if self.auto_var.get() else "normal")

    def _browse(self):
        path = filedialog.askopenfilename(
            initialdir=DATASETS_DIR, filetypes=[("CSV files", "*.csv")])
        if path:
            self.input_var.set(path)

    def _open_folder(self):
        try:
            os.startfile(PROJECT_DIR)  # Windows
        except AttributeError:
            import subprocess
            import sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",
                              PROJECT_DIR])

    def _log(self, message):
        self.messages.put(("log", str(message)))

    def _run(self):
        threshold = None
        if not self.auto_var.get():
            try:
                threshold = float(self.threshold_var.get())
            except ValueError:
                messagebox.showerror("Invalid threshold",
                                     "Enter a number between 0 and 1.")
                return
        processed = (self.processed_var.get().strip()
                     if self.preprocess_var.get() else "")
        if self.preprocess_var.get() and not processed:
            messagebox.showerror("Missing name", "Enter a processed file name.")
            return

        args = (self.input_var.get().strip(), processed,
                self.model_var.get().strip(), threshold)
        self._last_model = args[2]
        self.run_btn.configure(state="disabled")
        self.progress.start(12)
        self._set_text("")
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, input_file, processed, model, threshold):
        try:
            perf = run_pipeline(input_file, processed, model, threshold,
                                log=self._log)
            self.messages.put((
                "result",
                format_summary(perf) + "\n\nOutputs:\n" +
                "\n".join(f"  {k}: {v}" for k, v in perf["outputs"].items())))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _set_text(self, content):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", content)
        self.text.configure(state="disabled")

    def _append(self, content, tag=None):
        if tag is None:
            low = content.lower()
            if low.startswith("error"):
                tag = "err"
            elif low.startswith("done"):
                tag = "ok"
            elif low.startswith(("preprocessing", "training", "running")):
                tag = "info"
        self.text.configure(state="normal")
        self.text.insert("end", content + "\n", tag or ())
        self.text.see("end")
        self.text.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append(payload)
                else:
                    self.progress.stop()
                    self.run_btn.configure(state="normal")
                    if kind == "result":
                        self._append("\n" + "=" * 60, "head")
                        self._append(payload, "ok")
                        # make the freshly trained model available for prediction
                        if self._last_model:
                            self.model_choice.set(
                                os.path.splitext(self._last_model)[0])
                            self._load_model(silent=True)
                    else:
                        self._append("ERROR: " + payload, "err")
                        messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # =====================================================
    # TAB 2 - Predict
    # =====================================================
    def _build_predict(self, parent):
        # ---- model picker ----
        mf = ttk.LabelFrame(parent, text="1. Choose trained model")
        mf.pack(fill="x", padx=12, pady=(12, 6))
        mf.columnconfigure(1, weight=1)
        self.model_choice = tk.StringVar(value="rf_model")
        ttk.Label(mf, text="Model").grid(row=0, column=0, padx=8, pady=6)
        self.model_combo = ttk.Combobox(mf, textvariable=self.model_choice,
                                        values=list_models())
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(mf, text="Refresh", style="Info.TButton",
                   command=self._refresh_models).grid(row=0, column=2, padx=4)
        ttk.Button(mf, text="Load Model", style="Primary.TButton",
                   command=self._load_model).grid(row=0, column=3, padx=8)
        self.model_status = tk.Label(mf, text="No model loaded", bg=CARD,
                                     fg=RED_D, font=("Segoe UI", 9, "bold"))
        self.model_status.grid(row=1, column=0, columnspan=4, sticky="w",
                               padx=8, pady=(0, 6))

        # ---- input form ----
        ff = ttk.LabelFrame(parent, text="2. Enter feature values")
        ff.pack(fill="x", padx=12, pady=6)
        ff.columnconfigure(3, weight=1)
        for col, title in enumerate(["Feature", "Value", "Allowed range "
                                     "(training data)", "Meaning"]):
            tk.Label(ff, text=title, bg=PURPLE, fg="white",
                     font=("Segoe UI", 9, "bold"), anchor="w", padx=8,
                     pady=4).grid(row=0, column=col, sticky="ew")

        self.value_vars, self.entries, self.range_labels = {}, {}, {}
        for r, feat in enumerate(FEATURES, start=1):
            name, unit, desc = FEATURE_INFO[feat]
            shade = "#f5f3ff" if r % 2 else CARD
            label = f"{name}" + (f" ({unit})" if unit else "")
            tk.Label(ff, text=label, bg=shade, fg=TEXT, anchor="w", padx=8,
                     font=("Segoe UI", 10, "bold")
                     ).grid(row=r, column=0, sticky="nsew")
            var = tk.StringVar()
            var.trace_add("write", lambda *_a, f=feat: self._check_entry(f))
            ent = tk.Entry(ff, textvariable=var, width=16, relief="solid",
                           bd=1, font=("Segoe UI", 10), justify="right",
                           highlightthickness=0)
            ent.grid(row=r, column=1, padx=8, pady=3)
            rng = tk.Label(ff, text="load a model to see range", bg=shade,
                           fg=MUTED, anchor="w", padx=8, font=("Consolas", 9))
            rng.grid(row=r, column=2, sticky="nsew")
            tk.Label(ff, text=desc, bg=shade, fg=MUTED, anchor="w", padx=8,
                     font=("Segoe UI", 9)).grid(row=r, column=3, sticky="nsew")
            self.value_vars[feat], self.entries[feat] = var, ent
            self.range_labels[feat] = rng

        # ---- actions ----
        act = ttk.Frame(parent, style="Card.TFrame")
        act.pack(fill="x", padx=12, pady=6)
        ttk.Button(act, text="Predict", style="Success.TButton",
                   command=self._predict).pack(side="left")
        ttk.Button(act, text="Fill with typical values", style="Info.TButton",
                   command=self._fill_median).pack(side="left", padx=8)
        ttk.Button(act, text="Clear", style="Warn.TButton",
                   command=self._clear_inputs).pack(side="left")

        # ---- result card ----
        rf = ttk.LabelFrame(parent, text="3. Prediction")
        rf.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        rf.columnconfigure(1, weight=1)
        self.verdict = tk.Label(rf, text="Enter values and press Predict",
                                bg="#e2e8f0", fg=MUTED,
                                font=("Segoe UI", 16, "bold"), pady=10)
        self.verdict.grid(row=0, column=0, columnspan=3, sticky="ew",
                          padx=8, pady=8)

        ttk.Label(rf, text="P(Exoplanet / CONFIRMED)").grid(
            row=1, column=0, sticky="w", padx=8, pady=3)
        self.bar_conf = ttk.Progressbar(rf, maximum=100,
                                        style="Green.Horizontal.TProgressbar")
        self.bar_conf.grid(row=1, column=1, sticky="ew", padx=8)
        self.pct_conf = tk.Label(rf, text="--", bg=CARD, fg=GREEN_D,
                                 font=("Segoe UI", 11, "bold"), width=8)
        self.pct_conf.grid(row=1, column=2)

        ttk.Label(rf, text="P(Negative)").grid(
            row=2, column=0, sticky="w", padx=8, pady=3)
        self.bar_fp = ttk.Progressbar(rf, maximum=100,
                                      style="Red.Horizontal.TProgressbar")
        self.bar_fp.grid(row=2, column=1, sticky="ew", padx=8)
        self.pct_fp = tk.Label(rf, text="--", bg=CARD, fg=RED_D,
                               font=("Segoe UI", 11, "bold"), width=8)
        self.pct_fp.grid(row=2, column=2)

        self.conf_label = tk.Label(rf, text="Confidence: --", bg=CARD, fg=NAVY,
                                   font=("Segoe UI", 13, "bold"))
        self.conf_label.grid(row=3, column=0, columnspan=3, pady=(8, 0))
        self.note_label = tk.Label(rf, text="", bg=CARD, fg=MUTED,
                                   font=("Segoe UI", 9), wraplength=900,
                                   justify="center")
        self.note_label.grid(row=4, column=0, columnspan=3, pady=(0, 8))

        self._refresh_models()

    # ---------------- prediction helpers ----------------
    def _refresh_models(self):
        models = list_models()
        self.model_combo.configure(values=models)
        if models and self.model_choice.get() not in models:
            self.model_choice.set(models[0])

    def _load_model(self, silent=False):
        try:
            self.bundle = load_bundle(self.model_choice.get())
        except Exception as exc:
            self.bundle = None
            self.model_status.configure(text="No model loaded", fg=RED_D)
            if not silent:
                messagebox.showerror("Cannot load model", str(exc))
            return
        self._refresh_models()
        ranges = self.bundle["ranges"]
        for feat in FEATURES:
            if feat in ranges:
                r = ranges[feat]
                self.range_labels[feat].configure(
                    text=f"{r['min']:.6g} to {r['max']:.6g}   "
                         f"(typical {r['p05']:.4g} to {r['p95']:.4g})",
                    fg=PURPLE_D)
            else:
                self.range_labels[feat].configure(text="range unavailable",
                                                  fg=MUTED)
        self.model_status.configure(
            text=f"Loaded: {self.bundle['name']}   |   decision threshold = "
                 f"{self.bundle['threshold']:.3f}", fg=GREEN_D)
        for feat in FEATURES:
            self._check_entry(feat)

    def _check_entry(self, feat):
        """Colour the entry: white = fine, amber = outside training range,
        pink = not a number."""
        text = self.value_vars[feat].get().strip()
        ent = self.entries[feat]
        if not text:
            ent.configure(bg="white")
            return
        try:
            value = float(text)
        except ValueError:
            ent.configure(bg="#fecdd3")
            return
        r = (self.bundle or {}).get("ranges", {}).get(feat)
        if r and not (r["min"] <= value <= r["max"]):
            ent.configure(bg=AMBER_BG)
        else:
            ent.configure(bg="#d1fae5" if r else "white")

    def _fill_median(self):
        if not self.bundle or not self.bundle["ranges"]:
            messagebox.showinfo("No ranges", "Load a model first.")
            return
        for feat in FEATURES:
            r = self.bundle["ranges"].get(feat)
            if r:
                self.value_vars[feat].set(f"{r['median']:.6g}")

    def _clear_inputs(self):
        for var in self.value_vars.values():
            var.set("")
        self.verdict.configure(text="Enter values and press Predict",
                               bg="#e2e8f0", fg=MUTED)
        for bar, pct in ((self.bar_conf, self.pct_conf),
                         (self.bar_fp, self.pct_fp)):
            bar["value"] = 0
            pct.configure(text="--")
        self.conf_label.configure(text="Confidence: --", fg=NAVY)
        self.note_label.configure(text="")

    def _predict(self):
        if not self.bundle:
            self._load_model()
            if not self.bundle:
                return
        values, outside = {}, []
        for feat in FEATURES:
            name = FEATURE_INFO[feat][0]
            text = self.value_vars[feat].get().strip()
            if not text:
                messagebox.showerror("Missing value", f"Enter a value for "
                                                      f"'{name}'.")
                return
            try:
                values[feat] = float(text)
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"'{name}' must be a number "
                                     f"(got '{text}').")
                return
            r = self.bundle["ranges"].get(feat)
            if r and not (r["min"] <= values[feat] <= r["max"]):
                outside.append(f"{name} (allowed {r['min']:.6g} to "
                               f"{r['max']:.6g})")

        res = predict_single(self.bundle, values)
        p_conf, p_fp = res["probability_confirmed"], res["probability_false_positive"]
        conf = res["confidence"]

        if res["label"]:
            self.verdict.configure(text="LIKELY EXOPLANET  (CONFIRMED)",
                                   bg=GREEN, fg="white")
        else:
            self.verdict.configure(text="LIKELY FALSE POSITIVE",
                                   bg=RED, fg="white")
        self.bar_conf["value"] = p_conf * 100
        self.bar_fp["value"] = p_fp * 100
        self.pct_conf.configure(text=f"{p_conf * 100:.1f}%")
        self.pct_fp.configure(text=f"{p_fp * 100:.1f}%")

        if conf >= 0.90:
            level, colour = "Very high", GREEN_D
        elif conf >= 0.75:
            level, colour = "High", GREEN
        elif conf >= 0.60:
            level, colour = "Moderate", AMBER
        else:
            level, colour = "Low (borderline)", RED_D
        self.conf_label.configure(text=f"Confidence: {conf * 100:.1f}%  "
                                       f"-  {level}", fg=colour)

        note = (f"Predicted CONFIRMED when P(Exoplanet) >= "
                f"{res['threshold']:.3f}. Confidence is the model's "
                f"probability for the predicted class.")
        if outside:
            note += ("\nWarning - outside the training range (prediction may "
                     "be unreliable): " + "; ".join(outside))
        self.note_label.configure(text=note,
                                  fg=RED_D if outside else MUTED)


if __name__ == "__main__":
    App().mainloop()
