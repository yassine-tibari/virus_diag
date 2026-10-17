"""Interface clinique de virus_diag.

- Interface graphique (Tkinter) : ``run_gui(predictor)``
- Version texte (ligne de commande) : ``run(predictor)``

L'interface ne connaît pas le modèle : elle ne parle qu'au ClinicalPredictor
(predict_proba / label), comme l'API HTTP.
"""
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # tkinter absent de certaines installations Python/Linux
    tk = None
    ttk = None

# nom de variable -> (libellé, type, minimum, maximum, valeur par défaut)
FIELDS = {
    "age": ("Âge (années)", "num", 0, 120, "45"),
    "temperature": ("Température (°C)", "num", 30, 45, "37.0"),
    "heart_rate": ("Fréquence cardiaque (bpm)", "num", 20, 250, "75"),
    "oxygen_saturation": ("Saturation en oxygène (%)", "num", 50, 100, "98"),
    "cough": ("Toux", "bool", 0, 1, "0"),
    "fatigue": ("Fatigue", "bool", 0, 1, "0"),
    "contact_with_case": ("Contact avec un cas confirmé", "bool", 0, 1, "0"),
}

EXAMPLES = {
    "Exemple : patient infecté": {
        "age": 62, "temperature": 39.1, "heart_rate": 104, "oxygen_saturation": 92,
        "cough": 1, "fatigue": 1, "contact_with_case": 1,
    },
    "Exemple : patient sain": {
        "age": 30, "temperature": 36.8, "heart_rate": 68, "oxygen_saturation": 98.5,
        "cough": 0, "fatigue": 0, "contact_with_case": 0,
    },
}

# Couleurs
BG, CARD, HEADER = "#eef2f5", "#ffffff", "#0b5563"
RED, GREEN, GREY, LINE = "#c62828", "#2e7d32", "#5f6b76", "#dfe5ea"
FONT = "Helvetica"


def _spec(name):
    return FIELDS.get(name, (name, "num", float("-inf"), float("inf"), "0"))


# ---------------------------------------------------------------------------
# Interface graphique
# ---------------------------------------------------------------------------
_Base = tk.Tk if tk else object


class ClinicalInterface(_Base):
    def __init__(self, predictor):
        if tk is None:
            raise RuntimeError("Tkinter n'est pas disponible sur cette installation de Python.")
        super().__init__()
        self.predictor = predictor
        self.vars = {}
        self.consultations = 0

        self.title("virus_diag - Interface clinique")
        self.configure(bg=BG)
        self.minsize(900, 600)
        self._setup_style()
        self._build_header()

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=18)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_form(body)
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self._build_result(right)
        self._build_history(right)

        self.reset()
        self.bind("<Return>", self.diagnose)
        self._center()

    # ----- style -----------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Primary.TButton", background=HEADER, foreground="white",
                        font=(FONT, 11, "bold"), padding=(16, 9), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#0e6b7c"), ("pressed", "#08414c")])
        style.configure("Secondary.TButton", font=(FONT, 10), padding=(12, 8))
        style.configure("Card.TCheckbutton", background=CARD, font=(FONT, 11))
        style.map("Card.TCheckbutton", background=[("active", CARD)])
        style.configure("TEntry", padding=5)
        style.configure("Treeview", rowheight=26, font=(FONT, 10), borderwidth=0)
        style.configure("Treeview.Heading", font=(FONT, 10, "bold"))
        for name, color in (("Red", RED), ("Green", GREEN), ("Grey", "#b0bac2")):
            style.configure(f"{name}.Horizontal.TProgressbar", troughcolor="#e3e8ec",
                            background=color, bordercolor="#e3e8ec", thickness=18)

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        tk.Label(frame, text=title, bg=CARD, fg=HEADER, font=(FONT, 13, "bold"),
                 anchor="w").pack(fill="x", padx=18, pady=(14, 6))
        return frame

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        x = (self.winfo_screenwidth() - w) // 2
        y = max(0, (self.winfo_screenheight() - h) // 3)
        self.geometry(f"+{x}+{y}")

    # ----- construction ----------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self, bg=HEADER)
        header.pack(fill="x")
        tk.Label(header, text="virus_diag", bg=HEADER, fg="white",
                 font=(FONT, 20, "bold")).pack(side="left", padx=(20, 10), pady=14)
        model_name = type(self.predictor.model).__name__
        tk.Label(header, text=f"Aide au diagnostic  |  modèle : {model_name}",
                 bg=HEADER, fg="#bfe3ea", font=(FONT, 11)).pack(side="left", pady=14)

    def _build_form(self, parent):
        card = self._card(parent, "Dossier patient")
        card.grid(row=0, column=0, sticky="ns")

        names = self.predictor.feature_names
        numeric = [n for n in names if _spec(n)[1] == "num"]
        flags = [n for n in names if _spec(n)[1] == "bool"]

        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=18)
        for row, name in enumerate(numeric):
            label, _, lo, hi, default = _spec(name)
            var = tk.StringVar(value=default)
            self.vars[name] = var
            tk.Label(grid, text=label, bg=CARD, font=(FONT, 11), anchor="w").grid(
                row=row, column=0, sticky="w", pady=6)
            ttk.Entry(grid, textvariable=var, width=9, justify="right", font=(FONT, 11)).grid(
                row=row, column=1, padx=(14, 8), pady=6)
            hint = f"{lo:g} - {hi:g}" if hi != float("inf") else ""
            tk.Label(grid, text=hint, bg=CARD, fg=GREY, font=(FONT, 9)).grid(
                row=row, column=2, sticky="w")

        if flags:
            tk.Frame(card, bg=LINE, height=1).pack(fill="x", padx=18, pady=(10, 8))
            tk.Label(card, text="Symptômes et contexte", bg=CARD, fg=GREY,
                     font=(FONT, 10, "bold"), anchor="w").pack(fill="x", padx=18)
            for name in flags:
                var = tk.IntVar(value=0)
                self.vars[name] = var
                ttk.Checkbutton(card, text=_spec(name)[0], variable=var,
                                style="Card.TCheckbutton").pack(anchor="w", padx=18, pady=4)

        self.error_var = tk.StringVar()
        tk.Label(card, textvariable=self.error_var, bg=CARD, fg=RED, font=(FONT, 10),
                 wraplength=300, justify="left", anchor="w").pack(fill="x", padx=18, pady=(10, 0))

        buttons = tk.Frame(card, bg=CARD)
        buttons.pack(fill="x", padx=18, pady=(10, 18))
        ttk.Button(buttons, text="Diagnostiquer", style="Primary.TButton",
                   command=self.diagnose).pack(fill="x")
        row = tk.Frame(buttons, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Réinitialiser", style="Secondary.TButton",
                   command=self.reset).pack(side="left", expand=True, fill="x", padx=(0, 4))
        menu_button = ttk.Menubutton(row, text="Exemples", style="Secondary.TButton")
        menu = tk.Menu(menu_button, tearoff=0)
        for title in EXAMPLES:
            menu.add_command(label=title, command=lambda t=title: self.load_example(t))
        menu_button["menu"] = menu
        menu_button.pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _build_result(self, parent):
        card = self._card(parent, "Résultat")
        card.grid(row=0, column=0, sticky="ew")
        self.result_label = tk.Label(card, text="-", bg=CARD, fg=GREY, font=(FONT, 38, "bold"))
        self.result_label.pack(pady=(4, 6))
        self.progress = ttk.Progressbar(card, maximum=100, length=360,
                                        style="Grey.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=40)
        self.proba_label = tk.Label(card, text="", bg=CARD, fg=GREY, font=(FONT, 12))
        self.proba_label.pack(pady=(8, 2))
        tk.Label(card, text=f"Seuil de décision : {self.predictor.threshold:.2f}",
                 bg=CARD, fg=GREY, font=(FONT, 9)).pack()
        tk.Label(card, text="Outil d'aide à la décision : ne remplace pas l'avis d'un médecin.",
                 bg=CARD, fg=GREY, font=(FONT, 9, "italic")).pack(pady=(4, 16))

    def _build_history(self, parent):
        card = self._card(parent, "Historique des consultations")
        card.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        columns = ("n", "heure", "age", "temp", "spo2", "diag", "proba")
        titles = ("N°", "Heure", "Âge", "T °C", "SpO2 %", "Diagnostic", "Probabilité")
        widths = (40, 70, 50, 60, 70, 100, 90)
        self.history = ttk.Treeview(card, columns=columns, show="headings", height=6)
        for col, title, width in zip(columns, titles, widths):
            self.history.heading(col, text=title)
            self.history.column(col, width=width, anchor="center")
        self.history.tag_configure("inf", foreground=RED)
        self.history.tag_configure("sain", foreground=GREEN)
        self.history.pack(fill="both", expand=True, padx=18, pady=(0, 6))
        ttk.Button(card, text="Effacer l'historique", style="Secondary.TButton",
                   command=self.clear_history).pack(anchor="e", padx=18, pady=(0, 14))

    # ----- logique ---------------------------------------------------------
    def read_patient(self):
        """Lit et valide le formulaire ; lève ValueError avec un message lisible."""
        patient = {}
        for name in self.predictor.feature_names:
            label, kind, lo, hi, _ = _spec(name)
            if kind == "bool":
                patient[name] = float(self.vars[name].get())
                continue
            raw = self.vars[name].get().strip().replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"« {label} » doit être un nombre.") from None
            if not lo <= value <= hi:
                raise ValueError(f"« {label} » doit être compris entre {lo:g} et {hi:g}.")
            patient[name] = value
        return patient

    def diagnose(self, event=None):
        """Valide la saisie, interroge le ClinicalPredictor et affiche le résultat."""
        try:
            patient = self.read_patient()
        except ValueError as exc:
            self.error_var.set(str(exc))
            self._clear_result("Saisie invalide : corrigez les champs puis relancez.")
            return None
        self.error_var.set("")
        proba = self.predictor.predict_proba(patient)
        label = self.predictor.label(proba)
        self._show_result(label, proba, patient)
        return label, proba

    def _show_result(self, label, proba, patient):
        infected = label == self.predictor.LABEL_INFECTED
        color = RED if infected else GREEN
        self.result_label.configure(text=label, fg=color)
        self.progress.configure(style=f"{'Red' if infected else 'Green'}.Horizontal.TProgressbar")
        self.progress["value"] = proba * 100
        self.proba_label.configure(text=f"Probabilité d'infection : {proba:.1%}", fg=color)

        self.consultations += 1
        self.history.insert("", 0, tags=("inf" if infected else "sain",), values=(
            self.consultations,
            datetime.now().strftime("%H:%M:%S"),
            f"{patient.get('age', float('nan')):g}",
            f"{patient.get('temperature', float('nan')):g}",
            f"{patient.get('oxygen_saturation', float('nan')):g}",
            label,
            f"{proba:.1%}",
        ))

    def reset(self):
        for name, var in self.vars.items():
            default = _spec(name)[4]
            var.set(default if isinstance(var, tk.StringVar) else int(default))
        self.error_var.set("")
        self._clear_result("Saisissez les données puis cliquez sur « Diagnostiquer »")

    def _clear_result(self, message):
        """Grise le résultat affiché pour ne jamais montrer un diagnostic périmé."""
        self.result_label.configure(text="-", fg=GREY)
        self.progress.configure(style="Grey.Horizontal.TProgressbar")
        self.progress["value"] = 0
        self.proba_label.configure(text=message, fg=GREY)

    def load_example(self, title):
        for name, value in EXAMPLES[title].items():
            if name in self.vars:
                var = self.vars[name]
                var.set(f"{value:g}" if isinstance(var, tk.StringVar) else int(value))
        self.diagnose()

    def clear_history(self):
        self.history.delete(*self.history.get_children())
        self.consultations = 0


def run_gui(predictor):
    """Ouvre la fenêtre de l'interface clinique."""
    if tk is None:
        raise RuntimeError("Tkinter n'est pas disponible : utilisez `python main.py cli` ou `serve`.")
    try:
        app = ClinicalInterface(predictor)
    except tk.TclError as exc:
        raise RuntimeError(f"Impossible d'ouvrir la fenêtre (pas d'affichage ?) : {exc}") from exc
    app.mainloop()


# ---------------------------------------------------------------------------
# Version texte (ligne de commande)
# ---------------------------------------------------------------------------
PROMPTS = {
    name: spec[0] + (" ? (1 = oui, 0 = non)" if spec[1] == "bool" else "")
    for name, spec in FIELDS.items()
}


def ask_float(prompt, input_fn=input, output_fn=print):
    while True:
        raw = input_fn(f"  {prompt} : ").strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            output_fn("  Valeur invalide, veuillez saisir un nombre.")


def run(predictor, input_fn=input, output_fn=print):
    """Boucle de consultation en texte. input_fn/output_fn sont injectables (tests)."""
    output_fn("=== Interface clinique - virus_diag ===")
    while True:
        output_fn("\nNouveau patient :")
        patient = {
            name: ask_float(PROMPTS.get(name, name), input_fn, output_fn)
            for name in predictor.feature_names
        }
        proba = predictor.predict_proba(patient)
        output_fn(
            f"\n>>> Diagnostic : {predictor.label(proba)} "
            f"(probabilité d'infection : {proba:.1%})"
        )
        again = input_fn("\nAutre patient ? (o/n) : ").strip().lower()
        if again not in ("o", "oui", "y", "yes"):
            output_fn("Fin de la consultation.")
            break
