import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from sympy import Eq, latex, solve, sympify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ASTRO_PRESETS = [
    ("Distance modulus", "m - M = 5*log(d/10, 10)", "d"),
    ("Stefan-Boltzmann law", "L = 4*pi*R^2*sigma*T^4", "T"),
    ("Orbital velocity", "v = sqrt(G*M/r)", "r"),
    ("Escape velocity", "v_e = sqrt(2*G*M/r)", "r"),
    ("Wien's law", "lambda_max = b/T", "T"),
    ("Kepler's third law", "P^2 = 4*pi^2*a^3/(G*(M + m))", "a"),
]


def parse_math_expression(expr_text: str):
    return parse_expr(expr_text, transformations=TRANSFORMATIONS, evaluate=False)


def parse_equation(equation_text: str):
    if "=" not in equation_text:
        raise ValueError("Equation must contain '='.")

    left_text, right_text = equation_text.split("=", 1)
    left_expr = parse_math_expression(left_text.strip())
    right_expr = parse_math_expression(right_text.strip())
    return Eq(left_expr, right_expr)


def rearrange_equation(equation_text: str, variable_text: str):
    if not variable_text.strip():
        raise ValueError("Please enter a variable to solve for.")

    eq = parse_equation(equation_text)
    var = sympify(variable_text.strip())
    solutions = solve(eq, var, dict=False)

    if not solutions:
        raise ValueError(f"Could not solve the equation for '{variable_text}'.")

    return eq, var, solutions


class LatexPreview(ttk.LabelFrame):
    def __init__(self, parent, title="LaTeX Preview"):
        super().__init__(parent, text=title, padding=8)

        self.figure = Figure(figsize=(6, 1.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.set_latex(r"\mathrm{Preview\ will\ appear\ here}")

    def set_latex(self, latex_string: str):
        self.ax.clear()
        self.ax.axis("off")
        text = latex_string if latex_string else r"\mathrm{Preview\ unavailable}"

        try:
            self.ax.text(
                0.02,
                0.5,
                f"${text}$",
                fontsize=20,
                va="center",
                ha="left",
                wrap=True,
            )
        except Exception:
            self.ax.text(0.02, 0.5, "Preview unavailable", fontsize=16, va="center", ha="left")

        self.canvas.draw_idle()


class SimplePairDialog(tk.Toplevel):
    def __init__(self, parent, title, label1, default1, label2, default2):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self.transient(parent)
        self.grab_set()

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=label1).grid(row=0, column=0, sticky="w", pady=4)
        self.entry1 = ttk.Entry(frame, width=28)
        self.entry1.grid(row=0, column=1, pady=4)
        self.entry1.insert(0, default1)

        ttk.Label(frame, text=label2).grid(row=1, column=0, sticky="w", pady=4)
        self.entry2 = ttk.Entry(frame, width=28)
        self.entry2.grid(row=1, column=1, pady=4)
        self.entry2.insert(0, default2)

        btns = ttk.Frame(frame)
        btns.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="right", padx=(0, 6))

        self.entry1.focus_set()
        self.bind("<Return>", lambda _e: self.on_ok())
        self.bind("<Escape>", lambda _e: self.destroy())

    def on_ok(self):
        self.result = (self.entry1.get().strip(), self.entry2.get().strip())
        self.destroy()


class BuilderEditor(ttk.LabelFrame):
    def __init__(self, parent, on_change_callback=None):
        super().__init__(parent, text="Visual Equation Builder", padding=10)
        self.on_change_callback = on_change_callback
        self.segments = []
        self.insert_into_equation_callback = lambda: None
        self._build_ui()

    def _build_ui(self):
        ttk.Label(
            self,
            text=(
                "Build expressions from blocks. Click Insert into Equation to append the built "
                "expression to the main equation box."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        builder = ttk.Frame(self)
        builder.pack(fill="x", pady=(0, 8))

        self.token_entry = ttk.Entry(builder)
        self.token_entry.grid(row=0, column=0, columnspan=4, sticky="ew", padx=(0, 6), pady=4)
        self.token_entry.insert(0, "x")

        ttk.Button(builder, text="Add token", command=self.add_token).grid(row=0, column=4, sticky="ew", pady=4)
        ttk.Button(builder, text="+", command=lambda: self.add_raw(" + ")).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="-", command=lambda: self.add_raw(" - ")).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="×", command=lambda: self.add_raw("*")).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="÷", command=lambda: self.add_raw("/")).grid(row=1, column=3, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="=", command=lambda: self.add_raw(" = ")).grid(row=1, column=4, sticky="ew", padx=2, pady=2)

        ttk.Button(builder, text="( )", command=self.add_parentheses).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="sqrt( )", command=self.add_sqrt).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="a^b", command=self.add_power).grid(row=2, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="a/b", command=self.add_fraction).grid(row=2, column=3, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="subscript", command=self.add_subscript).grid(row=2, column=4, sticky="ew", padx=2, pady=2)

        ttk.Button(builder, text="sin( )", command=lambda: self.add_function("sin")).grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="cos( )", command=lambda: self.add_function("cos")).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="tan( )", command=lambda: self.add_function("tan")).grid(row=3, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="log( )", command=lambda: self.add_function("log")).grid(row=3, column=3, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="exp( )", command=lambda: self.add_function("exp")).grid(row=3, column=4, sticky="ew", padx=2, pady=2)

        ttk.Button(builder, text="π", command=lambda: self.add_raw("pi")).grid(row=4, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="θ", command=lambda: self.add_raw("theta")).grid(row=4, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="Δ", command=lambda: self.add_raw("Delta")).grid(row=4, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="λ", command=lambda: self.add_raw("lambda")).grid(row=4, column=3, sticky="ew", padx=2, pady=2)
        ttk.Button(builder, text="μ", command=lambda: self.add_raw("mu")).grid(row=4, column=4, sticky="ew", padx=2, pady=2)

        for i in range(5):
            builder.columnconfigure(i, weight=1)

        presets = ttk.LabelFrame(self, text="Physics / Maths Templates", padding=8)
        presets.pack(fill="x", pady=(0, 8))

        preset_buttons = [
            ("E = m*c^2", "E = m*c^2"),
            ("v = sqrt(G*M/r)", "v = sqrt(G*M/r)"),
            ("F = m*a", "F = m*a"),
            ("p = m*v", "p = m*v"),
            ("V = I*R", "V = I*R"),
            ("P*V = n*R*T", "P*V = n*R*T"),
            ("quadratic", "x = (-b + sqrt(b^2 - 4*a*c))/(2*a)"),
            ("distance modulus", "m - M = 5*log(d/10, 10)"),
            ("Stefan-Boltzmann", "L = 4*pi*R^2*sigma*T^4"),
            ("orbital velocity", "v = sqrt(G*M/r)"),
            ("escape velocity", "v_e = sqrt(2*G*M/r)"),
            ("Wien's law", "lambda_max = b/T"),
            ("Kepler's third law", "P^2 = 4*pi^2*a^3/(G*(M + m))"),
        ]

        preset_grid = ttk.Frame(presets)
        preset_grid.pack(fill="x")
        for i, (label, content) in enumerate(preset_buttons):
            ttk.Button(
                preset_grid,
                text=label,
                command=lambda c=content: self.load_expression(c),
            ).grid(row=i // 3, column=i % 3, sticky="ew", padx=3, pady=3)
        for i in range(3):
            preset_grid.columnconfigure(i, weight=1)

        preview_row = ttk.Frame(self)
        preview_row.pack(fill="both", expand=True)

        left = ttk.Frame(preview_row)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(preview_row)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(left, text="Builder expression (SymPy style):").pack(anchor="w")
        self.builder_text = tk.Text(left, height=6, wrap="word", font=("Consolas", 11))
        self.builder_text.pack(fill="both", expand=True)
        self.builder_text.configure(state="disabled")

        self.preview = LatexPreview(right, title="Builder Preview")
        self.preview.pack(fill="both", expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Undo", command=self.undo).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Clear Builder", command=self.clear).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Insert into Equation", command=self.insert_into_equation_callback).pack(side="left")

    def _notify_change(self):
        self.refresh_preview()
        if self.on_change_callback:
            self.on_change_callback(self.get_expression())

    def _update_builder_text(self, text: str):
        self.builder_text.configure(state="normal")
        self.builder_text.delete("1.0", tk.END)
        self.builder_text.insert(tk.END, text)
        self.builder_text.configure(state="disabled")

    def add_token(self):
        token = self.token_entry.get().strip() or "x"
        self.segments.append(token)
        self._notify_change()

    def add_raw(self, token: str):
        self.segments.append(token)
        self._notify_change()

    def add_parentheses(self):
        token = self.token_entry.get().strip() or "x"
        self.segments.append(f"({token})")
        self._notify_change()

    def add_sqrt(self):
        token = self.token_entry.get().strip() or "x"
        self.segments.append(f"sqrt({token})")
        self._notify_change()

    def add_power(self):
        token = self.token_entry.get().strip() or "x"
        popup = SimplePairDialog(self, "Power Builder", "Base:", token, "Exponent:", "2")
        self.wait_window(popup)
        if popup.result:
            base, exponent = popup.result
            self.segments.append(f"({base})^({exponent})")
            self._notify_change()

    def add_fraction(self):
        popup = SimplePairDialog(self, "Fraction Builder", "Numerator:", "a", "Denominator:", "b")
        self.wait_window(popup)
        if popup.result:
            numerator, denominator = popup.result
            self.segments.append(f"({numerator})/({denominator})")
            self._notify_change()

    def add_subscript(self):
        popup = SimplePairDialog(self, "Subscript Builder", "Base symbol:", "x", "Subscript:", "1")
        self.wait_window(popup)
        if popup.result:
            base, sub = popup.result
            self.segments.append(f"{base}_{sub}")
            self._notify_change()

    def add_function(self, fn_name: str):
        token = self.token_entry.get().strip() or "x"
        self.segments.append(f"{fn_name}({token})")
        self._notify_change()

    def load_expression(self, text: str):
        self.segments = [text]
        self._notify_change()

    def undo(self):
        if self.segments:
            self.segments.pop()
            self._notify_change()

    def clear(self):
        self.segments = []
        self._notify_change()

    def get_expression(self) -> str:
        return "".join(self.segments).strip()

    def refresh_preview(self):
        expression = self.get_expression()
        self._update_builder_text(expression)

        if not expression:
            self.preview.set_latex(r"\mathrm{Build\ an\ expression}")
            return

        try:
            if "=" in expression:
                eq = parse_equation(expression)
                self.preview.set_latex(latex(eq))
            else:
                expr = parse_math_expression(expression)
                self.preview.set_latex(latex(expr))
        except Exception:
            self.preview.set_latex(r"\mathrm{Invalid\ expression}")


class MathBlockBuilder(ttk.LabelFrame):
    def __init__(self, parent, on_insert_expression=None, on_use_equation=None):
        super().__init__(parent, text="Math Block Builder", padding=10)
        self.on_insert_expression = on_insert_expression
        self.on_use_equation = on_use_equation
        self._build_ui()

    def _build_ui(self):
        ttk.Label(
            self,
            text="Build structured maths blocks, preview them, then insert them into the equation.",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        chooser = ttk.Frame(self)
        chooser.pack(fill="x", pady=(0, 8))
        ttk.Label(chooser, text="Block type:").pack(side="left")
        self.block_var = tk.StringVar(value="fraction")
        combo = ttk.Combobox(
            chooser,
            textvariable=self.block_var,
            state="readonly",
            values=["fraction", "power", "root", "subscript", "function", "token", "equation"],
            width=16,
        )
        combo.pack(side="left", padx=(8, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.show_editor())

        self.editor_holder = ttk.Frame(self)
        self.editor_holder.pack(fill="x", pady=(0, 8))

        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=True)
        left = ttk.Frame(bottom)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(bottom)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(left, text="Generated expression:").pack(anchor="w")
        self.expression_box = tk.Text(left, height=6, wrap="word", font=("Consolas", 11))
        self.expression_box.pack(fill="both", expand=True)
        self.expression_box.configure(state="disabled")

        self.preview = LatexPreview(right, title="Block Preview")
        self.preview.pack(fill="both", expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Insert Expression", command=self.insert_expression).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Use as Equation", command=self.use_as_equation).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Copy Expression", command=self.copy_expression).pack(side="left")

        self.show_editor()

    def _make_labeled_entry(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        entry.bind("<KeyRelease>", lambda _e: self.refresh_preview())
        return entry

    def show_editor(self):
        for child in self.editor_holder.winfo_children():
            child.destroy()

        kind = self.block_var.get()
        frame = ttk.Frame(self.editor_holder)
        frame.pack(fill="x")
        frame.columnconfigure(1, weight=1)

        if kind == "fraction":
            self.frac_num = tk.StringVar(value="a")
            self.frac_den = tk.StringVar(value="b")
            self._make_labeled_entry(frame, 0, "Numerator:", self.frac_num)
            self._make_labeled_entry(frame, 1, "Denominator:", self.frac_den)
        elif kind == "power":
            self.pow_base = tk.StringVar(value="x")
            self.pow_exp = tk.StringVar(value="2")
            self._make_labeled_entry(frame, 0, "Base:", self.pow_base)
            self._make_labeled_entry(frame, 1, "Exponent:", self.pow_exp)
        elif kind == "root":
            self.root_value = tk.StringVar(value="x")
            self.root_index = tk.StringVar(value="2")
            self._make_labeled_entry(frame, 0, "Radicand:", self.root_value)
            self._make_labeled_entry(frame, 1, "Index:", self.root_index)
        elif kind == "subscript":
            self.sub_base = tk.StringVar(value="x")
            self.sub_index = tk.StringVar(value="1")
            self._make_labeled_entry(frame, 0, "Base symbol:", self.sub_base)
            self._make_labeled_entry(frame, 1, "Subscript:", self.sub_index)
        elif kind == "function":
            self.fn_name = tk.StringVar(value="sin")
            self.fn_arg = tk.StringVar(value="x")
            ttk.Label(frame, text="Function:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
            cb = ttk.Combobox(frame, textvariable=self.fn_name, state="readonly", values=["sin", "cos", "tan", "log", "exp"], width=12)
            cb.grid(row=0, column=1, sticky="w", pady=4)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh_preview())
            self._make_labeled_entry(frame, 1, "Argument:", self.fn_arg)
        elif kind == "token":
            self.token_value = tk.StringVar(value="x")
            self._make_labeled_entry(frame, 0, "Token:", self.token_value)
        elif kind == "equation":
            self.eq_left = tk.StringVar(value="E")
            self.eq_right = tk.StringVar(value="m*c^2")
            self._make_labeled_entry(frame, 0, "Left side:", self.eq_left)
            self._make_labeled_entry(frame, 1, "Right side:", self.eq_right)

        self.refresh_preview()

    def get_expression(self):
        kind = self.block_var.get()
        if kind == "fraction":
            return f"({self.frac_num.get().strip()})/({self.frac_den.get().strip()})"
        if kind == "power":
            return f"({self.pow_base.get().strip()})^({self.pow_exp.get().strip()})"
        if kind == "root":
            value = self.root_value.get().strip()
            index = self.root_index.get().strip()
            if index == "2":
                return f"sqrt({value})"
            return f"({value})^(1/({index}))"
        if kind == "subscript":
            return f"{self.sub_base.get().strip()}_{self.sub_index.get().strip()}"
        if kind == "function":
            return f"{self.fn_name.get().strip()}({self.fn_arg.get().strip()})"
        if kind == "token":
            return self.token_value.get().strip()
        if kind == "equation":
            return f"{self.eq_left.get().strip()} = {self.eq_right.get().strip()}"
        return ""

    def refresh_preview(self):
        expression = self.get_expression()
        self.expression_box.configure(state="normal")
        self.expression_box.delete("1.0", tk.END)
        self.expression_box.insert(tk.END, expression)
        self.expression_box.configure(state="disabled")

        if not expression:
            self.preview.set_latex(r"\mathrm{Build\ a\ block}")
            return

        try:
            if self.block_var.get() == "equation" or "=" in expression:
                self.preview.set_latex(latex(parse_equation(expression)))
            else:
                self.preview.set_latex(latex(parse_math_expression(expression)))
        except Exception:
            self.preview.set_latex(r"\mathrm{Invalid\ expression}")

    def insert_expression(self):
        expression = self.get_expression()
        if expression and self.on_insert_expression:
            self.on_insert_expression(expression)

    def use_as_equation(self):
        expression = self.get_expression()
        if expression and self.on_use_equation:
            self.on_use_equation(expression)

    def copy_expression(self):
        expression = self.get_expression()
        if not expression:
            messagebox.showinfo("Math Block Builder", "Nothing to copy yet.")
            return
        self.clipboard_clear()
        self.clipboard_append(expression)
        self.update()
        messagebox.showinfo("Copied", "Block expression copied to clipboard.")


class EquationRearrangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Equation Rearranger V8")
        self.root.geometry("1450x950")
        self.current_result_latex = ""
        self.current_input_latex = ""
        self._build_ui()
        self.load_example_1()
        self.update_input_preview()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        header = ttk.Label(
            main,
            text="Equation Rearranger V8 — Astronomy Presets + Math Blocks + LaTeX Preview",
            font=("Segoe UI", 16, "bold"),
        )
        header.pack(anchor="w", pady=(0, 10))

        top = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        top.pack(fill="both", expand=True)

        left = ttk.Frame(top, padding=4)
        center = ttk.Frame(top, padding=4)
        right = ttk.Frame(top, padding=4)

        top.add(left, weight=3)
        top.add(center, weight=4)
        top.add(right, weight=3)

        self._build_left(left)
        self._build_center(center)
        self._build_right(right)

    def _build_left(self, parent):
        input_frame = ttk.LabelFrame(parent, text="Equation Input", padding=10)
        input_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(input_frame, text="Equation:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        self.equation_entry = ttk.Entry(input_frame, font=("Consolas", 12))
        self.equation_entry.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(input_frame, text="Solve for:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        self.variable_entry = ttk.Entry(input_frame, font=("Consolas", 12), width=18)
        self.variable_entry.grid(row=1, column=1, sticky="w", pady=5)

        input_frame.columnconfigure(1, weight=1)
        self.equation_entry.bind("<KeyRelease>", lambda _e: self.update_input_preview())
        self.variable_entry.bind("<KeyRelease>", lambda _e: self.update_input_preview())

        preset_frame = ttk.LabelFrame(parent, text="Astronomy Presets", padding=10)
        preset_frame.pack(fill="x", pady=(0, 8))

        for i, (label, equation, solve_for) in enumerate(ASTRO_PRESETS):
            ttk.Button(
                preset_frame,
                text=label,
                command=lambda e=equation, s=solve_for: self.load_preset(e, s),
            ).grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="ew")
        for i in range(2):
            preset_frame.columnconfigure(i, weight=1)

        quick = ttk.LabelFrame(parent, text="Quick Insert", padding=10)
        quick.pack(fill="x", pady=(0, 8))

        symbols = [
            ("=", "="), ("+", "+"), ("-", "-"), ("*", "*"), ("/", "/"), ("^", "^"),
            ("(", "("), (")", ")"), ("sqrt()", "sqrt()"), ("sin()", "sin()"),
            ("cos()", "cos()"), ("log()", "log()"), ("pi", "pi"), ("theta", "theta"),
            ("x", "x"), ("y", "y"), ("z", "z"), ("m", "m"), ("r", "r"), ("v", "v"),
            ("c", "c"), ("G", "G"), ("M", "M"), ("R", "R"),
        ]

        quick_grid = ttk.Frame(quick)
        quick_grid.pack(fill="x")
        for i, (label, text) in enumerate(symbols):
            ttk.Button(
                quick_grid,
                text=label,
                command=lambda t=text: self.insert_into_equation(t),
                width=10,
            ).grid(row=i // 6, column=i % 6, padx=2, pady=2, sticky="ew")
        for i in range(6):
            quick_grid.columnconfigure(i, weight=1)

        self.input_preview = LatexPreview(parent, title="Input Equation Preview")
        self.input_preview.pack(fill="both", expand=True, pady=(0, 8))

        controls = ttk.LabelFrame(parent, text="Actions", padding=10)
        controls.pack(fill="x")
        ttk.Button(controls, text="Rearrange", command=self.on_rearrange).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Copy Result LaTeX", command=self.copy_result_latex).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Example 1", command=self.load_example_1).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Example 2", command=self.load_example_2).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Export .tex", command=self.export_result_tex).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Clear All", command=self.clear_all).pack(side="left")

    def _build_center(self, parent):
        self.math_blocks = MathBlockBuilder(
            parent,
            on_insert_expression=self.insert_builder_expression_text,
            on_use_equation=self.use_expression_as_equation,
        )
        self.math_blocks.pack(fill="both", expand=True, pady=(0, 8))

        self.builder = BuilderEditor(parent, on_change_callback=self.on_builder_change)
        self.builder.pack(fill="both", expand=True)
        self.builder.insert_into_equation_callback = self.insert_builder_expression

    def _build_right(self, parent):
        self.result_preview = LatexPreview(parent, title="Result Preview")
        self.result_preview.pack(fill="both", expand=True, pady=(0, 8))

        latex_frame = ttk.LabelFrame(parent, text="LaTeX Output", padding=10)
        latex_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.latex_box = tk.Text(latex_frame, wrap="word", font=("Consolas", 11), height=12)
        self.latex_box.pack(fill="both", expand=True)

        output_frame = ttk.LabelFrame(parent, text="Symbolic Output", padding=10)
        output_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.output_box = tk.Text(output_frame, wrap="word", font=("Consolas", 11), height=12)
        self.output_box.pack(fill="both", expand=True)

        notes = ttk.LabelFrame(parent, text="Notes", padding=10)
        notes.pack(fill="x")
        ttk.Label(
            notes,
            justify="left",
            text=(
                "• Astronomy presets include distance modulus, Stefan-Boltzmann, orbital velocity, "
                "escape velocity, Wien's law, and Kepler's third law.\n"
                "• Use log(x, 10) for base-10 logs such as the distance modulus.\n"
                "• The Math Block Builder creates structured expressions like fractions, powers, roots, and subscripts.\n"
                "• The Visual Equation Builder is still available for free-form composition."
            ),
        ).pack(anchor="w")

    def on_builder_change(self, _expression: str):
        pass

    def load_preset(self, equation: str, solve_for: str):
        self.equation_entry.delete(0, tk.END)
        self.equation_entry.insert(0, equation)
        self.variable_entry.delete(0, tk.END)
        self.variable_entry.insert(0, solve_for)
        self.update_input_preview()

    def insert_into_equation(self, text: str):
        entry = self.equation_entry
        entry.focus_set()
        pos = entry.index(tk.INSERT)
        current = entry.get()
        new_text = current[:pos] + text + current[pos:]
        entry.delete(0, tk.END)
        entry.insert(0, new_text)

        if text.endswith("()"):
            entry.icursor(pos + len(text) - 1)
        else:
            entry.icursor(pos + len(text))

        self.update_input_preview()

    def insert_builder_expression_text(self, expr: str):
        if not expr:
            return

        current = self.equation_entry.get().strip()
        combined = current + expr if current else expr
        self.equation_entry.delete(0, tk.END)
        self.equation_entry.insert(0, combined)
        self.update_input_preview()

    def use_expression_as_equation(self, expr: str):
        if not expr:
            return

        self.equation_entry.delete(0, tk.END)
        self.equation_entry.insert(0, expr)
        self.update_input_preview()

    def insert_builder_expression(self):
        expr = self.builder.get_expression()
        if not expr:
            messagebox.showinfo("Builder", "The builder is empty.")
            return

        current = self.equation_entry.get().strip()
        combined = current + expr if current else expr
        self.equation_entry.delete(0, tk.END)
        self.equation_entry.insert(0, combined)
        self.update_input_preview()

    def update_input_preview(self):
        equation_text = self.equation_entry.get().strip()
        if not equation_text:
            self.input_preview.set_latex(r"\mathrm{Preview\ will\ appear\ here}")
            return

        try:
            eq = parse_equation(equation_text)
            self.current_input_latex = latex(eq)
            self.input_preview.set_latex(self.current_input_latex)
        except Exception:
            try:
                expr = parse_math_expression(equation_text)
                self.input_preview.set_latex(latex(expr))
            except Exception:
                self.input_preview.set_latex(r"\mathrm{Invalid\ equation}")

    def on_rearrange(self):
        equation_text = self.equation_entry.get().strip()
        variable_text = self.variable_entry.get().strip()

        try:
            eq, var, solutions = rearrange_equation(equation_text, variable_text)

            symbolic_lines = ["Original equation:", f"  {eq}", "", f"Solved for {var}:", ""]
            latex_lines = []
            first_result_latex = None

            for i, sol in enumerate(solutions, start=1):
                result_eq = Eq(var, sol)
                result_latex = latex(result_eq)

                symbolic_lines.append(f"Solution {i}:")
                symbolic_lines.append(f"  {result_eq}")
                symbolic_lines.append(f"  LaTeX: {result_latex}")
                symbolic_lines.append("")

                latex_lines.append(result_latex)
                if first_result_latex is None:
                    first_result_latex = result_latex

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, "\n".join(symbolic_lines))

            self.latex_box.delete("1.0", tk.END)
            self.latex_box.insert(tk.END, "\n\n".join(latex_lines))

            self.current_result_latex = "\n\n".join(latex_lines)
            self.result_preview.set_latex(first_result_latex or r"\mathrm{No\ result}")

        except Exception as e:
            self.current_result_latex = ""
            self.output_box.delete("1.0", tk.END)
            self.latex_box.delete("1.0", tk.END)
            self.result_preview.set_latex(r"\mathrm{Error}")
            messagebox.showerror("Error", str(e))

    def export_result_tex(self):
        if not self.current_result_latex:
            messagebox.showinfo("Export .tex", "No LaTeX result available yet.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")],
            initialfile="equation_result.tex",
        )
        if not file_path:
            return

        tex_content = (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "\\begin{document}\n"
            "\\[\n"
            f"{self.current_result_latex.splitlines()[0]}\n"
            "\\]\n"
            "\\end{document}\n"
        )
        Path(file_path).write_text(tex_content, encoding="utf-8")
        messagebox.showinfo("Export .tex", f"Saved to {file_path}")

    def copy_result_latex(self):
        if not self.current_result_latex:
            messagebox.showinfo("Nothing to copy", "No LaTeX result available yet.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_result_latex)
        self.root.update()
        messagebox.showinfo("Copied", "LaTeX copied to clipboard.")

    def load_example_1(self):
        self.load_preset("E = m*c^2", "m")
        self.builder.load_expression("E = m*c^2")

    def load_example_2(self):
        self.load_preset("v = sqrt(G*M/r)", "r")
        self.builder.load_expression("v = sqrt(G*M/r)")

    def clear_all(self):
        self.equation_entry.delete(0, tk.END)
        self.variable_entry.delete(0, tk.END)
        self.output_box.delete("1.0", tk.END)
        self.latex_box.delete("1.0", tk.END)
        self.current_result_latex = ""
        self.current_input_latex = ""
        self.input_preview.set_latex(r"\mathrm{Preview\ will\ appear\ here}")
        self.result_preview.set_latex(r"\mathrm{Result\ preview\ will\ appear\ here}")
        self.builder.clear()
        self.math_blocks.show_editor()


if __name__ == "__main__":
    root = tk.Tk()
    app = EquationRearrangerApp(root)
    root.mainloop()
