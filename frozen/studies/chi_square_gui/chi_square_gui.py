"""一个用于观察卡方分布形状的极简交互窗口。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.stats import ncx2


def run_chi_square_window() -> None:
    """打开窗口；上方画图，下方用滑动条或输入框修改参数。"""
    root = tk.Tk()
    root.title("卡方分布交互图")
    root.geometry("900x680")

    figure = Figure(figsize=(8.5, 5.0), dpi=100)
    axis = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    controls = ttk.Frame(root, padding=10)
    controls.pack(side=tk.BOTTOM, fill=tk.X)
    status = ttk.Label(controls, text="λ = 0 时就是普通卡方分布")
    status.grid(row=3, column=0, columnspan=3, pady=(8, 0))

    settings = {
        "degrees_of_freedom": {
            "label": "自由度 k",
            "minimum": 0.2,
            "maximum": 30.0,
            "resolution": 0.1,
            "initial": 2.0,
        },
        "noncentrality": {
            "label": "非中心参数 λ",
            "minimum": 0.0,
            "maximum": 30.0,
            "resolution": 0.1,
            "initial": 0.0,
        },
        "x_maximum": {
            "label": "横轴最大值",
            "minimum": 5.0,
            "maximum": 100.0,
            "resolution": 1.0,
            "initial": 20.0,
        },
    }
    sliders: dict[str, tk.Scale] = {}
    entries = {
        name: tk.StringVar(value=str(specification["initial"]))
        for name, specification in settings.items()
    }

    def draw() -> None:
        try:
            degrees_of_freedom = float(entries["degrees_of_freedom"].get())
            noncentrality = float(entries["noncentrality"].get())
            x_maximum = float(entries["x_maximum"].get())
            if degrees_of_freedom <= 0.0 or noncentrality < 0.0 or x_maximum <= 0.0:
                raise ValueError
        except ValueError:
            status.configure(text="请输入：k > 0，λ ≥ 0，横轴最大值 > 0", foreground="red")
            return

        x = np.linspace(1e-4, x_maximum, 1000)
        density = ncx2.pdf(x, degrees_of_freedom, noncentrality)
        axis.clear()
        axis.plot(x, density, color="tab:blue", linewidth=2)
        axis.set_xlim(0.0, x_maximum)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel("x")
        axis.set_ylabel("Probability density")
        axis.set_title(
            rf"$\chi^2$ distribution: $k={degrees_of_freedom:.2f}$, "
            rf"$\lambda={noncentrality:.2f}$"
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        canvas.draw_idle()
        status.configure(text="λ = 0 时就是普通卡方分布", foreground="black")

    def slider_changed(name: str, value: str) -> None:
        number = float(value)
        resolution = settings[name]["resolution"]
        entries[name].set(f"{number:.1f}" if resolution < 1.0 else f"{number:.0f}")
        draw()

    def entry_changed(name: str) -> None:
        try:
            value = float(entries[name].get())
        except ValueError:
            draw()
            return
        sliders[name].set(value)
        draw()

    for row, (name, specification) in enumerate(settings.items()):
        ttk.Label(controls, text=specification["label"], width=16).grid(
            row=row, column=0, sticky="w"
        )
        slider = tk.Scale(
            controls,
            from_=specification["minimum"],
            to=specification["maximum"],
            resolution=specification["resolution"],
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=570,
            command=lambda value, active_name=name: slider_changed(active_name, value),
        )
        sliders[name] = slider
        slider.set(specification["initial"])
        slider.grid(row=row, column=1, padx=8, sticky="ew")
        entry = ttk.Entry(controls, textvariable=entries[name], width=10)
        entry.grid(row=row, column=2)
        entry.bind("<Return>", lambda _event, active_name=name: entry_changed(active_name))
        entry.bind("<FocusOut>", lambda _event, active_name=name: entry_changed(active_name))

    controls.columnconfigure(1, weight=1)
    draw()
    root.mainloop()


if __name__ == "__main__":
    run_chi_square_window()
