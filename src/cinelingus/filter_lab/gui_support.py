from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .models import FilterDefinition, FilterParameter


def render_parameter_controls(
    parent: ttk.Frame,
    definition: FilterDefinition,
    *,
    existing_values: dict[str, Any] | None = None,
) -> dict[str, tk.Variable]:
    for child in parent.winfo_children():
        child.destroy()
    values = definition.parameter_defaults | (existing_values or {})
    variables: dict[str, tk.Variable] = {}
    normal = ttk.Frame(parent)
    normal.grid(row=0, column=0, sticky="ew")
    normal.columnconfigure(1, weight=1)
    advanced = ttk.LabelFrame(parent, text="Advanced", padding=(8, 6))
    advanced.columnconfigure(1, weight=1)
    advanced.grid(row=2, column=0, sticky="ew", pady=(6, 0))
    advanced.grid_remove()
    advanced_toggle = tk.BooleanVar(value=False)

    def toggle() -> None:
        if advanced_toggle.get():
            advanced.grid()
        else:
            advanced.grid_remove()

    normal_row = 0
    advanced_row = 0
    for parameter in definition.parameters:
        target = advanced if parameter.advanced else normal
        row = advanced_row if parameter.advanced else normal_row
        variable = _variable(parent, parameter, values.get(parameter.id, parameter.default))
        variables[parameter.id] = variable
        ttk.Label(target, text=parameter.label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        widget = _control(target, parameter, variable)
        widget.grid(row=row, column=1, sticky="w", pady=3)
        ttk.Label(target, text=parameter.description, style="Hint.TLabel", wraplength=480).grid(row=row, column=2, sticky="w", padx=(10, 0), pady=3)
        if parameter.advanced:
            advanced_row += 1
        else:
            normal_row += 1
    if advanced_row:
        ttk.Checkbutton(parent, text="Show advanced controls", variable=advanced_toggle, command=toggle).grid(row=1, column=0, sticky="w", pady=(6, 0))
    if not definition.parameters:
        ttk.Label(normal, text="This apparatus uses its established defaults.", style="Hint.TLabel").grid(row=0, column=0, sticky="w")
    return variables


def collect_parameter_values(definition: FilterDefinition, variables: dict[str, tk.Variable]) -> dict[str, Any]:
    raw = {parameter.id: variables[parameter.id].get() for parameter in definition.parameters if parameter.id in variables}
    return definition.normalize_parameters(raw)


def _variable(parent: ttk.Frame, parameter: FilterParameter, value: Any) -> tk.Variable:
    if parameter.kind == "boolean":
        return tk.BooleanVar(parent, value=bool(value))
    return tk.StringVar(parent, value=str(value))


def _control(parent: ttk.Frame, parameter: FilterParameter, variable: tk.Variable) -> ttk.Widget:
    if parameter.kind == "boolean":
        return ttk.Checkbutton(parent, variable=variable)
    if parameter.kind == "choice":
        return ttk.Combobox(parent, textvariable=variable, values=parameter.choices, state="readonly", width=28)
    return ttk.Entry(parent, textvariable=variable, width=30)
