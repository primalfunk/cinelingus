from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tkinter as tk

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - exercised only in minimal installations
    Image = None
    ImageTk = None


PLATE_WIDTH = 1536
PLATE_HEIGHT = 1024
INSTRUMENT_PLATE_RELATIVE_PATH = Path("assets") / "instrument_plate.png"

INSTRUMENT_COLORS = {
    "surface": "#101719",
    "surface_deep": "#080d0f",
    "surface_raised": "#1b2325",
    "brass": "#b89a5d",
    "brass_dim": "#6f6043",
    "brass_bright": "#d8bd79",
    "cyan": "#83d8e8",
    "cyan_bright": "#c9ffff",
    "text": "#e8e1d2",
    "muted": "#879294",
    "dormant": "#4d554f",
    "amber": "#d09a48",
    "red": "#b9584f",
}

INSTRUMENT_SPACING = {"hairline": 1, "micro": 4, "unit": 8, "compact": 12, "panel": 16, "large": 24, "major": 32}
INSTRUMENT_DIMENSIONS = {"compact_height": 24, "control_height": 28, "standard_height": 32, "actuator_height": 60}
INSTRUMENT_FONTS = {
    "display": ("Georgia", 10, "bold"),
    "display_large": ("Georgia", 13, "bold"),
    "technical": ("Consolas", 9),
    "caps": ("Segoe UI", 7, "bold"),
}

CONTROL_STATES = ("normal", "hover", "focused", "pressed", "selected", "disabled", "active", "warning", "failed")


def ellipsize_text(value: str, maximum: int = 34) -> str:
    text = str(value)
    if len(text) <= maximum:
        return text
    keep = max(4, (maximum - 1) // 2)
    return f"{text[:keep]}…{text[-keep:]}"


def concise_material_name(value: str, *, empty: str = "EMPTY CHAMBER") -> str:
    text = str(value or "").strip()
    return ellipsize_text(Path(text).name, 38) if text else empty


def control_palette(state: str, role: str = "key") -> dict[str, str]:
    if state not in CONTROL_STATES:
        raise ValueError(f"Unknown machine control state: {state}")
    border = INSTRUMENT_COLORS["brass_dim"]
    foreground = INSTRUMENT_COLORS["text"]
    fill = INSTRUMENT_COLORS["surface_raised"]
    if state in {"selected", "active"}:
        border, foreground, fill = INSTRUMENT_COLORS["cyan"], INSTRUMENT_COLORS["cyan_bright"], "#183135"
    elif state == "warning":
        border, foreground, fill = INSTRUMENT_COLORS["amber"], "#f3d6a0", "#352719"
    elif state == "failed":
        border, foreground, fill = INSTRUMENT_COLORS["red"], "#ffd2cb", "#351b1a"
    elif state == "disabled":
        border, foreground, fill = "#443d30", "#77736c", "#101617"
    elif state in {"hover", "focused"}:
        border = INSTRUMENT_COLORS["cyan"] if state == "focused" else INSTRUMENT_COLORS["brass_bright"]
        foreground, fill = INSTRUMENT_COLORS["cyan_bright"], "#263033"
    elif state == "pressed":
        fill = INSTRUMENT_COLORS["surface_deep"]
    if role == "guarded" and state not in {"disabled", "failed"}:
        border = INSTRUMENT_COLORS["amber"]
    if role == "actuator" and state == "normal":
        border, foreground, fill = INSTRUMENT_COLORS["brass_bright"], "#111318", "#9c8350"
    return {"border": border, "foreground": foreground, "fill": fill}


class MachineKey(tk.Canvas):
    """Canvas-drawn engraved key with consistent mouse, keyboard, and state behavior."""

    role = "key"

    def __init__(self, parent, *, text: str, command=None, state: str = "normal", **kwargs) -> None:
        height = kwargs.pop("height", INSTRUMENT_DIMENSIONS["standard_height"])
        super().__init__(parent, height=height, background=INSTRUMENT_COLORS["surface"], highlightthickness=0, borderwidth=0, takefocus=True, cursor="hand2", **kwargs)
        self.text = text
        self.command = command
        self.control_state = "disabled" if state == "disabled" else "normal"
        self._hovered = False
        self._focused = False
        self._pressed = False
        self._selected = False
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", lambda _e: self._set_flag("_hovered", True))
        self.bind("<Leave>", self._leave)
        self.bind("<FocusIn>", lambda _e: self._set_flag("_focused", True))
        self.bind("<FocusOut>", lambda _e: self._set_flag("_focused", False))
        self.bind("<ButtonPress-1>", lambda _e: self._set_flag("_pressed", True))
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", lambda _e: self.invoke())
        self.bind("<space>", lambda _e: self.invoke())

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "text" in kwargs:
            self.text = str(kwargs.pop("text"))
        if "state" in kwargs:
            self.control_state = "disabled" if str(kwargs.pop("state")) == "disabled" else "normal"
        kwargs.pop("style", None)
        result = super().configure(**kwargs) if kwargs else None
        self._redraw()
        return result

    config = configure

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self._redraw()

    def set_visual_state(self, state: str) -> None:
        if state not in CONTROL_STATES:
            raise ValueError(f"Unknown machine control state: {state}")
        self.control_state = state
        self._redraw()

    def invoke(self) -> None:
        if self.control_state not in {"disabled", "active"} and self.command is not None:
            self.command()

    def _set_flag(self, name: str, value: bool) -> None:
        setattr(self, name, value)
        self._redraw()

    def _leave(self, _event=None) -> None:
        self._hovered = self._pressed = False
        self._redraw()

    def _release(self, _event=None) -> None:
        was_pressed = self._pressed
        self._pressed = False
        self._redraw()
        if was_pressed:
            self.invoke()

    def _effective_state(self) -> str:
        if self.control_state != "normal":
            return self.control_state
        if self._pressed:
            return "pressed"
        if self._selected:
            return "selected"
        if self._focused:
            return "focused"
        if self._hovered:
            return "hover"
        return "normal"

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width, height = max(8, self.winfo_width()), max(8, self.winfo_height())
        state = self._effective_state()
        palette = control_palette(state, self.role)
        offset = 2 if state == "pressed" else 0
        self.create_rectangle(1, 1, width - 2, height - 2, fill="#070a0b", outline=INSTRUMENT_COLORS["brass_dim"], width=1)
        self.create_rectangle(3 + offset, 3 + offset, width - 4 + offset, height - 4 + offset, fill=palette["fill"], outline=palette["border"], width=2)
        self.create_line(5 + offset, 5 + offset, width - 6 + offset, 5 + offset, fill="#766642", width=1)
        marker = "◆ " if state in {"selected", "active"} else "! " if state in {"warning", "failed"} else ""
        self.create_text(width / 2 + offset, height / 2 + offset, text=marker + self.text, fill=palette["foreground"], font=INSTRUMENT_FONTS["display"], width=max(20, width - 12))


class MachineActuator(MachineKey):
    role = "actuator"


class MachineGuardedControl(MachineKey):
    role = "guarded"


class MachineServiceControl(MachineKey):
    role = "key"


class MachineVerdictTag(MachineKey):
    def __init__(self, parent, *, text: str, command=None, state: str = "normal", **kwargs) -> None:
        super().__init__(parent, text=text, command=command, state=state, height=kwargs.pop("height", 28), **kwargs)


class MachineSelectorRail(tk.Canvas):
    """Discrete selector rendered as a brass rail rather than a native dropdown."""

    def __init__(self, parent, *, variable: tk.StringVar, values=(), command=None, display_values=None, **kwargs) -> None:
        super().__init__(parent, height=kwargs.pop("height", 30), background=INSTRUMENT_COLORS["surface"], highlightthickness=1, highlightbackground=INSTRUMENT_COLORS["brass_dim"], highlightcolor=INSTRUMENT_COLORS["cyan"], takefocus=True, cursor="hand2", **kwargs)
        self.variable, self.values, self.command = variable, list(values), command
        self.display_values = dict(display_values or {})
        self.control_state = "normal"
        self.variable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", self._redraw)
        self.bind("<Button-1>", self._click)
        self.bind("<Left>", lambda _e: self.step(-1))
        self.bind("<Right>", lambda _e: self.step(1))
        self.bind("<Up>", lambda _e: self.step(-1))
        self.bind("<Down>", lambda _e: self.step(1))
        self.bind("<Return>", lambda _e: self.step(1))
        self.bind("<space>", lambda _e: self.step(1))

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "values" in kwargs:
            self.values = list(kwargs.pop("values") or [])
        if "state" in kwargs:
            self.control_state = str(kwargs.pop("state"))
        result = super().configure(**kwargs) if kwargs else None
        self._redraw()
        return result

    config = configure

    def step(self, delta: int) -> None:
        if self.control_state == "disabled" or not self.values:
            return
        try:
            index = self.values.index(self.variable.get())
        except ValueError:
            index = 0
        self.variable.set(self.values[(index + delta) % len(self.values)])
        if self.command:
            self.command()

    def _click(self, event) -> None:
        self.step(-1 if event.x < self.winfo_width() * 0.25 else 1)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width, height = max(8, self.winfo_width()), max(8, self.winfo_height())
        disabled = self.control_state == "disabled"
        value = self.variable.get()
        display = self.display_values.get(value, value).upper()
        self.create_rectangle(1, 1, width - 2, height - 2, fill=INSTRUMENT_COLORS["surface_deep"], outline=INSTRUMENT_COLORS["brass_dim"], width=2)
        self.create_line(11, height / 2, width - 11, height / 2, fill="#4b422f", width=2)
        self.create_text(12, height / 2, text="‹", fill=INSTRUMENT_COLORS["brass"], font=INSTRUMENT_FONTS["display"], anchor="w")
        self.create_text(width - 12, height / 2, text="›", fill=INSTRUMENT_COLORS["brass"], font=INSTRUMENT_FONTS["display"], anchor="e")
        self.create_rectangle(25, 4, width - 25, height - 5, fill="#11191b", outline=INSTRUMENT_COLORS["dormant"] if disabled else INSTRUMENT_COLORS["brass"], width=1)
        self.create_text(width / 2, height / 2, text=display, fill="#77736c" if disabled else INSTRUMENT_COLORS["text"], font=INSTRUMENT_FONTS["caps"], width=max(20, width - 58))


class MachineInsetTrough(tk.Canvas):
    def __init__(self, parent, *, variable: tk.StringVar, formatter=concise_material_name, **kwargs) -> None:
        super().__init__(parent, height=kwargs.pop("height", INSTRUMENT_DIMENSIONS["control_height"]), background=INSTRUMENT_COLORS["surface"], highlightthickness=0, borderwidth=0, takefocus=True, **kwargs)
        self.variable, self.formatter = variable, formatter
        self.variable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", self._redraw)
        self.bind("<FocusIn>", self._redraw)
        self.bind("<FocusOut>", self._redraw)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width, height = max(8, self.winfo_width()), max(8, self.winfo_height())
        focused = self.focus_get() is self
        self.create_rectangle(1, 1, width - 2, height - 2, fill="#05090a", outline=INSTRUMENT_COLORS["cyan"] if focused else INSTRUMENT_COLORS["brass_dim"], width=2)
        self.create_line(4, height - 4, width - 4, height - 4, fill="#343e3e", width=1)
        self.create_text(9, height / 2, text=self.formatter(self.variable.get()), anchor="w", fill=INSTRUMENT_COLORS["text"] if self.variable.get().strip() else INSTRUMENT_COLORS["muted"], font=INSTRUMENT_FONTS["technical"], width=max(20, width - 16))


class MachinePlaque(tk.Canvas):
    def __init__(self, parent, *, variable: tk.StringVar, display_values=None, command=None, **kwargs) -> None:
        super().__init__(parent, height=kwargs.pop("height", 28), background=INSTRUMENT_COLORS["surface"], highlightthickness=0, borderwidth=0, takefocus=bool(command), cursor="hand2" if command else "", **kwargs)
        self.variable, self.display_values, self.command = variable, dict(display_values or {}), command
        self.variable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", self._redraw)
        if command:
            self.bind("<Button-1>", lambda _e: command())
            self.bind("<Return>", lambda _e: command())
            self.bind("<space>", lambda _e: command())

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width, height = max(8, self.winfo_width()), max(8, self.winfo_height())
        value = self.variable.get()
        display = self.display_values.get(value, value).upper()
        self.create_rectangle(1, 1, width - 2, height - 2, fill="#111719", outline=INSTRUMENT_COLORS["brass"], width=1)
        self.create_line(5, 4, width - 5, 4, fill=INSTRUMENT_COLORS["brass_bright"], width=1)
        self.create_text(width / 2, height / 2 + 1, text=display, fill=INSTRUMENT_COLORS["text"], font=INSTRUMENT_FONTS["display"], width=max(20, width - 12))


def selector_angle(index: int, count: int) -> float:
    """Return the selector's calibrated angle in degrees."""
    bounded_count = max(2, int(count))
    bounded_index = min(max(0, int(index)), bounded_count - 1)
    return 225.0 - (270.0 * bounded_index / (bounded_count - 1))


def meter_fraction(value: float, maximum: float = 100.0) -> float:
    if maximum <= 0:
        return 0.0
    return min(1.0, max(0.0, float(value) / float(maximum)))


@dataclass(frozen=True)
class OverlayBox:
    x: float
    y: float
    width: float
    height: float

    def pixels(self, bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, top, width, height = bounds
        return (
            round(left + self.x * width),
            round(top + self.y * height),
            max(1, round(self.width * width)),
            max(1, round(self.height * height)),
        )

    def inset(self, horizontal: float, vertical: float) -> "OverlayBox":
        """Return a content box inset from a decorative recess footprint."""
        if horizontal < 0 or vertical < 0:
            raise ValueError("Overlay insets must be non-negative")
        if horizontal * 2 >= self.width or vertical * 2 >= self.height:
            raise ValueError("Overlay insets must leave a positive content area")
        return OverlayBox(
            self.x + horizontal,
            self.y + vertical,
            self.width - horizontal * 2,
            self.height - vertical * 2,
        )


# These boxes describe the full engraved recesses in the plate artwork. Native
# widgets must not occupy their perimeter: that is where the plate's bevels,
# corner ornaments, and separator rules live.
INSTRUMENT_RECESS_BOXES = {
    "transformation": OverlayBox(0.055, 0.075, 0.215, 0.190),
    "material": OverlayBox(0.290, 0.075, 0.410, 0.190),
    "quality": OverlayBox(0.735, 0.075, 0.210, 0.190),
    "filter": OverlayBox(0.055, 0.300, 0.215, 0.235),
    "status": OverlayBox(0.290, 0.285, 0.410, 0.280),
    "activate": OverlayBox(0.735, 0.300, 0.210, 0.235),
    "progress": OverlayBox(0.245, 0.585, 0.510, 0.130),
    "stages": OverlayBox(0.220, 0.725, 0.560, 0.080),
    "curator": OverlayBox(0.065, 0.820, 0.405, 0.125),
    "notes": OverlayBox(0.525, 0.820, 0.410, 0.125),
}


# Insets are normalized to the plate rather than to the window, so the exposed
# bezel scales with the artwork. The shallow stage and footer recesses need a
# smaller vertical allowance to retain their useful content height.
_STANDARD_RECESS_INSET = (12 / PLATE_WIDTH, 10 / PLATE_HEIGHT)
_SHALLOW_RECESS_INSETS = {
    "progress": (12 / PLATE_WIDTH, 7 / PLATE_HEIGHT),
    "stages": (10 / PLATE_WIDTH, 4 / PLATE_HEIGHT),
    "curator": (12 / PLATE_WIDTH, 7 / PLATE_HEIGHT),
    "notes": (12 / PLATE_WIDTH, 7 / PLATE_HEIGHT),
}

INSTRUMENT_OVERLAY_BOXES = {
    name: recess.inset(*_SHALLOW_RECESS_INSETS.get(name, _STANDARD_RECESS_INSET))
    for name, recess in INSTRUMENT_RECESS_BOXES.items()
}


def instrument_plate_path(root: Path) -> Path:
    return Path(root) / INSTRUMENT_PLATE_RELATIVE_PATH


def fit_plate_bounds(width: int, height: int, *, aspect: float = PLATE_WIDTH / PLATE_HEIGHT) -> tuple[int, int, int, int]:
    available_width = max(1, int(width))
    available_height = max(1, int(height))
    if available_width / available_height > aspect:
        plate_height = available_height
        plate_width = round(plate_height * aspect)
    else:
        plate_width = available_width
        plate_height = round(plate_width / aspect)
    return (
        (available_width - plate_width) // 2,
        (available_height - plate_height) // 2,
        plate_width,
        plate_height,
    )


class InstrumentPlateCanvas(tk.Canvas):
    """Scales one static plate and keeps native widgets aligned over its recesses."""

    def __init__(self, parent, *, asset_path: Path, **kwargs) -> None:
        super().__init__(parent, background="#080a0b", highlightthickness=0, borderwidth=0, **kwargs)
        self.asset_path = Path(asset_path)
        self._source_image = None
        self._tk_image = None
        self._image_item = self.create_image(0, 0, anchor="nw")
        self._overlays: dict[str, tuple[int, OverlayBox]] = {}
        self._resize_job: str | None = None
        self._last_render_size: tuple[int, int] | None = None
        self._load_plate()
        self.bind("<Configure>", self._schedule_layout)

    @property
    def plate_available(self) -> bool:
        return self._source_image is not None or self._tk_image is not None

    def _load_plate(self) -> None:
        if not self.asset_path.exists():
            return
        if Image is not None:
            self._source_image = Image.open(self.asset_path).convert("RGB")
            return
        try:
            self._tk_image = tk.PhotoImage(file=str(self.asset_path))
        except tk.TclError:
            self._tk_image = None

    def register_overlay(self, name: str, widget: tk.Widget, box: OverlayBox | None = None) -> None:
        selected_box = box or INSTRUMENT_OVERLAY_BOXES[name]
        item = self.create_window(0, 0, anchor="nw", window=widget)
        self._overlays[name] = (item, selected_box)
        self._layout()

    def set_overlay_box(self, name: str, box: OverlayBox) -> None:
        item, _old_box = self._overlays[name]
        self._overlays[name] = (item, box)
        self._layout()

    def show_overlay(self, name: str, visible: bool) -> None:
        item, _box = self._overlays[name]
        self.itemconfigure(item, state="normal" if visible else "hidden")

    def _schedule_layout(self, _event=None) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._layout)

    def _layout(self) -> None:
        self._resize_job = None
        canvas_width = max(1, self.winfo_width())
        canvas_height = max(1, self.winfo_height())
        bounds = fit_plate_bounds(canvas_width, canvas_height)
        left, top, plate_width, plate_height = bounds
        if self._source_image is not None and ImageTk is not None:
            render_size = (plate_width, plate_height)
            if render_size != self._last_render_size:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                resized = self._source_image.resize(render_size, resampling)
                self._tk_image = ImageTk.PhotoImage(resized)
                self._last_render_size = render_size
        if self._tk_image is not None:
            self.itemconfigure(self._image_item, image=self._tk_image)
            self.coords(self._image_item, left, top)
        else:
            self.itemconfigure(self._image_item, image="")
            self.configure(background="#111315")
        for item, box in self._overlays.values():
            x, y, width, height = box.pixels(bounds)
            self.coords(item, x, y)
            self.itemconfigure(item, width=width, height=height)


class RotarySelector(tk.Canvas):
    """Keyboard-accessible rotary selector backed by a normal Tk variable."""

    def __init__(self, parent, *, variable: tk.StringVar, values=(), command=None, display_values=None, **kwargs) -> None:
        background = kwargs.pop("background", INSTRUMENT_COLORS["surface"])
        super().__init__(
            parent,
            background=background,
            highlightthickness=1,
            highlightbackground=INSTRUMENT_COLORS["brass_dim"],
            highlightcolor=INSTRUMENT_COLORS["cyan"],
            takefocus=True,
            cursor="hand2",
            **kwargs,
        )
        self.variable = variable
        self.values = list(values)
        self.command = command
        self.display_values = dict(display_values or {})
        self.control_state = "normal"
        self.variable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", self._redraw)
        self.bind("<Button-1>", lambda _event: self.step(1))
        self.bind("<Button-3>", lambda _event: self.step(-1))
        self.bind("<MouseWheel>", lambda event: self.step(1 if event.delta > 0 else -1))
        self.bind("<Left>", lambda _event: self.step(-1))
        self.bind("<Right>", lambda _event: self.step(1))
        self.bind("<Up>", lambda _event: self.step(1))
        self.bind("<Down>", lambda _event: self.step(-1))
        self.bind("<Return>", lambda _event: self.step(1))
        self.bind("<space>", lambda _event: self.step(1))

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "values" in kwargs:
            self.values = list(kwargs.pop("values") or [])
        if "state" in kwargs:
            self.control_state = str(kwargs.pop("state"))
        result = super().configure(**kwargs) if kwargs else None
        self._redraw()
        return result

    config = configure

    def step(self, delta: int) -> None:
        if self.control_state == "disabled" or not self.values:
            return
        current = self.variable.get()
        try:
            index = self.values.index(current)
        except ValueError:
            index = -1 if delta > 0 else 0
        self.variable.set(self.values[(index + delta) % len(self.values)])
        if self.command is not None:
            self.command()

    def _redraw(self, _event=None) -> None:
        self.delete("dial")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        radius = max(18, min(width * 0.22, height * 0.29))
        cx, cy = width / 2, min(height * 0.41, radius + 13)
        muted = self.control_state == "disabled"
        outline = "#5c574d" if muted else INSTRUMENT_COLORS["brass"]
        fill = "#17191a" if muted else INSTRUMENT_COLORS["surface_raised"]
        count = max(2, len(self.values))
        current = self.variable.get()
        index = self.values.index(current) if current in self.values else 0

        for tick_index in range(min(count, 11)):
            tick_angle = math.radians(selector_angle(tick_index, min(count, 11)))
            inner = radius * 1.12
            outer = radius * 1.27
            self.create_line(
                cx + math.cos(tick_angle) * inner,
                cy - math.sin(tick_angle) * inner,
                cx + math.cos(tick_angle) * outer,
                cy - math.sin(tick_angle) * outer,
                fill=INSTRUMENT_COLORS["brass_dim"] if muted else INSTRUMENT_COLORS["brass"],
                width=2,
                tags="dial",
            )
        self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#090d0e", outline=outline, width=3, tags="dial")
        self.create_oval(cx - radius * .83, cy - radius * .83, cx + radius * .83, cy + radius * .83, fill="#111617", outline="#4a4030", width=2, tags="dial")
        self.create_oval(cx - radius * .72, cy - radius * .72, cx + radius * .72, cy + radius * .72, fill=fill, outline="#292f30", width=2, tags="dial")
        angle = math.radians(selector_angle(index, count))
        px = cx + math.cos(angle) * radius * 0.66
        py = cy - math.sin(angle) * radius * 0.66
        self.create_line(cx + 2, cy + 2, px + 2, py + 2, fill="#050708", width=5, capstyle="round", tags="dial")
        self.create_line(cx, cy, px, py, fill=INSTRUMENT_COLORS["cyan"] if not muted else "#59666a", width=4, capstyle="round", tags="dial")
        self.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#292e2e", outline=outline, width=1, tags="dial")
        readout_y = min(height - 13, cy + radius + 18)
        self.create_rectangle(8, readout_y - 11, width - 8, readout_y + 11, fill=INSTRUMENT_COLORS["surface_deep"], outline=INSTRUMENT_COLORS["brass_dim"], width=1, tags="dial")
        self.create_text(width / 2, readout_y, text=self.display_values.get(current, current), fill=INSTRUMENT_COLORS["text"] if not muted else "#77736c", font=("Georgia", max(8, int(height * 0.078)), "bold"), width=max(60, width - 20), tags="dial")


class InstrumentMeter(tk.Canvas):
    """A recessed, continuously updating instrument meter."""

    def __init__(self, parent, *, variable: tk.Variable, maximum: float = 100.0, **kwargs) -> None:
        background = kwargs.pop("background", INSTRUMENT_COLORS["surface"])
        super().__init__(parent, height=18, background=background, highlightthickness=0, borderwidth=0, takefocus=False, **kwargs)
        self.variable = variable
        self.maximum = maximum
        self.variable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(8, self.winfo_width())
        height = max(8, self.winfo_height())
        inset = 2
        try:
            fraction = meter_fraction(float(self.variable.get()), self.maximum)
        except (TypeError, ValueError, tk.TclError):
            fraction = 0.0
        self.create_rectangle(inset, inset, width - inset, height - inset, fill=INSTRUMENT_COLORS["surface_deep"], outline="#332d23", width=2)
        inner_left, inner_top = inset + 3, inset + 3
        inner_right, inner_bottom = width - inset - 3, height - inset - 3
        fill_right = inner_left + max(0, inner_right - inner_left) * fraction
        if fill_right > inner_left:
            self.create_rectangle(inner_left, inner_top, fill_right, inner_bottom, fill="#4aaab4", outline="")
            self.create_line(inner_left, inner_top, fill_right, inner_top, fill=INSTRUMENT_COLORS["cyan_bright"], width=1)
        for division in range(1, 10):
            x = inner_left + (inner_right - inner_left) * division / 10
            self.create_line(x, inner_top, x, inner_bottom, fill="#263437", width=1)


class ActivityLamp(tk.Canvas):
    def __init__(self, parent, *, diameter: int = 18, **kwargs) -> None:
        background = kwargs.pop("background", INSTRUMENT_COLORS["surface"])
        super().__init__(parent, width=diameter, height=diameter, background=background, highlightthickness=0, takefocus=False, **kwargs)
        self.state = "off"
        self.phase = False
        self.bind("<Configure>", self._redraw)

    def set_active(self, active: bool) -> None:
        self.set_state("active" if active else "off")

    def set_state(self, state: str) -> None:
        if state not in {"off", "active", "complete", "skipped", "warning", "failed"}:
            raise ValueError(f"Unknown lamp state: {state}")
        self.state = state
        self.phase = False
        self._redraw()

    def pulse(self) -> None:
        if self.state == "active":
            self.phase = not self.phase
            self._redraw()

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(4, self.winfo_width())
        height = max(4, self.winfo_height())
        inset = 2
        if self.state == "active":
            fill = "#a9f3ef" if self.phase else "#4eb9bd"
            outline = "#d2ffff"
        elif self.state == "complete":
            fill = "#72d4d2"
            outline = "#c8ffff"
        elif self.state == "warning":
            fill, outline = INSTRUMENT_COLORS["amber"], "#f4d29a"
        elif self.state == "failed":
            fill, outline = INSTRUMENT_COLORS["red"], "#ffd0c8"
        elif self.state == "skipped":
            fill, outline = INSTRUMENT_COLORS["surface_deep"], INSTRUMENT_COLORS["muted"]
        else:
            fill = "#253033"
            outline = "#776748"
        self.create_oval(inset, inset, width - inset, height - inset, fill="#0a0d0e", outline=INSTRUMENT_COLORS["brass_dim"], width=2)
        self.create_oval(inset + 3, inset + 3, width - inset - 3, height - inset - 3, fill=fill, outline=outline, width=2)
        if self.state != "off":
            self.create_oval(inset + 6, inset + 5, width * .48, height * .42, fill=INSTRUMENT_COLORS["cyan_bright"], outline="")
        if self.state == "skipped":
            self.create_line(inset + 4, height - inset - 4, width - inset - 4, inset + 4, fill=INSTRUMENT_COLORS["muted"], width=2)


class MachineProcessStation(tk.Frame):
    def __init__(self, parent, *, text: str, **kwargs) -> None:
        super().__init__(parent, background=INSTRUMENT_COLORS["surface"], **kwargs)
        self.columnconfigure(0, weight=1)
        self.lamp = ActivityLamp(self, diameter=18)
        self.lamp.grid(row=0, column=0)
        self.label = tk.Label(self, text=text, background=INSTRUMENT_COLORS["surface"], foreground=INSTRUMENT_COLORS["text"], font=INSTRUMENT_FONTS["caps"], anchor="center")
        self.label.grid(row=1, column=0, sticky="ew")

    def set_state(self, state: str) -> None:
        self.lamp.set_state(state)
        colors = {
            "off": INSTRUMENT_COLORS["dormant"], "active": INSTRUMENT_COLORS["cyan_bright"],
            "complete": INSTRUMENT_COLORS["cyan"], "skipped": INSTRUMENT_COLORS["muted"],
            "warning": INSTRUMENT_COLORS["amber"], "failed": INSTRUMENT_COLORS["red"],
        }
        self.label.configure(foreground=colors[state])


class ObservationTrace(tk.Canvas):
    """Low-cost abstract process-activity trace driven only by reported progress state."""

    def __init__(self, parent, *, progress_variable: tk.Variable, active_getter=None, reduced_motion: bool = False, **kwargs) -> None:
        super().__init__(parent, height=kwargs.pop("height", 74), background=INSTRUMENT_COLORS["surface_deep"], highlightthickness=1, highlightbackground=INSTRUMENT_COLORS["brass_dim"], borderwidth=0, **kwargs)
        self.progress_variable = progress_variable
        self.active_getter = active_getter or (lambda: False)
        self.reduced_motion = bool(reduced_motion)
        self.phase = 0
        self.bind("<Configure>", self._redraw)
        self.after(250, self._tick)

    def _tick(self) -> None:
        visible = self.winfo_toplevel().state() != "iconic"
        if visible and self.active_getter() and not self.reduced_motion:
            self.phase = (self.phase + 1) % 120
        if visible:
            self._redraw()
        self.after(500 if self.reduced_motion else 250, self._tick)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width, height = max(20, self.winfo_width()), max(20, self.winfo_height())
        for division in range(1, 6):
            x = width * division / 6
            self.create_line(x, 8, x, height - 8, fill="#222a27", width=1)
        for division in range(1, 3):
            y = 8 + (height - 16) * division / 3
            self.create_line(6, y, width - 6, y, fill="#222a27", width=1)
        active = bool(self.active_getter())
        try:
            progress = meter_fraction(float(self.progress_variable.get()))
        except (TypeError, ValueError, tk.TclError):
            progress = 0.0
        center = height * 0.56
        points = []
        for x in range(6, width - 5, 4):
            if active:
                amplitude = 5 + progress * 12
                y = center + math.sin((x + self.phase * 5) * 0.075) * amplitude * (0.45 + 0.55 * math.sin(x * 0.021) ** 2)
            else:
                y = center
            points.extend((x, y))
        if len(points) >= 4:
            self.create_line(*points, fill=INSTRUMENT_COLORS["cyan"] if active else INSTRUMENT_COLORS["dormant"], width=2, smooth=True)
        self.create_text(8, 8, text="PROCESS ACTIVITY", anchor="nw", fill=INSTRUMENT_COLORS["brass_dim"], font=INSTRUMENT_FONTS["caps"])


class MachineComponentGallery(tk.Toplevel):
    """Isolated review sheet for the machine control vocabulary and states."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, background=INSTRUMENT_COLORS["surface"])
        self.title("Cinelingus Machine Component Gallery")
        self.geometry("1180x720")
        self.minsize(960, 620)
        self._variables: list[tk.Variable] = []
        self._build()

    def _label(self, parent, text: str, *, large: bool = False) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            background=INSTRUMENT_COLORS["surface"],
            foreground=INSTRUMENT_COLORS["brass_bright"],
            font=INSTRUMENT_FONTS["display_large" if large else "caps"],
            anchor="w",
        )

    def _remember(self, value: tk.Variable) -> tk.Variable:
        self._variables.append(value)
        return value

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self._label(self, "MACHINE CONTROL LIBRARY", large=True).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 4))
        self._label(self, "Canonical controls and semantic states — review at 100%, 125%, and 150% scale").grid(row=1, column=0, sticky="ew", padx=16)

        state_frame = tk.Frame(self, background=INSTRUMENT_COLORS["surface"])
        state_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(18, 12))
        for column, state in enumerate(CONTROL_STATES):
            state_frame.columnconfigure(column, weight=1, uniform="states")
            self._label(state_frame, state.upper()).grid(row=0, column=column, sticky="ew", padx=3, pady=(0, 4))
            key = MachineKey(state_frame, text="KEY", width=112)
            key.grid(row=1, column=column, sticky="ew", padx=3)
            key.set_visual_state(state)

        components = tk.Frame(self, background=INSTRUMENT_COLORS["surface"])
        components.grid(row=3, column=0, sticky="nsew", padx=16, pady=8)
        self.rowconfigure(3, weight=1)
        for column in range(4):
            components.columnconfigure(column, weight=1, uniform="components")

        self._label(components, "READOUTS").grid(row=0, column=0, sticky="ew", padx=4)
        material = self._remember(tk.StringVar(value=r"C:\\archive\\specimens\\a_very_long_cinematic_material_name.mov"))
        MachineInsetTrough(components, variable=material).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        plaque_value = self._remember(tk.StringVar(value="TRIANGLE"))
        MachinePlaque(components, variable=plaque_value).grid(row=2, column=0, sticky="ew", padx=4, pady=4)

        self._label(components, "SELECTORS").grid(row=0, column=1, sticky="ew", padx=4)
        rail_value = self._remember(tk.StringVar(value="Several Films"))
        MachineSelectorRail(components, variable=rail_value, values=("One Film", "Several Films")).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        dial_value = self._remember(tk.StringVar(value="Study"))
        RotarySelector(components, variable=dial_value, values=("Glimpse", "Study", "Divination"), height=150).grid(row=2, column=1, rowspan=4, sticky="nsew", padx=4, pady=4)

        self._label(components, "ACTIONS").grid(row=0, column=2, sticky="ew", padx=4)
        MachineActuator(components, text="INVOKE", height=60).grid(row=1, column=2, sticky="ew", padx=4, pady=4)
        MachineGuardedControl(components, text="SAFE INTERRUPT").grid(row=2, column=2, sticky="ew", padx=4, pady=4)
        MachineServiceControl(components, text="SERVICE").grid(row=3, column=2, sticky="ew", padx=4, pady=4)
        verdict = MachineVerdictTag(components, text="RARE ALIGNMENT")
        verdict.grid(row=4, column=2, sticky="ew", padx=4, pady=4)
        verdict.set_selected(True)

        self._label(components, "INSTRUMENTS").grid(row=0, column=3, sticky="ew", padx=4)
        lamp_row = tk.Frame(components, background=INSTRUMENT_COLORS["surface"])
        lamp_row.grid(row=1, column=3, sticky="ew", padx=4, pady=4)
        for state in ("off", "active", "complete", "skipped", "warning", "failed"):
            lamp = ActivityLamp(lamp_row, diameter=24)
            lamp.pack(side="left", padx=3)
            lamp.set_state(state)
        progress = self._remember(tk.DoubleVar(value=62))
        InstrumentMeter(components, variable=progress).grid(row=2, column=3, sticky="ew", padx=4, pady=8)
        station = MachineProcessStation(components, text="RENDER")
        station.grid(row=3, column=3, sticky="ew", padx=4, pady=4)
        station.set_state("active")
        ObservationTrace(components, progress_variable=progress, active_getter=lambda: True, reduced_motion=True).grid(row=4, column=3, rowspan=2, sticky="nsew", padx=4, pady=4)


def show_component_gallery() -> None:
    """Launch the isolated machine component review sheet."""
    root = tk.Tk()
    root.withdraw()
    gallery = MachineComponentGallery(root)
    gallery.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


class ToolTip:
    def __init__(self, widget: tk.Widget, text) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<FocusIn>", self._show, add="+")
        widget.bind("<FocusOut>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        text = self.text() if callable(self.text) else self.text
        if self.window is not None or not text:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{self.widget.winfo_rootx() + 16}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 8}")
        tk.Label(self.window, text=text, background="#f1e7cf", foreground="#17191a", relief="solid", borderwidth=1, padx=7, pady=4).pack()

    def _hide(self, _event=None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


if __name__ == "__main__":  # pragma: no cover - visual review utility
    show_component_gallery()
