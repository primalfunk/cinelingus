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
}


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


INSTRUMENT_OVERLAY_BOXES = {
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

    def __init__(self, parent, *, variable: tk.StringVar, values=(), command=None, **kwargs) -> None:
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
        self.create_text(width / 2, readout_y, text=current, fill=INSTRUMENT_COLORS["text"] if not muted else "#77736c", font=("Georgia", max(8, int(height * 0.078)), "bold"), width=max(60, width - 20), tags="dial")


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
        if state not in {"off", "active", "complete"}:
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
        else:
            fill = "#253033"
            outline = "#776748"
        self.create_oval(inset, inset, width - inset, height - inset, fill="#0a0d0e", outline=INSTRUMENT_COLORS["brass_dim"], width=2)
        self.create_oval(inset + 3, inset + 3, width - inset - 3, height - inset - 3, fill=fill, outline=outline, width=2)
        if self.state != "off":
            self.create_oval(inset + 6, inset + 5, width * .48, height * .42, fill=INSTRUMENT_COLORS["cyan_bright"], outline="")


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
