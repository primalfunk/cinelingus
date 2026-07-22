from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget,
)

from .qt_controller import MATCHING_LABELS, QUALITY_LABELS, QtEngineController
from .cache import clear_pipeline_cache


DESIGN_WIDTH = 1536
DESIGN_HEIGHT = 1024
COLORS = {
    "glass": QColor("#071012"),
    "glass_raised": QColor("#0d191c"),
    "brass": QColor("#c4a66b"),
    "brass_dim": QColor("#67583d"),
    "ivory": QColor("#eee6d5"),
    "cyan": QColor("#83d8e8"),
    "cyan_bright": QColor("#c9ffff"),
    "dormant": QColor("#53605b"),
    "amber": QColor("#d09a48"),
    "red": QColor("#b9584f"),
}
FONT_FILES = {
    "machine": Path("assets/fonts/cinzel/Cinzel-Variable.ttf"),
    "panel": Path("assets/fonts/ibm-plex-sans-condensed/IBMPlexSansCondensed-Medium.ttf"),
    "instrument": Path("assets/fonts/share-tech-mono/ShareTechMono-Regular.ttf"),
}
FOCUS_ORDER = ("apparatus", "materials", "scrutiny", "calibration", "actuation", "curator", "ledger", "service")
STATIONS = ("CATALOG", "VOICE", "IDENTITY", "PERFORMANCE", "ASSEMBLY", "RENDER", "REVIEW", "ARCHIVE")


def marquee_offset(text_width: float, viewport_width: float, elapsed: float, *, speed: float = 22.0, pause: float = 1.5, gap: float = 44.0) -> float:
    """Return a leftward, hold-then-reset marquee offset for overflowing text."""
    if text_width <= viewport_width:
        return 0.0
    cycle = pause + (text_width + gap) / speed
    position = elapsed % cycle
    return 0.0 if position < pause else -speed * (position - pause)


def open_path_or_reveal(path: Path) -> str:
    try:
        os.startfile(path)
        return "opened"
    except OSError as open_error:
        command = ["explorer.exe", str(path)] if path.is_dir() else ["explorer.exe", "/select,", str(path)]
        try:
            subprocess.Popen(command)
        except OSError as reveal_error:
            raise OSError(f"Could not open {path}: {open_error}; Explorer fallback failed: {reveal_error}") from reveal_error
        return "revealed"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(root: Path | None = None) -> dict:
    base = root or repository_root()
    return json.loads((base / "assets" / "instrument_apertures.json").read_text(encoding="utf-8"))


def register_fonts(root: Path | None = None) -> dict[str, str]:
    base = root or repository_root()
    families: dict[str, str] = {}
    for role, relative in FONT_FILES.items():
        font_id = QFontDatabase.addApplicationFont(str(base / relative))
        available = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        if not available:
            raise RuntimeError(f"Unable to register required {role} font: {base / relative}")
        families[role] = available[0]
    return families


def apply_application_style(app: QApplication, root: Path | None = None) -> None:
    fonts = register_fonts(root)
    panel = fonts["panel"]
    instrument = fonts["instrument"]
    app.setStyleSheet(
        f"""
        QWidget {{
            background-color: #0a1012;
            color: #eee6d5;
            font-family: "{panel}";
            font-size: 14px;
        }}
        QDialog, QMainWindow {{ background-color: #080b0d; }}
        QTabWidget::pane, QListWidget, QPlainTextEdit, QLineEdit, QComboBox,
        QSpinBox, QDoubleSpinBox {{
            background-color: #071012;
            border: 1px solid #67583d;
            color: #eee6d5;
            selection-background-color: #29464b;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {{
            font-family: "{instrument}";
            padding: 5px;
        }}
        QPushButton {{
            background-color: #171b19;
            border: 1px solid #9d8555;
            color: #d8bd79;
            padding: 7px 14px;
        }}
        QPushButton:hover, QPushButton:focus {{
            border-color: #c9ffff;
            color: #c9ffff;
        }}
        QTabBar::tab {{
            background: #101719;
            border: 1px solid #67583d;
            padding: 7px 16px;
        }}
        QTabBar::tab:selected {{ color: #c9ffff; border-bottom-color: #0a1012; }}
        """
    )


def aperture_path(aperture: dict) -> QPainterPath:
    path = QPainterPath()
    shape = aperture["shape"]
    if shape == "rounded_rect":
        x1, y1, x2, y2 = aperture["bounds"]
        radius = float(aperture["radius"])
        path.addRoundedRect(QRectF(x1, y1, x2 - x1, y2 - y1), radius, radius)
    elif shape == "ellipse":
        x1, y1, x2, y2 = aperture["bounds"]
        path.addEllipse(QRectF(x1, y1, x2 - x1, y2 - y1))
    elif shape == "polygon":
        points = [QPointF(*point) for point in aperture["points"]]
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
    elif shape == "circles":
        radius = float(aperture["radius"])
        for x, y in aperture["centers"]:
            path.addEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))
    else:
        raise ValueError(f"Unknown aperture shape: {shape}")
    return path


def content_rect(aperture: dict) -> QRectF:
    x1, y1, x2, y2 = aperture["content"]
    return QRectF(x1, y1, x2 - x1, y2 - y1)


class FaceplateWidget(QWidget):
    invoke_requested = Signal()
    service_requested = Signal()
    configuration_requested = Signal()
    archive_requested = Signal()
    curator_requested = Signal()

    def __init__(self, root: Path | None = None, controller: QtEngineController | None = None) -> None:
        super().__init__()
        self.root = root or repository_root()
        self.manifest = load_manifest(self.root)
        self.apertures = {row["id"]: row for row in self.manifest["apertures"]}
        self.fonts = register_fonts(self.root)
        self.overlay = QPixmap(str(self.root / "assets" / self.manifest["plate_overlay"]))
        if self.overlay.isNull():
            raise RuntimeError("The layered faceplate asset is unavailable. Run tools/build_faceplate_overlay.py.")
        self.materials = ["EMPTY CHAMBER", "EMPTY CHAMBER", "EMPTY CHAMBER"]
        self.quality_index = 1
        self.calibration_index = 0
        self.running = False
        self.completed = False
        self.progress = 0.0
        self.phase = 0.0
        self.marquee_time = 0.0
        self._marquee_materials: tuple[str, ...] = ()
        self.focus_index = 0
        self.keyboard_focus_visible = False
        self.setMinimumSize(960, 640)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Cinelingus Engine machine interface")
        self.setAccessibleDescription("Layered cinematic instrument faceplate with keyboard-focusable apparatus controls")
        self.timer = QTimer(self)
        self.timer.setInterval(125)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.controller = controller
        if self.controller is not None:
            self.controller.changed.connect(self.update)

    @property
    def focused_control(self) -> str:
        return FOCUS_ORDER[self.focus_index]

    def set_demo_state(self, state: str) -> None:
        self.running = state == "running"
        self.completed = state == "complete"
        self.progress = 0.46 if self.running else 1.0 if self.completed else 0.0
        if state in {"ready", "running", "complete"}:
            self.materials = ["AMERICAN SAINTS.IA.MP4", "WKYK — S1E6.MP4", "ROBOCOP CARTOON — EP 02.MP4"]
        self.update()

    def _tick(self) -> None:
        self.phase = (self.phase + 0.15) % (math.pi * 2)
        self.marquee_time += self.timer.interval() / 1000
        if self.running:
            self.progress = min(0.98, self.progress + 0.0008)
        self.update()

    def _transform(self) -> tuple[float, float, float]:
        scale = min(self.width() / DESIGN_WIDTH, self.height() / DESIGN_HEIGHT)
        return scale, (self.width() - DESIGN_WIDTH * scale) / 2, (self.height() - DESIGN_HEIGHT * scale) / 2

    def _design_point(self, position: QPointF) -> QPointF:
        scale, left, top = self._transform()
        return QPointF((position.x() - left) / scale, (position.y() - top) / scale)

    def _font(self, role: str, pixels: int, *, bold: bool = False) -> QFont:
        font = QFont(self.fonts[role])
        font.setPixelSize(pixels)
        font.setWeight(QFont.Weight.Bold if bold else QFont.Weight.Medium)
        return font

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#050708"))
        scale, left, top = self._transform()
        painter.translate(left, top)
        painter.scale(scale, scale)
        for aperture in self.apertures.values():
            painter.fillPath(aperture_path(aperture), COLORS["glass"])
        self._paint_apparatus(painter)
        self._paint_materials(painter)
        self._paint_scrutiny(painter)
        self._paint_calibration(painter)
        self._paint_observation(painter)
        self._paint_actuation(painter)
        self._paint_progress(painter)
        self._paint_stations(painter)
        self._paint_curator(painter)
        self._paint_ledger(painter)
        painter.drawPixmap(0, 0, self.overlay)
        self._paint_focus(painter)

    def _text(self, painter: QPainter, text: str, rect: QRectF, size: int = 18, color: QColor | None = None, alignment=Qt.AlignmentFlag.AlignCenter, role: str = "instrument") -> None:
        painter.setFont(self._font(role, size))
        painter.setPen(color or COLORS["ivory"])
        painter.drawText(rect, alignment | Qt.TextFlag.TextWordWrap, text)

    def _line(self, painter: QPainter, text: str, rect: QRectF, y: float, size: int = 17, color: QColor | None = None, role: str = "instrument") -> None:
        self._text(painter, text, QRectF(rect.left(), y, rect.width(), size + 8), size, color, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, role)

    def _marquee_line(self, painter: QPainter, text: str, rect: QRectF, y: float, size: int = 13) -> None:
        viewport = QRectF(rect.left(), y, rect.width(), size + 8)
        font = self._font("instrument", size)
        metrics = QFontMetricsF(font)
        text_width = metrics.horizontalAdvance(text)
        if text_width <= viewport.width():
            self._text(painter, text, viewport, size, COLORS["ivory"])
            return
        offset = marquee_offset(text_width, viewport.width(), self.marquee_time)
        baseline = viewport.center().y() + (metrics.ascent() - metrics.descent()) / 2
        painter.save()
        painter.setClipRect(viewport)
        painter.setFont(font)
        painter.setPen(COLORS["ivory"])
        painter.drawText(QPointF(viewport.left() + offset, baseline), text)
        painter.restore()

    def _paint_apparatus(self, painter: QPainter) -> None:
        rect = content_rect(self.apertures["apparatus"])
        values = QRectF(rect.left() + 66, rect.top(), rect.width() - 70, rect.height())
        reality = self.controller.state.reality if self.controller else "SEVERAL FILMS"
        discipline = self.controller.state.discipline if self.controller else "ALCHEMICAL ENGINE"
        apparatus = self.controller.state.apparatus if self.controller else "TRIANGLE"
        self._line(painter, reality.upper(), values, rect.top() + 4, 15, COLORS["ivory"])
        self._line(painter, discipline.upper(), values, rect.top() + 40, 15, COLORS["ivory"])
        self._line(painter, apparatus.upper(), values, rect.top() + 77, 22, COLORS["cyan"])

    def _paint_materials(self, painter: QPainter) -> None:
        first = content_rect(self.apertures["materials"])
        second = content_rect(self.apertures["materials_secondary"])
        first_values = QRectF(first.left() + 55, first.top(), first.width() - 60, first.height())
        second_values = QRectF(second.left() + 62, second.top(), second.width() - 67, second.height())
        materials = self.materials
        if self.controller is not None:
            materials = [path.name.upper() if path else "EMPTY CHAMBER" for path in self.controller.state.films]
            materials = (materials + ["EMPTY CHAMBER"] * 3)[:3]
        signature = tuple(materials)
        if signature != self._marquee_materials:
            self._marquee_materials = signature
            self.marquee_time = 0.0
        self._marquee_line(painter, materials[0], first_values, first.top() + 10, 13)
        self._marquee_line(painter, materials[1], first_values, first.top() + 64, 13)
        self._marquee_line(painter, materials[2], second_values, second.top() + 10, 13)
        self._line(painter, "ADMIT MATERIAL", second_values, second.top() + 65, 13, COLORS["brass"], "panel")

    def _paint_scrutiny(self, painter: QPainter) -> None:
        qualities = (("GLIMPSE", "FAST PREVIEW"), ("STUDY", "BALANCED"), ("DIVINATION", "HIGH ACCURACY"))
        quality_index = QUALITY_LABELS.index(self.controller.state.quality) if self.controller else self.quality_index
        name, practical = qualities[quality_index]
        rect = content_rect(self.apertures["scrutiny"])
        values = QRectF(rect.left() + 67, rect.top(), rect.width() - 72, rect.height())
        self._line(painter, name, values, rect.top() + 11, 23, COLORS["cyan"])
        self._line(painter, practical, values, rect.top() + 57, 14)
        model = self.controller.whisper_model.upper() if self.controller else "MEDIUM"
        self._line(painter, model, values, rect.top() + 88, 13, COLORS["ivory"])

    def _paint_calibration(self, painter: QPainter) -> None:
        values = ("BALANCED", "RHYTHM", "DENSE COMEDY", "DEADPAN", "CONTRAST", "MINIMAL REUSE", "CHAOS")
        rect = content_rect(self.apertures["calibration"])
        center = QPointF(rect.center().x(), rect.top() + 58)
        painter.setPen(QPen(COLORS["brass_dim"], 3))
        painter.setBrush(COLORS["glass_raised"])
        painter.drawEllipse(center, 42, 42)
        calibration_index = MATCHING_LABELS.index(self.controller.state.matching) if self.controller else self.calibration_index
        angle = math.radians(225 - calibration_index * (270 / max(1, len(MATCHING_LABELS) - 1)))
        tip = QPointF(center.x() + math.cos(angle) * 30, center.y() - math.sin(angle) * 30)
        painter.setPen(QPen(COLORS["cyan"], 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(center, tip)
        calibration = self.controller.state.matching.upper() if self.controller else values[self.calibration_index]
        self._line(painter, calibration, rect, rect.bottom() - 35, 15, COLORS["ivory"])

    def _paint_observation(self, painter: QPainter) -> None:
        rect = content_rect(self.apertures["observation"])
        running = self.controller.state.running if self.controller else self.running
        completed = self.controller.state.completed if self.controller else self.completed
        state = self.controller.state.machine_state if self.controller else "ACTIVE" if running else "COMPLETE" if completed else "DORMANT"
        operation = self.controller.state.operation if self.controller else "TRANSCRIBING SPOKEN PASSAGES" if running else "ARTIFACT READY FOR REVIEW" if completed else "AWAITING MATERIAL"
        active_color = COLORS["cyan"] if running or completed else COLORS["red"] if state == "FAULT" else COLORS["dormant"]
        self._text(painter, state, QRectF(rect.left() + 8, rect.top() + 20, 120, 25), 15, active_color, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._text(painter, operation, QRectF(rect.left() + 140, rect.top() + 20, rect.width() - 148, 25), 17, active_color)
        trace = QRectF(rect.left() + 12, rect.top() + 50, rect.width() - 24, 82)
        painter.setPen(QPen(QColor("#233033"), 1))
        for index in range(1, 8):
            x = trace.left() + trace.width() * index / 8
            painter.drawLine(QPointF(x, trace.top()), QPointF(x, trace.bottom()))
        path = QPainterPath(QPointF(trace.left(), trace.center().y()))
        for x in range(int(trace.left()), int(trace.right()) + 1, 4):
            amplitude = 18 if running else 0
            y = trace.center().y() + math.sin(x * 0.055 + self.phase * 4) * amplitude * (0.5 + 0.5 * math.sin(x * 0.013) ** 2)
            path.lineTo(x, y)
        painter.setPen(QPen(active_color, 2))
        painter.drawPath(path)
        elapsed = self.controller.state.elapsed if self.controller else "02:42" if running else "00:00"
        if self.controller:
            remaining = self.controller.state.remaining if running else "00:00" if completed else "—"
            completion = self.controller.state.completion_time if running or completed else "—"
        else:
            remaining = "14:12" if running else "—"
            completion = "4:20 PM" if running else "—"
        metric_y = rect.bottom() - 22
        column_width = rect.width() / 3
        for index, value in enumerate((elapsed, remaining, completion)):
            cell = QRectF(rect.left() + index * column_width, metric_y, column_width, 19)
            self._text(painter, value, cell, 13, COLORS["ivory"])

    def _paint_actuation(self, painter: QPainter) -> None:
        rect = content_rect(self.apertures["actuation"])
        running = self.controller.state.running if self.controller else self.running
        completed = self.controller.state.completed if self.controller else self.completed
        if running:
            title = "CANCELLING" if self.controller and self.controller.state.cancelling else "ENGAGED"
            detail, color = "SAFE INTERRUPT", COLORS["cyan"]
        elif completed:
            title, detail, color = "COMPLETE", "REVIEW ARTIFACT", COLORS["cyan"]
        else:
            title, detail, color = "INVOKE", "INSTRUMENT DORMANT", COLORS["brass"]
        self._line(painter, title, rect, rect.top() + 53, 29, color)
        self._line(painter, detail, rect, rect.top() + 111, 14, COLORS["ivory"])

    def _paint_progress(self, painter: QPainter) -> None:
        overall = self.controller.state.overall_progress if self.controller else self.progress
        running = self.controller.state.running if self.controller else self.running
        stage = self.controller.state.stage_progress if self.controller else min(1.0, self.progress * 2.4 % 1.0 if running else self.progress)
        for aperture_id, fraction in (("overall_progress", overall), ("stage_progress", stage)):
            rect = content_rect(self.apertures[aperture_id])
            painter.fillRect(rect, QColor("#081012"))
            if fraction > 0:
                painter.fillRect(QRectF(rect.left(), rect.top(), rect.width() * fraction, rect.height()), QColor("#3c959f"))
                painter.setPen(QPen(COLORS["cyan_bright"], 1))
                painter.drawLine(rect.topLeft(), QPointF(rect.left() + rect.width() * fraction, rect.top()))

    def _paint_stations(self, painter: QPainter) -> None:
        aperture = self.apertures["stations"]
        running = self.controller.state.running if self.controller else self.running
        completed = self.controller.state.completed if self.controller else self.completed
        completed_count = self.controller.state.active_stage_index if self.controller else 8 if completed else int(self.progress * 8)
        if completed:
            completed_count = len(aperture["centers"])
        for index, (x, y) in enumerate(aperture["centers"]):
            active = running and index == min(7, completed_count)
            color = COLORS["cyan_bright"] if active else COLORS["cyan"] if index < completed_count else COLORS["dormant"]
            painter.setBrush(color)
            painter.setPen(QPen(COLORS["brass_dim"], 2))
            painter.drawEllipse(QPointF(x, y), 9, 9)

    def _paint_curator(self, painter: QPainter) -> None:
        rect = content_rect(self.apertures["curator"])
        for index in range(6):
            column, row = index % 3, index // 3
            box = QRectF(rect.left() + column * rect.width() / 3, rect.top() + row * rect.height() / 2, rect.width() / 3, rect.height() / 2)
            completed = self.controller.state.completed if self.controller else self.completed
            painter.setBrush(COLORS["cyan"] if index == 0 and completed else COLORS["dormant"])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(box.left() + 12, box.center().y()), 4, 4)

    def _paint_ledger(self, painter: QPainter) -> None:
        rect = content_rect(self.apertures["ledger"])
        note = self.controller.state.summary if self.controller else "INVOCATION IN PROGRESS — NO ANOMALIES" if self.running else "NO ANOMALIES REQUIRING INTERVENTION"
        self._marquee_line(painter, note, QRectF(rect.left() + 10, rect.top(), rect.width() - 20, rect.height()), rect.top() + 8, 14)

    def _paint_focus(self, painter: QPainter) -> None:
        if not self.keyboard_focus_visible or not self.hasFocus() or self.focused_control == "service":
            return
        aperture_ids = ("materials", "materials_secondary") if self.focused_control == "materials" else (self.focused_control,)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for aperture_id in aperture_ids:
            if aperture_id in self.apertures:
                aperture = self.apertures[aperture_id]
                path = aperture_path(aperture)
                edge = QColor(COLORS["cyan_bright"])
                edge.setAlpha(190)
                painter.setPen(QPen(edge, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                if aperture["shape"] == "ellipse":
                    bounds = path.boundingRect().adjusted(8, 8, -8, -8)
                    for start in (35, 125, 215, 305):
                        painter.drawArc(bounds, start * 16, 20 * 16)
                    continue
                painter.save()
                painter.setClipPath(path)
                bounds = path.boundingRect().adjusted(7, 7, -7, -7)
                length = min(14.0, bounds.width() / 8, bounds.height() / 8)
                left, right, top, bottom = bounds.left(), bounds.right(), bounds.top(), bounds.bottom()
                segments = (
                    (QPointF(left, top + length), QPointF(left, top), QPointF(left + length, top)),
                    (QPointF(right - length, top), QPointF(right, top), QPointF(right, top + length)),
                    (QPointF(left, bottom - length), QPointF(left, bottom), QPointF(left + length, bottom)),
                    (QPointF(right - length, bottom), QPointF(right, bottom), QPointF(right, bottom - length)),
                )
                for first, corner, last in segments:
                    painter.drawLine(first, corner)
                    painter.drawLine(corner, last)
                painter.restore()

    def _control_at(self, point: QPointF) -> str | None:
        for control in ("apparatus", "materials", "materials_secondary", "scrutiny", "calibration", "actuation", "curator", "ledger"):
            if aperture_path(self.apertures[control]).contains(point):
                return "materials" if control == "materials_secondary" else control
        if QRectF(1410, 958, 100, 50).contains(point):
            return "service"
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        control = self._control_at(self._design_point(event.position()))
        if control in FOCUS_ORDER:
            self.focus_index = FOCUS_ORDER.index(control)
            self.keyboard_focus_visible = False
            self.setFocus()
            self._activate(control)
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in {Qt.Key.Key_Tab, Qt.Key.Key_Right, Qt.Key.Key_Down}:
            self.keyboard_focus_visible = True
            self.focus_index = (self.focus_index + 1) % len(FOCUS_ORDER)
            self.update()
            return
        if event.key() in {Qt.Key.Key_Backtab, Qt.Key.Key_Left, Qt.Key.Key_Up}:
            self.keyboard_focus_visible = True
            self.focus_index = (self.focus_index - 1) % len(FOCUS_ORDER)
            self.update()
            return
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.keyboard_focus_visible = True
            self._activate(self.focused_control)
            return
        super().keyPressEvent(event)

    def focusNextPrevChild(self, next_control: bool) -> bool:
        self.keyboard_focus_visible = True
        step = 1 if next_control else -1
        self.focus_index = (self.focus_index + step) % len(FOCUS_ORDER)
        self.setAccessibleDescription(f"Focused control: {self.focused_control}")
        self.update()
        return True

    def _activate(self, control: str) -> None:
        if self.controller is not None:
            if control == "apparatus":
                self.configuration_requested.emit()
            elif control == "scrutiny":
                self.controller.cycle_quality()
            elif control == "calibration":
                self.controller.cycle_matching()
            elif control == "materials":
                chosen, _selected = QFileDialog.getOpenFileName(
                    self, "Admit cinematic material", str(self.controller.default_media_directory()),
                    "Video files (*.mp4 *.mov *.mkv *.avi *.mpg *.mpeg);;All files (*)",
                )
                if chosen:
                    try:
                        self.controller.admit_film(Path(chosen))
                    except ValueError as exc:
                        self.controller.error.emit("Material could not be admitted", str(exc))
            elif control == "actuation":
                self.controller.cancel() if self.controller.state.running else self.controller.invoke()
            elif control == "curator":
                self.curator_requested.emit()
            elif control == "ledger":
                self.archive_requested.emit()
            elif control == "service":
                self.service_requested.emit()
            self.update()
            return
        if control == "scrutiny":
            self.quality_index = (self.quality_index + 1) % 3
        elif control == "calibration":
            self.calibration_index = (self.calibration_index + 1) % 7
        elif control == "materials":
            chosen, _selected = QFileDialog.getOpenFileName(self, "Admit cinematic material", str(self.root), "Video files (*.mp4 *.mov *.mkv *.avi);;All files (*)")
            if chosen:
                try:
                    index = self.materials.index("EMPTY CHAMBER")
                except ValueError:
                    index = 0
                self.materials[index] = Path(chosen).name.upper()
        elif control == "actuation":
            self.running = not self.running
            self.completed = False
            self.invoke_requested.emit()
        elif control == "service":
            self.service_requested.emit()
        self.update()


class ConfigurationDialog(QDialog):
    def __init__(self, controller: QtEngineController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.parameter_widgets: dict[str, QWidget] = {}
        self.setWindowTitle("Configure Cinelingus Engine")
        self.setMinimumSize(760, 660)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        apparatus_tab = QWidget()
        apparatus_form = QFormLayout(apparatus_tab)
        self.reality = QComboBox()
        self.reality.addItems(controller.realities())
        self.reality.setCurrentText(controller.state.reality)
        self.discipline = QComboBox()
        self.apparatus = QComboBox()
        self.quality = QComboBox()
        self.quality.addItems(QUALITY_LABELS)
        self.quality.setCurrentText(controller.state.quality)
        self.matching = QComboBox()
        self.matching.addItems(MATCHING_LABELS)
        self.matching.setCurrentText(controller.state.matching)
        apparatus_form.addRow("Reality", self.reality)
        apparatus_form.addRow("Discipline", self.discipline)
        apparatus_form.addRow("Apparatus", self.apparatus)
        apparatus_form.addRow("Scrutiny", self.quality)
        apparatus_form.addRow("Calibration", self.matching)
        tabs.addTab(apparatus_tab, "Apparatus")

        materials_tab = QWidget()
        materials_layout = QVBoxLayout(materials_tab)
        self.films = QListWidget()
        materials_layout.addWidget(QLabel("Film A is the anchor. Each chamber must contain a distinct complete film."))
        materials_layout.addWidget(self.films)
        material_actions = QHBoxLayout()
        admit = QPushButton("Admit Film")
        eject = QPushButton("Eject Selected")
        material_actions.addWidget(admit)
        material_actions.addWidget(eject)
        material_actions.addStretch(1)
        materials_layout.addLayout(material_actions)
        self.output = QLineEdit(str(controller.state.output_dir))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output, 1)
        browse_output = QPushButton("Choose Archive")
        output_row.addWidget(browse_output)
        materials_layout.addWidget(QLabel("Output archive"))
        materials_layout.addLayout(output_row)
        tabs.addTab(materials_tab, "Materials")

        parameters_tab = QWidget()
        self.parameters_layout = QFormLayout(parameters_tab)
        tabs.addTab(parameters_tab, "Parameters")

        recipe_actions = QHBoxLayout()
        save_recipe_button = QPushButton("Save Recipe")
        load_recipe_button = QPushButton("Load Recipe")
        recipe_actions.addWidget(save_recipe_button)
        recipe_actions.addWidget(load_recipe_button)
        recipe_actions.addStretch(1)
        layout.addLayout(recipe_actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._commit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.reality.currentTextChanged.connect(self._refresh_disciplines)
        self.discipline.currentTextChanged.connect(self._refresh_apparatuses)
        self.apparatus.currentTextChanged.connect(self._apparatus_changed)
        admit.clicked.connect(self._admit)
        eject.clicked.connect(self._eject)
        browse_output.clicked.connect(self._choose_output)
        save_recipe_button.clicked.connect(self._save_recipe)
        load_recipe_button.clicked.connect(self._load_recipe)
        self._refresh_disciplines(controller.state.discipline)
        self._set_film_rows(controller.state.films)

    def _refresh_disciplines(self, preferred: str = "") -> None:
        values = self.controller.disciplines(self.reality.currentText())
        self.discipline.blockSignals(True)
        self.discipline.clear()
        self.discipline.addItems(values)
        self.discipline.setCurrentText(preferred if preferred in values else values[0])
        self.discipline.blockSignals(False)
        self._refresh_apparatuses(self.controller.state.apparatus)

    def _refresh_apparatuses(self, preferred: str = "") -> None:
        entries = self.controller.apparatuses(self.reality.currentText(), self.discipline.currentText())
        labels = [entry.public_name + ("" if entry.invokable else " · DORMANT") for entry in entries]
        self.apparatus.blockSignals(True)
        self.apparatus.clear()
        self.apparatus.addItems(labels)
        preferred_label = next((label for label in labels if label.split(" ·", 1)[0] == preferred), labels[0])
        self.apparatus.setCurrentText(preferred_label)
        self.apparatus.blockSignals(False)
        self._apparatus_changed()

    def _selected_entry(self):
        name = self.apparatus.currentText().split(" ·", 1)[0]
        mode = self.reality.currentText()
        return next(entry for entry in self.controller.apparatuses(mode, self.discipline.currentText()) if entry.public_name == name)

    def _apparatus_changed(self) -> None:
        if not self.apparatus.currentText():
            return
        entry = self._selected_entry()
        definition = self.controller.registry.get(entry.internal_id)
        existing = self._film_paths()
        selected_count = max(definition.minimum_films, len(existing))
        if definition.maximum_films is not None:
            selected_count = min(selected_count, definition.maximum_films)
        self._set_film_rows((existing + [None] * selected_count)[:selected_count])
        self._build_parameter_form(definition)

    def _film_paths(self) -> list[Path | None]:
        values: list[Path | None] = []
        for index in range(self.films.count()):
            value = self.films.item(index).data(Qt.ItemDataRole.UserRole)
            values.append(Path(value) if value else None)
        return values

    def _set_film_rows(self, films: list[Path | None]) -> None:
        self.films.clear()
        for index, path in enumerate(films):
            label = f"FILM {chr(65 + index)}"
            visible = f"{label} · {path}" if path else f"{label} · EMPTY CHAMBER"
            self.films.addItem(visible)
            self.films.item(index).setData(Qt.ItemDataRole.UserRole, str(path) if path else "")

    def _admit(self) -> None:
        entry = self._selected_entry()
        selected = self.films.currentRow()
        if selected < 0:
            selected = next((i for i, path in enumerate(self._film_paths()) if path is None), self.films.count())
        if entry.maximum_films is not None and selected >= entry.maximum_films:
            QMessageBox.information(self, "Chambers full", f"{entry.public_name} accepts at most {entry.maximum_films} films.")
            return
        chosen, _filter = QFileDialog.getOpenFileName(
            self, "Admit cinematic material", str(self.controller.default_media_directory()),
            "Video files (*.mp4 *.mov *.mkv *.avi *.mpg *.mpeg);;All files (*)",
        )
        if not chosen:
            return
        paths = self._film_paths()
        if selected >= len(paths):
            paths.append(Path(chosen))
        else:
            paths[selected] = Path(chosen)
        self._set_film_rows(paths)
        self.films.setCurrentRow(selected)

    def _eject(self) -> None:
        row = self.films.currentRow()
        if row < 0:
            return
        paths = self._film_paths()
        minimum = self._selected_entry().minimum_films
        if len(paths) > minimum:
            paths.pop(row)
        else:
            paths[row] = None
        self._set_film_rows(paths)

    def _choose_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose output archive", self.output.text())
        if chosen:
            self.output.setText(chosen)

    def _clear_form(self) -> None:
        while self.parameters_layout.rowCount():
            self.parameters_layout.removeRow(0)
        self.parameter_widgets.clear()

    def _build_parameter_form(self, definition) -> None:
        self._clear_form()
        for parameter in definition.parameters:
            value = self.controller.state.parameters.get(parameter.id, parameter.default)
            if parameter.kind == "boolean":
                widget = QCheckBox(parameter.description)
                widget.setChecked(bool(value))
            elif parameter.kind == "choice":
                widget = QComboBox()
                widget.addItems(parameter.choices)
                widget.setCurrentText(str(value))
                widget.setToolTip(parameter.description)
            elif parameter.kind == "integer":
                widget = QSpinBox()
                widget.setRange(int(parameter.minimum if parameter.minimum is not None else -1_000_000), int(parameter.maximum if parameter.maximum is not None else 1_000_000))
                widget.setValue(int(value))
                widget.setToolTip(parameter.description)
            elif parameter.kind == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(float(parameter.minimum if parameter.minimum is not None else -1_000_000), float(parameter.maximum if parameter.maximum is not None else 1_000_000))
                widget.setValue(float(value))
                widget.setToolTip(parameter.description)
            else:
                widget = QLineEdit(str(value))
                widget.setToolTip(parameter.description)
            self.parameter_widgets[parameter.id] = widget
            self.parameters_layout.addRow(parameter.label, widget)
        if not definition.parameters:
            self.parameters_layout.addRow(QLabel("This apparatus has no adjustable parameters."))

    def _parameter_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, widget in self.parameter_widgets.items():
            if isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text()
        return values

    def _apply_configuration(self) -> bool:
        try:
            self.controller.configure(
                reality=self.reality.currentText(),
                discipline=self.discipline.currentText(),
                apparatus=self._selected_entry().public_name,
                films=self._film_paths(),
                output_dir=Path(self.output.text()),
                quality=self.quality.currentText(),
                matching=self.matching.currentText(),
                parameters=self._parameter_values(),
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Configuration could not be applied", str(exc))
            return False
        return True

    def _commit(self) -> None:
        if not self._apply_configuration():
            return
        self.accept()

    def _save_recipe(self) -> None:
        if not self._apply_configuration():
            return
        chosen, _filter = QFileDialog.getSaveFileName(
            self, "Save apparatus recipe", str(self.controller.root / "presets"),
            "Apparatus recipe (*.json);;All files (*)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        try:
            self.controller.save_recipe(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Recipe could not be saved", str(exc))

    def _load_recipe(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(
            self, "Load apparatus recipe", str(self.controller.root / "presets"),
            "Apparatus recipe (*.json);;All files (*)",
        )
        if not chosen:
            return
        try:
            notes = self.controller.load_recipe(Path(chosen))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Recipe could not be loaded", str(exc))
            return
        self.reality.setCurrentText(self.controller.state.reality)
        self._refresh_disciplines(self.controller.state.discipline)
        self.quality.setCurrentText(self.controller.state.quality)
        self.matching.setCurrentText(self.controller.state.matching)
        self.output.setText(str(self.controller.state.output_dir))
        self._set_film_rows(self.controller.state.films)
        if notes:
            QMessageBox.information(self, "Recipe loaded with notes", "\n".join(notes))


class TechnicalRecordDialog(QDialog):
    def __init__(self, controller: QtEngineController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cinelingus Technical Record")
        self.resize(900, 620)
        layout = QVBoxLayout(self)
        record = QPlainTextEdit(controller.technical_record())
        record.setReadOnly(True)
        record.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(record)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.clicked.connect(self.accept)
        layout.addWidget(buttons)


class FaceplateWindow(QMainWindow):
    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.controller = QtEngineController(root or repository_root(), self)
        self.faceplate = FaceplateWidget(root, self.controller)
        self.setWindowTitle("Cinelingus Engine")
        self.setCentralWidget(self.faceplate)
        self.resize(1536, 1024)
        self.faceplate.configuration_requested.connect(self.open_configuration)
        self.faceplate.service_requested.connect(self.open_service)
        self.faceplate.archive_requested.connect(self.open_archive)
        self.faceplate.curator_requested.connect(self.open_curator)
        self.controller.error.connect(lambda title, detail: QMessageBox.critical(self, title, detail))
        self.controller.run_finished.connect(lambda output: self.statusBar().showMessage(f"Artifact archived: {output}", 15000))
        self._close_when_stopped = False
        self.controller.changed.connect(self._finish_deferred_close)

    def open_configuration(self) -> None:
        ConfigurationDialog(self.controller, self).exec()

    def open_record(self) -> None:
        TechnicalRecordDialog(self.controller, self).exec()

    def open_archive(self) -> None:
        target = self.controller.state.last_output or self.controller.state.output_dir
        try:
            open_path_or_reveal(Path(target))
        except OSError as exc:
            QMessageBox.critical(self, "Archive could not be opened", str(exc))

    def open_curator(self) -> None:
        if self.controller.state.last_output is None:
            QMessageBox.information(self, "Curator", "No completed artifact has yet been indexed.")
            return
        self.open_archive()

    def open_service(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Cinelingus Service")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Configuration, diagnostics, and archive access"))
        configure = QPushButton("Configure Apparatus")
        record = QPushButton("Technical Record")
        archive = QPushButton("Open Archive")
        clear_cache = QPushButton("Clear Analysis Cache")
        close = QPushButton("Close")
        for button in (configure, record, archive, clear_cache, close):
            layout.addWidget(button)
        configure.clicked.connect(lambda: (dialog.accept(), self.open_configuration()))
        record.clicked.connect(lambda: (dialog.accept(), self.open_record()))
        archive.clicked.connect(lambda: (dialog.accept(), self.open_archive()))
        clear_cache.clicked.connect(lambda: self._clear_cache(dialog))
        close.clicked.connect(dialog.accept)
        dialog.exec()

    def _clear_cache(self, dialog: QDialog) -> None:
        if self.controller.worker_active:
            QMessageBox.information(dialog, "Cache in use", "The analysis cache cannot be cleared during an invocation.")
            return
        answer = QMessageBox.warning(
            dialog,
            "Clear analysis cache",
            f"Remove pipeline-owned analysis data from {self.controller.base_config.cache_dir}? Source films and archived output are preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            result = clear_pipeline_cache(self.controller.base_config.cache_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(dialog, "Cache could not be cleared", str(exc))
            return
        QMessageBox.information(
            dialog,
            "Analysis cache cleared",
            f"Removed {result['files_removed']} files and {result['directories_removed']} directories.",
        )

    def closeEvent(self, event) -> None:
        if self.controller.worker_active:
            answer = QMessageBox.question(
                self,
                "Invocation in progress",
                "Request cancellation and close after the current safe boundary?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.controller.cancel()
            self._close_when_stopped = True
            self.statusBar().showMessage("Waiting for the pipeline to reach a safe cancellation boundary…")
            event.ignore()
            return
        event.accept()

    def _finish_deferred_close(self) -> None:
        if self._close_when_stopped and not self.controller.worker_active:
            self._close_when_stopped = False
            self.close()


def render_screenshot(path: Path, *, state: str = "rest", scale: float = 1.0) -> Path:
    widget = FaceplateWidget()
    widget.set_demo_state(state)
    width, height = round(DESIGN_WIDTH * scale), round(DESIGN_HEIGHT * scale)
    widget.resize(width, height)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("#050708"))
    widget.render(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path)):
        raise RuntimeError(f"Unable to save faceplate screenshot: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch or capture the Cinelingus Qt machine interface.")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--state", choices=("rest", "ready", "running", "complete"), default="rest")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--windowed", action="store_true", help="Open at the canonical design size instead of maximized.")
    args = parser.parse_args(argv)
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_application_style(app)
    if args.screenshot:
        print(render_screenshot(args.screenshot, state=args.state, scale=max(0.5, args.scale)))
        return 0
    window = FaceplateWindow()
    window.show() if args.windowed else window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
