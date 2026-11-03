"""Main screen: the video, a thin options bar on top and playback controls below."""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime

from PySide6.QtCore import QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QSlider, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget, QWidgetAction,
)

from .draw import VIEW_OPTIONS, Options
from .engine import DetectionWorker, Source, fmt_time

LOG_DIR = os.path.expanduser("~/Documents/YOLO App/logs")
REC_DIR = os.path.expanduser("~/Movies/YOLO App")
STEP_SEC = 5.0


def _qbool(v, default: bool) -> bool:
    return default if v is None else v in (True, "true", "1", 1)


class VideoView(QWidget):
    def __init__(self):
        super().__init__()
        self.img: QImage | None = None
        self.setMinimumSize(320, 180)

    def set_image(self, img: QImage | None):
        if img is not None:
            self.img = img
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.img is None:
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        iw, ih = self.img.width(), self.img.height()
        k = min(self.width() / iw, self.height() / ih)
        w, h = iw * k, ih * k
        p.drawImage(QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), self.img)


def _popup_button(text: str, content: QWidget) -> QToolButton:
    """A toolbar button that opens `content` in a dropdown that stays open while it's used."""
    btn = QToolButton()
    btn.setText(text)
    btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(btn)
    act = QWidgetAction(menu)
    act.setDefaultWidget(content)
    menu.addAction(act)
    btn.setMenu(menu)
    return btn


class ClassList(QWidget):
    """Search box, All / None, and a checkbox per class."""
    changed = Signal()

    def __init__(self, names: dict, selected: set[int] | None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search…")
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        row = QHBoxLayout()
        for label, state in (("All", True), ("None", False)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=state: self._set_all(s))
            row.addWidget(b)
        lay.addLayout(row)
        self.list = QListWidget()
        self.list.setFixedSize(240, 320)
        for idx, name in sorted(names.items()):
            it = QListWidgetItem(name)
            it.setData(Qt.ItemDataRole.UserRole, idx)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = selected is None or idx in selected
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(lambda _: self.changed.emit())
        lay.addWidget(self.list)

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _filter(self, text: str):
        for it in self._items():
            it.setHidden(text.lower() not in it.text().lower())

    def _set_all(self, on: bool):
        self.list.blockSignals(True)
        for it in self._items():
            if not it.isHidden():
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.changed.emit()

    def selected(self) -> set[int] | None:
        """None when every class is checked."""
        items = self._items()
        sel = {it.data(Qt.ItemDataRole.UserRole) for it in items if it.checkState() == Qt.CheckState.Checked}
        return None if len(sel) == len(items) else sel


class LogDialog(QDialog):
    def __init__(self, rows: list, file_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection log")
        self.resize(640, 480)
        self.rows = rows
        lay = QVBoxLayout(self)
        info = QLabel(f"Saved automatically to: {file_path}")
        info.setObjectName("Muted")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setWordWrap(True)
        lay.addWidget(info)
        self.table = QTableWidget(0, len(DetectionWorker.LOG_HEADER))
        self.table.setHorizontalHeaderLabels(["Time", "Video time", "Class", "Track ID", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addStretch()
        save = QPushButton("Save CSV…")
        save.clicked.connect(self._save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(save)
        row.addWidget(close)
        lay.addLayout(row)
        self.add_rows(rows)

    def add_rows(self, rows: list):
        at_end = self.table.verticalScrollBar().value() == self.table.verticalScrollBar().maximum()
        for r in rows:
            n = self.table.rowCount()
            self.table.insertRow(n)
            for c, v in enumerate(r):
                self.table.setItem(n, c, QTableWidgetItem(str(v)))
        if at_end:
            self.table.scrollToBottom()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save log", os.path.expanduser("~/detections.csv"), "CSV (*.csv)")
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(DetectionWorker.LOG_HEADER)
                w.writerows(self.rows)


class PlayerPage(QWidget):
    back_requested = Signal()

    def __init__(self, model, model_path: str, device: str, source: Source, qs: QSettings, parent=None):
        super().__init__(parent)
        self.qs = qs
        self.source = source
        self.live = source.live
        self.names = model.names
        self.model_key = f"classes/{os.path.basename(model_path)}"
        self.rows: list = []
        self.log_dialog: LogDialog | None = None
        self._pos, self._total, self._fps = 0, 0, 30.0
        self._ended = False
        self._review_t: float | None = None  # live camera: time being reviewed, None = live
        self._review_playing = False
        self._last_tick = time.monotonic()
        self._dragging = False

        saved = qs.value(self.model_key)
        classes = {int(x) for x in saved.split(",") if x} if isinstance(saved, str) else None
        opts = Options(conf=float(qs.value("conf", 0.35)), classes=classes,
                       tracking=_qbool(qs.value("tracking"), False),
                       **{k: _qbool(qs.value(k), True) for k, _ in VIEW_OPTIONS})
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = os.path.splitext(source.describe())[0].replace(" ", "_")
        self.stem = f"{stem}_{stamp}"
        self.log_path = os.path.join(LOG_DIR, f"{self.stem}.csv")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._top_bar(opts))
        self.view = VideoView()
        lay.addWidget(self.view, 1)
        lay.addWidget(self._bottom_bar())

        self.worker = DetectionWorker(model, model_path, device, source, opts, self.log_path)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.position.connect(self._on_position)
        self.worker.ended.connect(self._on_ended)
        self.worker.error.connect(self._on_error)
        self.worker.warning.connect(self.status.setText)
        self.worker.recording.connect(self._on_recording)
        self.worker.logged.connect(self._on_logged)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        if self.live:
            self.timer.start(33)

        for key, fn in ((Qt.Key.Key_Space, self._play_pause), (Qt.Key.Key_Left, self._back5),
                        (Qt.Key.Key_Right, self._fwd5)):
            QShortcut(QKeySequence(key), self, activated=fn)

    # ---------------------------------------------------------------- layout
    def _top_bar(self, opts: Options) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Bar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 8, 10, 8)
        back = QPushButton("← Settings")
        back.clicked.connect(self._back)
        row.addWidget(back)
        row.addSpacing(12)

        self.classes = ClassList(self.names, opts.classes)
        self.classes.changed.connect(self._classes_changed)
        row.addWidget(_popup_button("Objects ▾", self.classes))

        view_btn = QToolButton()
        view_btn.setText("View ▾")
        view_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(view_btn)
        for key, label in VIEW_OPTIONS:
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(getattr(opts, key))
            a.toggled.connect(lambda on, k=key: self._set_option(k, on))
        view_btn.setMenu(menu)
        row.addWidget(view_btn)

        conf_box = QWidget()
        cl = QHBoxLayout(conf_box)
        cl.setContentsMargins(10, 8, 10, 8)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(round(opts.conf * 100))
        self.conf_slider.setFixedWidth(200)
        self.conf_label = QLabel(f"{opts.conf:.2f}")
        self.conf_slider.valueChanged.connect(self._conf_changed)
        cl.addWidget(self.conf_slider)
        cl.addWidget(self.conf_label)
        self.conf_btn = _popup_button(f"Confidence {opts.conf:.2f} ▾", conf_box)
        row.addWidget(self.conf_btn)

        row.addSpacing(12)
        mode = QButtonGroup(self)
        for label, track in (("Detect", False), ("Track", True)):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(opts.tracking == track)
            b.clicked.connect(lambda _=False, t=track: self._set_option("tracking", t))
            mode.addButton(b)
            row.addWidget(b)

        row.addSpacing(12)
        self.rec_btn = QPushButton("● Rec")
        self.rec_btn.setObjectName("Rec")
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self._toggle_rec)
        row.addWidget(self.rec_btn)
        self.log_btn = QPushButton("Log (0)")
        self.log_btn.clicked.connect(self._show_log)
        row.addWidget(self.log_btn)

        self.status = QLabel(self.source.describe())
        self.status.setObjectName("Muted")
        row.addWidget(self.status, 1, Qt.AlignmentFlag.AlignRight)
        return bar

    def _bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Bar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 8, 10, 8)
        self.back_btn = QPushButton("⏪ 5s")
        self.back_btn.clicked.connect(self._back5)
        self.play_btn = QPushButton("⏸")
        self.play_btn.setFixedWidth(52)
        self.play_btn.clicked.connect(self._play_pause)
        self.fwd_btn = QPushButton("5s ⏩")
        self.fwd_btn.clicked.connect(self._fwd5)
        self.fwd_btn.setEnabled(not self.live)  # a live camera can only go back
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setMinimumWidth(140)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for w in (self.back_btn, self.play_btn, self.fwd_btn):
            row.addWidget(w)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time_label)
        if self.live:
            self.live_btn = QPushButton("● LIVE")
            self.live_btn.setCheckable(True)
            self.live_btn.setChecked(True)
            self.live_btn.clicked.connect(self._go_live)
            row.addWidget(self.live_btn)
        return bar

    # ---------------------------------------------------------------- options
    def _set_option(self, key: str, value):
        self.worker.set_options(**{key: value})
        self.qs.setValue(key, value)

    def _classes_changed(self):
        sel = self.classes.selected()
        self.worker.set_options(classes=sel)
        if sel is None:
            self.qs.remove(self.model_key)
        else:
            self.qs.setValue(self.model_key, ",".join(map(str, sorted(sel))))

    def _conf_changed(self, v: int):
        conf = v / 100
        self.conf_label.setText(f"{conf:.2f}")
        self.conf_btn.setText(f"Confidence {conf:.2f} ▾")
        self._set_option("conf", conf)

    # ---------------------------------------------------------------- recording and log
    def _toggle_rec(self, on: bool):
        if on:
            self.worker.start_recording(os.path.join(REC_DIR, f"{self.stem}_{datetime.now():%H%M%S}.mp4"))
        else:
            self.worker.stop_recording()

    def _on_recording(self, on: bool, path: str):
        self.rec_btn.setChecked(on)
        self.rec_btn.setText("■ Stop" if on else "● Rec")
        self.status.setText("Recording…" if on else f"Saved: {path}")

    def _on_logged(self, rows: list):
        self.rows.extend(rows)
        self.log_btn.setText(f"Log ({len(self.rows)})")
        if self.log_dialog is not None:
            self.log_dialog.add_rows(rows)

    def _show_log(self):
        if self.log_dialog is None:
            self.log_dialog = LogDialog(self.rows, self.log_path, self)
            self.log_dialog.finished.connect(lambda _: setattr(self, "log_dialog", None))
        self.log_dialog.show()
        self.log_dialog.raise_()

    # ---------------------------------------------------------------- frames
    def _on_frame(self):
        img = self.worker.take_frame()
        if self._review_t is None:
            self.view.set_image(img)

    def _on_position(self, pos: int, total: int, fps: float):
        self._pos, self._total, self._fps = pos, total, fps
        if not self._dragging:
            self.slider.setRange(0, max(0, total - 1))
            self.slider.setValue(pos)
        self.time_label.setText(f"{fmt_time(pos / fps)} / {fmt_time(total / fps)}")

    def _on_ended(self):
        self._ended = True
        self.play_btn.setText("▶")

    def _on_error(self, msg: str):
        self.status.setText(msg)
        QMessageBox.warning(self, "YOLO Vision", msg)

    # ---------------------------------------------------------------- playback
    def _play_pause(self):
        if self.live:
            span = self.worker.buffer_span()
            if span is None:
                return
            if self._review_t is None:  # pausing live = freezing the newest frame
                self._review_t, self._review_playing = span[1], False
            else:
                self._review_playing = not self._review_playing
            self._update_live_ui()
            return
        if self._ended:
            self._ended = False
            self.worker.seek(0)
            self.worker.set_paused(False)
        else:
            self.worker.set_paused(not self.worker.paused)
        self.play_btn.setText("▶" if self.worker.paused else "⏸")

    def _back5(self):
        if self.live:
            span = self.worker.buffer_span()
            if span is not None:
                t = span[1] if self._review_t is None else self._review_t
                if self._review_t is None:
                    self._review_playing = True
                self._review_t = max(span[0], t - STEP_SEC)
                self._update_live_ui()
        else:
            self._seek_to(self._pos - int(STEP_SEC * self._fps))

    def _fwd5(self):
        if not self.live:
            self._seek_to(self._pos + int(STEP_SEC * self._fps))

    def _seek_to(self, frame: int):
        frame = max(0, min(frame, max(0, self._total - 1)))
        self._ended = False
        self._pos = frame
        self.worker.seek(frame)

    def _slider_moved(self, v: int):
        if self.live:
            span = self.worker.buffer_span()
            if span is not None:
                self._review_t = span[0] + v / 1000
                self._update_live_ui()
        else:
            self._seek_to(v)

    def _slider_released(self):
        self._dragging = False
        if self.live and self._review_t is not None:
            span = self.worker.buffer_span()
            if span is not None and span[1] - self._review_t < 0.3:
                self._go_live()

    def _go_live(self):
        self._review_t, self._review_playing = None, False
        self._update_live_ui()

    def _tick(self):
        """Live camera only: moves through the rewind buffer and keeps the slider in sync."""
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        span = self.worker.buffer_span()
        if span is None:
            return
        t0, t1 = span
        if self._review_t is not None:
            if self._review_playing and not self._dragging:
                self._review_t += dt
            if self._review_t >= t1:
                self._go_live()
                return
            self._review_t = max(t0, self._review_t)
            self.view.set_image(self.worker.buffered_frame(self._review_t))
        if not self._dragging:
            self.slider.setRange(0, int((t1 - t0) * 1000))
            self.slider.setValue(int(((self._review_t or t1) - t0) * 1000))
        behind = 0.0 if self._review_t is None else t1 - self._review_t
        self.time_label.setText("LIVE" if self._review_t is None else f"−{fmt_time(behind)}")

    def _update_live_ui(self):
        is_live = self._review_t is None
        self.live_btn.setChecked(is_live)
        self.play_btn.setText("⏸" if is_live or self._review_playing else "▶")
        self._tick()

    # ---------------------------------------------------------------- lifecycle
    def _back(self):
        self.shutdown()
        self.back_requested.emit()

    def shutdown(self):
        self.timer.stop()
        if self.worker.isRunning():
            self.worker.stop()
