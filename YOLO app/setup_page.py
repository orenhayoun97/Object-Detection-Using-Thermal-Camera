"""Settings screen: choose the model and the input (video file or camera)."""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from .engine import ModelLoader, Source, pick_device

VIDEO_EXT = "*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.wmv"
DEFAULT_MODEL = "yolo26n.pt"  # downloaded automatically by ultralytics if missing


def _restyle(w):
    w.style().unpolish(w)
    w.style().polish(w)


class SetupPage(QWidget):
    start_requested = Signal(object, str, str, object)  # model, model path, device, Source

    def __init__(self, qs: QSettings, parent=None):
        super().__init__(parent)
        self.qs = qs
        self.model = None
        self.loaded_path = None
        self.device = pick_device()
        self.loader: ModelLoader | None = None
        self.video_path = qs.value("video_path", "")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch()
        col = QVBoxLayout()
        col.setSpacing(16)
        box = QWidget()
        box.setFixedWidth(560)
        box.setLayout(col)
        outer.addWidget(box, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch()

        title = QLabel("YOLO Vision")
        title.setObjectName("Title")
        col.addWidget(title)

        # --- model ---
        mg = QGroupBox("Model")
        ml = QVBoxLayout(mg)
        row = QHBoxLayout()
        self.model_edit = QLineEdit(qs.value("model_path", DEFAULT_MODEL))
        self.model_edit.setPlaceholderText("Model file (.pt, .onnx …)")
        self.model_edit.returnPressed.connect(self._load_model)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_model)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_model)
        row.addWidget(self.model_edit, 1)
        row.addWidget(browse)
        row.addWidget(self.load_btn)
        ml.addLayout(row)
        self.model_status = QLabel("No model loaded")
        self.model_status.setObjectName("Muted")
        self.model_status.setWordWrap(True)
        ml.addWidget(self.model_status)
        col.addWidget(mg)

        # --- input ---
        ig = QGroupBox("Input")
        il = QVBoxLayout(ig)
        row = QHBoxLayout()
        self.kind_group = QButtonGroup(self)
        self.video_btn = QPushButton("Video file")
        self.camera_btn = QPushButton("Camera")
        for b in (self.video_btn, self.camera_btn):
            b.setCheckable(True)
            self.kind_group.addButton(b)
            row.addWidget(b, 1)
        self.video_btn.clicked.connect(self._choose_video)
        self.camera_btn.clicked.connect(self._validate)
        il.addLayout(row)
        row = QHBoxLayout()
        self.src_label = QLabel()
        self.src_label.setObjectName("Muted")
        row.addWidget(self.src_label, 1)
        self.cam_label = QLabel("Camera #")
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 9)
        self.cam_index.setValue(int(qs.value("camera_index", 0)))
        row.addWidget(self.cam_label)
        row.addWidget(self.cam_index)
        il.addLayout(row)
        col.addWidget(ig)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        col.addWidget(self.start_btn, 0, Qt.AlignmentFlag.AlignRight)

        (self.camera_btn if qs.value("source_kind", "video") == "camera" else self.video_btn).setChecked(True)
        self._validate()

    # ---- model ----
    def _set_status(self, text: str, state: str = ""):
        self.model_status.setText(text)
        self.model_status.setProperty("state", state)
        _restyle(self.model_status)

    def _browse_model(self):
        start = os.path.dirname(self.model_edit.text()) or os.path.expanduser("~/Downloads")
        path, _ = QFileDialog.getOpenFileName(self, "Choose a model", start,
                                              "YOLO models (*.pt *.onnx *.engine *.torchscript);;All files (*)")
        if path:
            self.model_edit.setText(path)
            self._load_model()

    def _load_model(self):
        path = self.model_edit.text().strip()
        if not path or (self.loader and self.loader.isRunning()):
            return
        self.model = None
        self.load_btn.setEnabled(False)
        self._set_status("Loading…")
        self.loader = ModelLoader(path, self.device)
        self.loader.done.connect(lambda m, err: self._model_loaded(m, err, path))
        self.loader.start()
        self._validate()

    def _model_loaded(self, model, err, path):
        self.load_btn.setEnabled(True)
        if model is None:
            self._set_status(f"Could not load the model: {err}", "error")
        else:
            self.model, self.loaded_path = model, path
            self.qs.setValue("model_path", path)
            self._set_status(f"✓ {os.path.basename(path)} · {len(model.names)} classes · {model.task} · {self.device}",
                             "ok")
        self._validate()

    # ---- input ----
    def _choose_video(self):
        start = os.path.dirname(self.video_path) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", start, f"Videos ({VIDEO_EXT});;All files (*)")
        if path:
            self.video_path = path
        self._validate()

    def _validate(self):
        is_cam = self.camera_btn.isChecked()
        self.cam_label.setVisible(is_cam)
        self.cam_index.setVisible(is_cam)
        if is_cam:
            self.src_label.setText("Live camera")
        else:
            self.src_label.setText(os.path.basename(self.video_path) if self.video_path
                                   else "Click “Video file” to choose a file")
        source_ok = is_cam or os.path.isfile(self.video_path)
        self.start_btn.setEnabled(self.model is not None and source_ok)

    def _start(self):
        path = self.model_edit.text().strip()
        if path != self.loaded_path:  # the path was edited after loading
            self._load_model()
            return
        kind = "camera" if self.camera_btn.isChecked() else "video"
        self.qs.setValue("source_kind", kind)
        self.qs.setValue("video_path", self.video_path)
        self.qs.setValue("camera_index", self.cam_index.value())
        self.start_requested.emit(self.model, path, self.device, Source(kind, self.video_path, self.cam_index.value()))

    def showEvent(self, e):
        super().showEvent(e)
        if self.model is None and not (self.loader and self.loader.isRunning()) and self.model_edit.text().strip():
            self._load_model()  # auto-load the last used model
