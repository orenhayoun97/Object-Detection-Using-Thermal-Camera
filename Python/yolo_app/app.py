"""YOLO Vision Simple — entry point."""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from .player_page import PlayerPage
from .setup_page import SetupPage
from .style import QSS


class MainWindow(QMainWindow):
    def __init__(self, qs: QSettings):
        super().__init__()
        self.setWindowTitle("YOLO Vision")
        self.resize(1280, 820)
        self.setMinimumSize(900, 600)
        self.qs = qs
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setup = SetupPage(qs)
        self.setup.start_requested.connect(self._open_player)
        self.stack.addWidget(self.setup)
        self.player: PlayerPage | None = None

    def _open_player(self, model, path, device, source):
        self.player = PlayerPage(model, path, device, source, self.qs)
        self.player.back_requested.connect(self._back)
        self.stack.addWidget(self.player)
        self.stack.setCurrentWidget(self.player)

    def _back(self):
        page, self.player = self.player, None
        self.stack.setCurrentWidget(self.setup)
        if page is not None:
            self.stack.removeWidget(page)
            page.deleteLater()

    def closeEvent(self, e):
        if self.player is not None:
            self.player.shutdown()
        loader = self.setup.loader
        if loader is not None and loader.isRunning():
            loader.wait(10000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Vision")
    icon = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo_vision", "assets", "icon.png")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = MainWindow(QSettings("YOLOVision", "YOLOSimple"))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
