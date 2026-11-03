"""One small, flat dark theme."""

ACCENT = "#4c8dff"

QSS = f"""
QWidget {{ background: #121418; color: #e6e8ec; font-size: 14px; }}
QLabel#Title {{ font-size: 24px; font-weight: 600; }}
QLabel#Muted {{ color: #8a909c; }}
QLabel[state="ok"] {{ color: #3ddc84; }}
QLabel[state="error"] {{ color: #ff6b6b; }}
QGroupBox {{ border: 1px solid #2a2e36; border-radius: 8px; margin-top: 14px; padding: 14px 12px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8a909c; }}
QPushButton, QToolButton {{ background: #22262e; border: 1px solid #2f343e; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover, QToolButton:hover {{ background: #2b303a; }}
QPushButton:checked, QToolButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton:disabled, QToolButton:disabled {{ color: #555b66; background: #1a1d23; }}
QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: 600; padding: 10px 28px; }}
QPushButton#Primary:disabled {{ background: #26324a; border-color: #26324a; color: #7d8799; }}
QPushButton#Rec:checked {{ background: #d93636; border-color: #d93636; }}
QToolButton::menu-indicator {{ image: none; }}
QLineEdit, QSpinBox, QListWidget, QTableWidget {{ background: #1a1d23; border: 1px solid #2f343e; border-radius: 6px; padding: 4px 6px; }}
QHeaderView::section {{ background: #22262e; border: none; padding: 4px; }}
QMenu {{ background: #1a1d23; border: 1px solid #2f343e; padding: 4px; }}
QMenu::item {{ padding: 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: #2b303a; }}
QSlider::groove:horizontal {{ height: 4px; background: #2f343e; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 14px; margin: -5px 0; border-radius: 7px; }}
QWidget#Bar {{ background: #181b20; }}
"""
