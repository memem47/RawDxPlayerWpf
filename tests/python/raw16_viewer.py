import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QSettings, QPointF, Signal, QEvent
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QSpinBox, QGroupBox,
    QFormLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)


# -----------------------------
# RAW size inference
# -----------------------------
COMMON_SQUARE = [128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096]
COMMON_WH = [
    (512, 256), (640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080),
    (1536, 1536), (2048, 2048)
]


def infer_shape_from_u16_count(n_pixels: int) -> Optional[Tuple[int, int]]:
    """Return (h, w) or None."""
    if n_pixels <= 0:
        return None

    r = int(math.isqrt(n_pixels))
    if r * r == n_pixels:
        # Perfect square
        return (r, r)

    # Try common squares (prefer square)
    best = None
    best_err = None
    for s in COMMON_SQUARE:
        if s * s == n_pixels:
            return (s, s)
        # near-square heuristic: compare n_pixels to s*s
        err = abs(n_pixels - s * s)
        if best_err is None or err < best_err:
            best_err = err
            best = (s, s)

    # Try common wh list exact
    for (w, h) in COMMON_WH:
        if w * h == n_pixels:
            return (h, w)

    # Try factorization: find w close to sqrt(n)
    w0 = int(math.sqrt(n_pixels))
    candidates = []
    for w in range(max(1, w0 - 2000), w0 + 2001):
        if w != 0 and n_pixels % w == 0:
            h = n_pixels // w
            candidates.append((h, w))

    if candidates:
        # Prefer "reasonable" shapes: not extremely skinny; w >= h often for images but not guaranteed
        def score(hw):
            h, w = hw
            ratio = max(h / w, w / h)
            # penalize huge ratios, and huge dims
            return ratio + 0.000001 * (h + w)

        candidates.sort(key=score)
        return candidates[0]

    # Fallback: near best square (still may display cropped/garbage if wrong)
    return best


# -----------------------------
# WL/WW mapping
# -----------------------------
def apply_window_u16_to_u8(img_u16: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """Map uint16 image to uint8 using WL/WW."""
    if ww <= 1e-6:
        ww = 1.0
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    x = img_u16.astype(np.float32)
    x = (x - low) / (high - low)  # 0..1
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x


def qimage_from_gray_u8(gray: np.ndarray) -> QImage:
    """Create QImage (8-bit grayscale) from numpy array (H,W) uint8."""
    h, w = gray.shape
    # Make sure it's contiguous
    gray_c = np.ascontiguousarray(gray)
    qimg = QImage(gray_c.data, w, h, w, QImage.Format_Grayscale8)
    # Important: copy to own memory (otherwise numpy buffer lifetime issues)
    return qimg.copy()


# -----------------------------
# Zoomable graphics view with mouse tracking
# -----------------------------
class ImageView(QGraphicsView):
    mouseImagePos = Signal(int, int, int)  # x, y, value
    mouseOut = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._img_u16: Optional[np.ndarray] = None
        self._w = 0
        self._h = 0

    def set_image(self, pixmap: QPixmap, img_u16: np.ndarray):
        self._pixmap_item.setPixmap(pixmap)
        self._img_u16 = img_u16
        self._h, self._w = img_u16.shape
        self.scene().setSceneRect(0, 0, self._w, self._h)

    def clear_image(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._img_u16 = None
        self._w = self._h = 0
        self.scene().setSceneRect(0, 0, 0, 0)

    def wheelEvent(self, event):
        if self._img_u16 is None:
            return super().wheelEvent(event)
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._img_u16 is None:
            self.mouseOut.emit()
            return

        p: QPointF = self.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        if 0 <= x < self._w and 0 <= y < self._h:
            val = int(self._img_u16[y, x])
            self.mouseImagePos.emit(x, y, val)
        else:
            self.mouseOut.emit()

    def reset_zoom(self):
        self.resetTransform()

    def get_zoom(self) -> float:
        # Approx zoom factor from transform (assumes uniform scaling)
        t = self.transform()
        return float(t.m11())

    def set_zoom(self, zoom: float):
        self.resetTransform()
        if zoom <= 0:
            zoom = 1.0
        self.scale(zoom, zoom)


# -----------------------------
# App main window
# -----------------------------
@dataclass
class LoadedRaw:
    path: str
    img_u16: np.ndarray
    shape: Tuple[int, int]  # (h,w)
    vmin: int
    vmax: int


class MainWindow(QMainWindow):
    ORG = "memem47"
    APP = "raw16_viewer_pyside6"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW16 Viewer (PySide6)")

        self.settings = QSettings(self.ORG, self.APP)

        self.raw_files: List[str] = []
        self.index: int = 0

        self.current: Optional[LoadedRaw] = None
        self.current_dir: str = ""

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # Top controls
        top = QHBoxLayout()
        v.addLayout(top)

        self.btn_select = QPushButton("RAW選択...")
        self.btn_prev = QPushButton("前の画像")
        self.btn_next = QPushButton("次の画像")
        self.btn_reset_zoom = QPushButton("ズームリセット")

        top.addWidget(self.btn_select)
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        top.addWidget(self.btn_reset_zoom)

        self.lbl_name = QLabel("（未選択）")
        self.lbl_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_name)

        # Slider for index
        nav = QHBoxLayout()
        v.addLayout(nav)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)

        self.spin_index = QSpinBox()
        self.spin_index.setMinimum(0)
        self.spin_index.setMaximum(0)
        self.spin_index.setEnabled(False)

        nav.addWidget(QLabel("選択:"))
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.spin_index)

        # Middle area: image + right panel
        mid = QHBoxLayout()
        v.addLayout(mid, 1)

        self.view = ImageView()
        mid.addWidget(self.view, 1)

        right = QVBoxLayout()
        mid.addLayout(right)

        # WL/WW group
        g = QGroupBox("WL/WW")
        f = QFormLayout(g)
        right.addWidget(g)

        self.sld_wl = QSlider(Qt.Horizontal)
        self.sld_ww = QSlider(Qt.Horizontal)
        self.spn_wl = QSpinBox()
        self.spn_ww = QSpinBox()

        self.sld_wl.setRange(0, 65535)
        self.sld_ww.setRange(1, 65535)
        self.spn_wl.setRange(0, 65535)
        self.spn_ww.setRange(1, 65535)

        f.addRow("WL", self._pair(self.sld_wl, self.spn_wl))
        f.addRow("WW", self._pair(self.sld_ww, self.spn_ww))

        # Info labels
        self.lbl_mouse = QLabel("x=?, y=?, value=?")
        self.lbl_zoom = QLabel("zoom=1.00")
        right.addWidget(QGroupBox("情報"))
        right.addWidget(self.lbl_mouse)
        right.addWidget(self.lbl_zoom)
        right.addStretch(1)

        # Menu actions (optional convenience)
        act_open = QAction("RAW選択...", self)
        act_open.triggered.connect(self.on_select_raw)
        self.menuBar().addMenu("File").addAction(act_open)

        # Signals
        self.btn_select.clicked.connect(self.on_select_raw)
        self.btn_prev.clicked.connect(lambda: self.step_index(-1))
        self.btn_next.clicked.connect(lambda: self.step_index(+1))
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)

        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_index.valueChanged.connect(self.on_spin_changed)

        self.sld_wl.valueChanged.connect(self.on_wl_changed)
        self.sld_ww.valueChanged.connect(self.on_ww_changed)
        self.spn_wl.valueChanged.connect(self.on_wl_changed)
        self.spn_ww.valueChanged.connect(self.on_ww_changed)

        self.view.mouseImagePos.connect(self.on_mouse_pos)
        self.view.mouseOut.connect(self.on_mouse_out)

        # Timer-less zoom label update: hook viewport events via mouse move + wheel is enough
        self._viewport = self.view.viewport()
        self._viewport.installEventFilter(self)

        # Load persisted state
        self.restore_state()

        # Buttons initial
        self.update_nav_enabled()

    def _pair(self, slider: QSlider, spin: QSpinBox) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return w

    def eventFilter(self, obj, event):
        # Update zoom label on wheel/resize
        if obj is self._viewport:
            et = event.type()
            if et in (QEvent.Type.Wheel, QEvent.Type.Resize, QEvent.Type.MouseMove):
                self.lbl_zoom.setText(f"zoom={self.view.get_zoom():.2f}")
        return super().eventFilter(obj, event)
    # -----------------------------
    # Persistence
    # -----------------------------
    def restore_state(self):
        last_path = self.settings.value("last_path", "", type=str)
        last_index = self.settings.value("last_index", 0, type=int)
        last_wl = self.settings.value("last_wl", 32768, type=int)
        last_ww = self.settings.value("last_ww", 65535, type=int)
        last_zoom = self.settings.value("last_zoom", 1.0, type=float)

        # Pre-set WL/WW (may be overridden after first image load)
        self._set_wl_ww_ui(last_wl, max(1, last_ww), block_signals=True)

        if last_path and os.path.isfile(last_path):
            self.build_list_from_selected(last_path)
            if self.raw_files:
                self.index = max(0, min(last_index, len(self.raw_files) - 1))
                self._sync_nav_widgets()
                self.load_and_show(self.index, keep_wlww=True)
                self.view.set_zoom(last_zoom)

    def save_state(self):
        last_path = ""
        if self.raw_files and 0 <= self.index < len(self.raw_files):
            last_path = self.raw_files[self.index]

        self.settings.setValue("last_path", last_path)
        self.settings.setValue("last_index", int(self.index))
        self.settings.setValue("last_wl", int(self.spn_wl.value()))
        self.settings.setValue("last_ww", int(self.spn_ww.value()))
        self.settings.setValue("last_zoom", float(self.view.get_zoom()))

    def closeEvent(self, event):
        self.save_state()
        super().closeEvent(event)

    # -----------------------------
    # File selection / list build
    # -----------------------------
    def on_select_raw(self):
        start_dir = self.current_dir or self.settings.value("last_dir", "", type=str) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "RAWファイルを選択",
            start_dir,
            "RAW files (*.raw *.bin);;All files (*.*)"
        )
        if not path:
            return

        self.settings.setValue("last_dir", os.path.dirname(path))
        self.build_list_from_selected(path)
        if not self.raw_files:
            QMessageBox.warning(self, "Info", "同フォルダ内に .raw/.bin が見つかりませんでした。")
            return

        self.index = self.raw_files.index(path) if path in self.raw_files else 0
        self._sync_nav_widgets()
        self.load_and_show(self.index, keep_wlww=False)

    def build_list_from_selected(self, selected_path: str):
        d = os.path.dirname(selected_path)
        self.current_dir = d
        exts = (".raw", ".bin")
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
        files.sort(key=lambda p: os.path.basename(p).lower())
        self.raw_files = files

        self.slider.setEnabled(len(files) > 0)
        self.spin_index.setEnabled(len(files) > 0)

        self.slider.setMaximum(max(0, len(files) - 1))
        self.spin_index.setMaximum(max(0, len(files) - 1))

        self.update_nav_enabled()

    # -----------------------------
    # Navigation
    # -----------------------------
    def update_nav_enabled(self):
        has = len(self.raw_files) > 0
        self.btn_prev.setEnabled(has and self.index > 0)
        self.btn_next.setEnabled(has and self.index < len(self.raw_files) - 1)
        self.slider.setEnabled(has)
        self.spin_index.setEnabled(has)

    def _sync_nav_widgets(self):
        # Block to avoid recursion
        self.slider.blockSignals(True)
        self.spin_index.blockSignals(True)
        self.slider.setValue(self.index)
        self.spin_index.setValue(self.index)
        self.slider.blockSignals(False)
        self.spin_index.blockSignals(False)
        self.update_nav_enabled()

    def on_slider_changed(self, v: int):
        if v == self.index:
            return
        self.index = v
        self.spin_index.blockSignals(True)
        self.spin_index.setValue(v)
        self.spin_index.blockSignals(False)
        self.load_and_show(self.index, keep_wlww=True)
        self.update_nav_enabled()

    def on_spin_changed(self, v: int):
        if v == self.index:
            return
        self.index = v
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.load_and_show(self.index, keep_wlww=True)
        self.update_nav_enabled()

    def step_index(self, delta: int):
        if not self.raw_files:
            return
        ni = self.index + delta
        ni = max(0, min(ni, len(self.raw_files) - 1))
        if ni == self.index:
            return
        self.index = ni
        self._sync_nav_widgets()
        self.load_and_show(self.index, keep_wlww=True)

    # -----------------------------
    # Loading / rendering
    # -----------------------------
    def load_raw_u16(self, path: str) -> LoadedRaw:
        size_bytes = os.path.getsize(path)
        if size_bytes % 2 != 0:
            raise ValueError(f"ファイルサイズが偶数ではありません（ushort16想定）: {size_bytes} bytes")

        n_pixels = size_bytes // 2
        shape = infer_shape_from_u16_count(n_pixels)
        if shape is None:
            raise ValueError("画像サイズ推定に失敗しました。")

        h, w = shape
        img = np.fromfile(path, dtype=np.uint16, count=h * w)
        if img.size != h * w:
            raise ValueError("読み込みサイズが不足しました（推定shapeが不正な可能性）")

        img = img.reshape((h, w))
        vmin = int(img.min())
        vmax = int(img.max())
        return LoadedRaw(path=path, img_u16=img, shape=shape, vmin=vmin, vmax=vmax)

    def load_and_show(self, idx: int, keep_wlww: bool):
        if not (0 <= idx < len(self.raw_files)):
            return
        path = self.raw_files[idx]
        try:
            loaded = self.load_raw_u16(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"{os.path.basename(path)}\n\n{e}")
            return

        self.current = loaded
        self.lbl_name.setText(f"[{idx+1}/{len(self.raw_files)}]  {loaded.path}  "
                              f"({loaded.shape[1]}x{loaded.shape[0]})  min={loaded.vmin} max={loaded.vmax}")

        if not keep_wlww:
            wl = int((loaded.vmin + loaded.vmax) / 2)
            ww = max(1, int(loaded.vmax - loaded.vmin))
            self._set_wl_ww_ui(wl, ww, block_signals=True)

        self.render_current()

    def render_current(self):
        if self.current is None:
            self.view.clear_image()
            return
        wl = float(self.spn_wl.value())
        ww = float(self.spn_ww.value())

        gray = apply_window_u16_to_u8(self.current.img_u16, wl, ww)
        qimg = qimage_from_gray_u8(gray)
        pix = QPixmap.fromImage(qimg)

        self.view.set_image(pix, self.current.img_u16)

    # -----------------------------
    # WL/WW sync
    # -----------------------------
    def _set_wl_ww_ui(self, wl: int, ww: int, block_signals: bool):
        wl = max(0, min(65535, int(wl)))
        ww = max(1, min(65535, int(ww)))

        widgets = [self.sld_wl, self.sld_ww, self.spn_wl, self.spn_ww]
        if block_signals:
            for w in widgets:
                w.blockSignals(True)

        self.sld_wl.setValue(wl)
        self.spn_wl.setValue(wl)
        self.sld_ww.setValue(ww)
        self.spn_ww.setValue(ww)

        if block_signals:
            for w in widgets:
                w.blockSignals(False)

    def on_wl_changed(self, v: int):
        # unify from either slider or spin
        wl = int(self.sender().value())
        ww = int(self.spn_ww.value())
        self._set_wl_ww_ui(wl, ww, block_signals=True)
        self.render_current()

    def on_ww_changed(self, v: int):
        ww = int(self.sender().value())
        wl = int(self.spn_wl.value())
        self._set_wl_ww_ui(wl, ww, block_signals=True)
        self.render_current()

    # -----------------------------
    # Mouse info
    # -----------------------------
    def on_mouse_pos(self, x: int, y: int, value: int):
        self.lbl_mouse.setText(f"x={x}, y={y}, value={value}")

    def on_mouse_out(self):
        self.lbl_mouse.setText("x=?, y=?, value=?")

    # -----------------------------
    # Zoom
    # -----------------------------
    def on_reset_zoom(self):
        self.view.reset_zoom()
        self.lbl_zoom.setText(f"zoom={self.view.get_zoom():.2f}")


def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
