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
    QFormLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QComboBox, QCheckBox
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


def normalize_minmax_to_u8(img: np.ndarray) -> np.ndarray:
    """
    任意の数値配列を min-max で 0..255 に正規化して uint8 にする。
    差分(i32)表示用。定数画像の場合は0で埋める。
    """
    x = img.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


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

        self._img: Optional[np.ndarray] = None
        self._w = 0
        self._h = 0

    def set_image(self, pixmap: QPixmap, img: np.ndarray):
        self._pixmap_item.setPixmap(pixmap)
        self._img = img
        self._h, self._w = img.shape
        self.scene().setSceneRect(0, 0, self._w, self._h)

    def clear_image(self):
        self._pixmap_item.setPixmap(QPixmap())
        self._img = None
        self._w = self._h = 0
        self.scene().setSceneRect(0, 0, 0, 0)

    def wheelEvent(self, event):
        if self._img is None:
            return super().wheelEvent(event)
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._img is None:
            self.mouseOut.emit()
            return

        p: QPointF = self.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        if 0 <= x < self._w and 0 <= y < self._h:
            val = int(self._img[y, x])
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
    ORG = "raw16_viewer"
    APP = "pyside6"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW16 Viewer (PySide6)")

        self.settings = QSettings(self.ORG, self.APP)

        self.raw_files: List[str] = []
        self.index: int = 0

        self.current: Optional[LoadedRaw] = None
        self.current_dir: str = ""

        # --- diff feature ---
        self.ref: Optional[LoadedRaw] = None          # 差分に使う画像（u16）
        self.diff_i32: Optional[np.ndarray] = None    # base - ref（i32）
        self.view_mode: str = "base"                  # "base" | "ref" | "diff"
        self.diff_auto_dir: str = ""               # 自動差分元探索に使うフォルダ（差分画像選択で決める）

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # Top controls
        top = QHBoxLayout()
        v.addLayout(top)

        self.btn_select = QPushButton("RAW選択...")
        self.btn_select_diff = QPushButton("差分画像選択...")
        self.chk_auto_diff = QCheckBox("自動ファイル選択")
        self.btn_prev = QPushButton("前の画像")
        self.btn_next = QPushButton("次の画像")
        self.btn_reset_zoom = QPushButton("ズームリセット")
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["元画像", "差分元画像", "差分画像(元-差分元)"])

        top.addWidget(self.btn_select)
        top.addWidget(self.btn_select_diff)
        top.addWidget(self.chk_auto_diff)
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        top.addWidget(self.btn_reset_zoom)
        top.addWidget(QLabel("表示:"))
        top.addWidget(self.cmb_view)

        self.lbl_name = QLabel("（未選択）")
        self.lbl_name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_name)

        self.lbl_diff = QLabel("差分元: (未選択)")
        self.lbl_diff.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_diff)

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
        # 右パネルのガタつき対策：複数行 + 等幅フォント + 幅固定
        self.lbl_mouse.setTextFormat(Qt.PlainText)
        self.lbl_mouse.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_mouse.setWordWrap(False)
        self.lbl_mouse.setMinimumWidth(320)   # ここは好みで 300〜400
        self.lbl_mouse.setFixedHeight(70)     # 2〜3行表示の高さ確保

        font = self.lbl_mouse.font()
        font.setFamily("Consolas")  # Windows 等幅
        self.lbl_mouse.setFont(font)
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
        self.btn_select_diff.clicked.connect(self.on_select_diff_raw)
        self.chk_auto_diff.stateChanged.connect(self.on_auto_diff_toggled)
        self.btn_prev.clicked.connect(lambda: self.step_index(-1))
        self.btn_next.clicked.connect(lambda: self.step_index(+1))
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)
        self.cmb_view.currentIndexChanged.connect(self.on_view_mode_changed)

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
        last_diff_path = self.settings.value("last_diff_path", "", type=str)
        last_diff_auto_enabled = self.settings.value("last_diff_auto_enabled", False, type=bool)
        last_diff_auto_dir = self.settings.value("last_diff_auto_dir", "", type=str)
        last_view_mode = self.settings.value("last_view_mode", "base", type=str)

        # Pre-set WL/WW (may be overridden after first image load)
        self._set_wl_ww_ui(last_wl, max(1, last_ww), block_signals=True)

        # 自動ファイル選択（差分元）の復元
        self.chk_auto_diff.blockSignals(True)
        self.chk_auto_diff.setChecked(bool(last_diff_auto_enabled))
        self.chk_auto_diff.blockSignals(False)
        self.diff_auto_dir = last_diff_auto_dir or ""

        if last_path and os.path.isfile(last_path):
            self.build_list_from_selected(last_path)
            if self.raw_files:
                self.index = max(0, min(last_index, len(self.raw_files) - 1))
                self._sync_nav_widgets()
                self.load_and_show(self.index, keep_wlww=True)
                self.view.set_zoom(last_zoom)
                # 差分元の復元
                # - 自動ONかつフォルダが設定されている場合：現在の元画像と同名ファイルをそのフォルダから探して設定
                # - 自動OFFの場合：最後に手動選択した差分元パスを復元（サイズが合えば）
                if self.chk_auto_diff.isChecked() and self.diff_auto_dir:
                    self._auto_select_ref_for_current(silent=True)
                elif last_diff_path and os.path.isfile(last_diff_path):
                    try:
                        ref = self.load_raw_u16(last_diff_path)
                        if ref.shape == self.current.shape:
                            self.ref = ref
                            self.lbl_diff.setText(f"差分元: {ref.path}")
                            self.diff_i32 = self.current.img_u16.astype(np.int32) - self.ref.img_u16.astype(np.int32)
                    except Exception:
                        pass

                # 表示モード復元
                mode_to_idx = {"base": 0, "ref": 1, "diff": 2}
                self.view_mode = last_view_mode if last_view_mode in mode_to_idx else "base"
                self.cmb_view.setCurrentIndex(mode_to_idx.get(self.view_mode, 0))
                self.render_by_mode()

    def save_state(self):
        last_path = ""
        if self.raw_files and 0 <= self.index < len(self.raw_files):
            last_path = self.raw_files[self.index]

        self.settings.setValue("last_path", last_path)
        self.settings.setValue("last_index", int(self.index))
        self.settings.setValue("last_wl", int(self.spn_wl.value()))
        self.settings.setValue("last_ww", int(self.spn_ww.value()))
        self.settings.setValue("last_zoom", float(self.view.get_zoom()))
        self.settings.setValue("last_diff_path", self.ref.path if self.ref is not None else "")
        self.settings.setValue("last_diff_auto_enabled", bool(self.chk_auto_diff.isChecked()))
        self.settings.setValue("last_diff_auto_dir", self.diff_auto_dir)
        self.settings.setValue("last_view_mode", self.view_mode)

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

        # 差分元の自動選択（ONなら常に「現在の元画像と同名ファイル」を差分元フォルダから選ぶ）
        if self.chk_auto_diff.isChecked() and self.diff_auto_dir:
            self._auto_select_ref_for_current(silent=True)
        else:
            # 手動選択モード：差分元が設定済みで同サイズなら、差分を再計算
            if self.ref is not None:
                if self.ref.shape == self.current.shape:
                    self.diff_i32 = self.current.img_u16.astype(np.int32) - self.ref.img_u16.astype(np.int32)
                else:
                    # サイズが合わなくなったら解除
                    self.ref = None
                    self.diff_i32 = None
                    self.lbl_diff.setText("差分元: (未選択)")
                    if self.view_mode in ("ref", "diff"):
                        self.view_mode = "base"
                        self.cmb_view.setCurrentIndex(0)

        self.render_by_mode()

    def render_current(self):
        self.render_by_mode()

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
        if self.view_mode != "diff":
            self.render_current()

    def on_ww_changed(self, v: int):
        ww = int(self.sender().value())
        wl = int(self.spn_wl.value())
        self._set_wl_ww_ui(wl, ww, block_signals=True)
        if self.view_mode != "diff":
            self.render_current()

    # -----------------------------
    # Mouse info
    # -----------------------------
    def on_mouse_pos(self, x: int, y: int, value: int):
        if self.current is None:
            self.lbl_mouse.setText("x=?, y=?, value=?")
            return

        base_v = int(self.current.img_u16[y, x])

        if self.view_mode == "diff":
            diff_v = int(value)
            if self.ref is not None:
                ref_v = int(self.ref.img_u16[y, x])
                # 改行 + 桁揃え
                self.lbl_mouse.setText(
                    f"x={x:4d}, y={y:4d}\n"
                    f"base={base_v:6d}, ref={ref_v:6d}\n"
                    f"diff={diff_v:7d}"
                )
            else:
                self.lbl_mouse.setText(
                    f"x={x:4d}, y={y:4d}\n"
                    f"base={base_v:6d}\n"
                    f"diff={diff_v:7d}"
                )
            return

        # base/ref表示中（従来）→ ここも改行にすると安定
        self.lbl_mouse.setText(
            f"x={x:4d}, y={y:4d}\n"
            f"value={int(value):6d}"
        )

    def on_mouse_out(self):
        self.lbl_mouse.setText("x=?, y=?, value=?")

    # -----------------------------
    # Zoom
    # -----------------------------
    def on_reset_zoom(self):
        self.view.reset_zoom()
        self.lbl_zoom.setText(f"zoom={self.view.get_zoom():.2f}")

    # -----------------------------
    # Diff auto selection helpers
    # -----------------------------
    def on_auto_diff_toggled(self, state: int):
        """自動ファイル選択ON/OFF時の挙動。"""
        # ONにした瞬間に、フォルダが既に分かっていれば現在の画像に合わせて差分元を合わせる
        if self.chk_auto_diff.isChecked() and self.current is not None and self.diff_auto_dir:
            self._auto_select_ref_for_current(silent=True)
            self.render_by_mode()

    def _auto_select_ref_for_current(self, silent: bool = True) -> bool:
        """現在の元画像(self.current)のファイル名と同名の差分元を self.diff_auto_dir から探して設定する。"""
        if self.current is None:
            return False
        if not self.diff_auto_dir:
            return False

        base_name = os.path.basename(self.current.path)
        cand = os.path.join(self.diff_auto_dir, base_name)

        if not os.path.isfile(cand):
            # 見つからない：ラベルだけ更新（ポップアップは出さない）
            self.ref = None
            self.diff_i32 = None
            self.lbl_diff.setText(f"差分元(自動): 見つかりません: {cand}")
            if self.view_mode in ("ref", "diff"):
                self.view_mode = "base"
                self.cmb_view.setCurrentIndex(0)
            return False

        try:
            ref = self.load_raw_u16(cand)
        except Exception as e:
            self.ref = None
            self.diff_i32 = None
            self.lbl_diff.setText(f"差分元(自動): 読み込み失敗: {cand}")
            if not silent:
                QMessageBox.critical(self, "Load error", f"{os.path.basename(cand)}\n\n{e}")
            if self.view_mode in ("ref", "diff"):
                self.view_mode = "base"
                self.cmb_view.setCurrentIndex(0)
            return False

        if self.current.shape != ref.shape:
            # サイズ不一致：自動探索ではポップアップを避けてラベル更新のみ
            self.ref = None
            self.diff_i32 = None
            self.lbl_diff.setText(
                f"差分元(自動): サイズ不一致: {cand}  (base={self.current.shape[1]}x{self.current.shape[0]}, ref={ref.shape[1]}x{ref.shape[0]})"
            )
            if not silent:
                QMessageBox.warning(
                    self,
                    "サイズ不一致",
                    f"サイズが違います。\n\n"
                    f"元: {self.current.shape[1]}x{self.current.shape[0]}\n"
                    f"差分元: {ref.shape[1]}x{ref.shape[0]}"
                )
            if self.view_mode in ("ref", "diff"):
                self.view_mode = "base"
                self.cmb_view.setCurrentIndex(0)
            return False

        self.ref = ref
        self.lbl_diff.setText(f"差分元(自動): {ref.path}")
        self.diff_i32 = self.current.img_u16.astype(np.int32) - self.ref.img_u16.astype(np.int32)
        return True

    def on_select_diff_raw(self):
        if self.current is None:
            QMessageBox.information(self, "Info", "先に元画像（RAW選択）を読み込んでください。")
            return

        start_dir = self.current_dir or self.settings.value("last_dir", "", type=str) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "差分元画像を選択",
            start_dir,
            "RAW files (*.raw *.bin);;All files (*.*)"
        )
        if not path:
            return

        # 自動選択用フォルダを更新（チェックON/OFFに関係なく「最後に選んだフォルダ」として保持）
        self.diff_auto_dir = os.path.dirname(path)
        self.settings.setValue("last_diff_auto_dir", self.diff_auto_dir)

        # 自動ファイル選択ONなら、現在の元画像と同名のファイルをこのフォルダから探して選択する
        if self.chk_auto_diff.isChecked() and self.current is not None:
            cand = os.path.join(self.diff_auto_dir, os.path.basename(self.current.path))
            if os.path.isfile(cand):
                path = cand
            else:
                QMessageBox.information(
                    self,
                    "Info",
                    "自動ファイル選択がONです。\n"
                    "差分元フォルダ内に、元画像と同名のファイルが見つかりませんでした。\n\n"
                    f"探したパス: {cand}"
                )
                self.ref = None
                self.diff_i32 = None
                self.lbl_diff.setText(f"差分元(自動): 見つかりません: {cand}")
                if self.view_mode in ("ref", "diff"):
                    self.view_mode = "base"
                    self.cmb_view.setCurrentIndex(0)
                return

        # 読み込み（u16）
        try:
            ref = self.load_raw_u16(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"{os.path.basename(path)}\n\n{e}")
            return

        # サイズチェック（②）
        if self.current.shape != ref.shape:
            QMessageBox.warning(
                self,
                "サイズ不一致",
                f"サイズが違います。\n\n"
                f"元: {self.current.shape[1]}x{self.current.shape[0]}\n"
                f"差分元: {ref.shape[1]}x{ref.shape[0]}"
            )
            self.ref = None
            self.diff_i32 = None
            self.lbl_diff.setText("差分元: (未選択)")
            return

        self.ref = ref
        # 自動ONならラベルに(自動)と出す
        if self.chk_auto_diff.isChecked():
            self.lbl_diff.setText(f"差分元(自動): {ref.path}")
        else:
            self.lbl_diff.setText(f"差分元: {ref.path}")

        # 差分計算（③）
        self.diff_i32 = self.current.img_u16.astype(np.int32) - self.ref.img_u16.astype(np.int32)

        # 設定保存（任意）
        self.settings.setValue("last_diff_path", path)

        # 表示が差分/差分元なら即更新
        self.render_by_mode()

    def on_view_mode_changed(self, idx: int):
        # 0: base, 1: ref, 2: diff
        self.view_mode = ["base", "ref", "diff"][idx]
        self.settings.setValue("last_view_mode", self.view_mode)
        self.render_by_mode()

    def render_by_mode(self):
        if self.current is None:
            self.view.clear_image()
            return

        if self.view_mode == "base":
            self.render_base_or_ref(self.current.img_u16, use_wlww=True)
            return

        if self.view_mode == "ref":
            if self.ref is None:
                QMessageBox.information(self, "Info", "差分元画像が未選択です。")
                self.cmb_view.setCurrentIndex(0)
                return
            self.render_base_or_ref(self.ref.img_u16, use_wlww=True)
            return

        # diff
        if self.ref is None or self.diff_i32 is None:
            QMessageBox.information(self, "Info", "差分元画像が未選択です。")
            self.cmb_view.setCurrentIndex(0)
            return

        # 差分表示はmin-maxで表示（WL/WWは無視）
        gray = normalize_minmax_to_u8(self.diff_i32)
        qimg = qimage_from_gray_u8(gray)
        pix = QPixmap.fromImage(qimg)
        self.view.set_image(pix, self.diff_i32)

    def render_base_or_ref(self, img_u16: np.ndarray, use_wlww: bool):
        if use_wlww:
            wl = float(self.spn_wl.value())
            ww = float(self.spn_ww.value())
            gray = apply_window_u16_to_u8(img_u16, wl, ww)
        else:
            gray = normalize_minmax_to_u8(img_u16)

        qimg = qimage_from_gray_u8(gray)
        pix = QPixmap.fromImage(qimg)
        self.view.set_image(pix, img_u16)


def main():
    app = QApplication([])
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
