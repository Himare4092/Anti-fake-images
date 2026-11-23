import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import tempfile
import traceback
import numpy as np
import cv2
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QSlider,
    QProgressBar,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QFrame,
)

from engine.analyzer import FakeImageAnalyzer, AnalysisResult


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #12141a;
    color: #e6e8ef;
    font-family: "Yu Gothic UI", "Meiryo", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a2f3d;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #9aa5ce;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #3d59a1;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #4f6ec2;
}
QPushButton:disabled {
    background-color: #292e3b;
    color: #6b7280;
}
QComboBox {
    background-color: #1b1f2a;
    border: 1px solid #2e3446;
    border-radius: 5px;
    padding: 5px 10px;
    color: #e6e8ef;
}
QProgressBar {
    border: 1px solid #2a2f3d;
    border-radius: 5px;
    text-align: center;
    background-color: #1b1f2a;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #7aa2f7;
    border-radius: 4px;
}
QTextEdit {
    background-color: #181b24;
    border: 1px solid #2a2f3d;
    border-radius: 6px;
    padding: 6px;
    color: #cbd5e1;
}
"""


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class AnalyzeWorker(QThread):
    progress = Signal(str)
    finished_result = Signal(object)
    error = Signal(str)

    def __init__(self, analyzer: FakeImageAnalyzer, image_path: str):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path

    def run(self):
        try:
            self.analyzer.progress_callback = self.progress.emit
            self.analyzer.ai_detector.progress_callback = self.progress.emit
            result = self.analyzer.analyze(self.image_path)
            self.finished_result.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ImageDropLabel(QLabel):
    file_dropped = Signal(str)

    def __init__(self, text=""):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "border: 2px dashed #2e3446; border-radius: 10px; background-color: #161922; color: #78829d;"
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
                self.file_dropped.emit(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anti-Fake Image Detector (高精度フェイク画像検出)")
        self.resize(1280, 820)
        self.setStyleSheet(DARK_STYLESHEET)

        self.analyzer = FakeImageAnalyzer()
        self.current_image_path: str | None = None
        self.orig_np: np.ndarray | None = None
        self.current_result: AnalysisResult | None = None
        self.worker: AnalyzeWorker | None = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # 上部ツールバー
        top_bar = QHBoxLayout()
        self.btn_open = QPushButton("画像を開く (またはドラッグ＆ドロップ)")
        self.btn_open.clicked.connect(self.open_file_dialog)
        top_bar.addWidget(self.btn_open)

        self.btn_analyze = QPushButton("高精度解析を実行")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.start_analysis)
        top_bar.addWidget(self.btn_analyze)

        top_bar.addStretch()

        top_bar.addWidget(QLabel("表示モード:"))
        self.combo_view = QComboBox()
        self.combo_view.addItems(["ヒートマップ重ね合わせ", "ヒートマップのみ", "元画像のみ"])
        self.combo_view.currentIndexChanged.connect(self.update_image_display)
        top_bar.addWidget(self.combo_view)

        top_bar.addWidget(QLabel("マップ種類:"))
        self.combo_map = QComboBox()
        self.combo_map.addItems(["総合異常マップ", "AI生成注目領域", "ELA (圧縮誤差)", "ノイズ不均一性"])
        self.combo_map.currentIndexChanged.connect(self.update_image_display)
        top_bar.addWidget(self.combo_map)

        top_bar.addWidget(QLabel("透明度:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(55)
        self.slider_alpha.setFixedWidth(110)
        self.slider_alpha.valueChanged.connect(self.update_image_display)
        top_bar.addWidget(self.slider_alpha)

        main_layout.addLayout(top_bar)

        # メインスプリッター（左：画像 / 右：解析結果）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左側：画像表示
        self.image_label = ImageDropLabel("ここに画像をドラッグ＆ドロップするか、「画像を開く」をクリックしてください")
        self.image_label.setMinimumSize(640, 500)
        self.image_label.file_dropped.connect(self.load_image)
        splitter.addWidget(self.image_label)

        # 右側：結果パネル
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 0, 0, 0)

        # 総合判定ボックス
        verdict_box = QGroupBox("総合判定 (フェイク確率)")
        v_layout = QVBoxLayout(verdict_box)
        self.lbl_overall_pct = QLabel("-- %")
        self.lbl_overall_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_overall_pct.setStyleSheet("font-size: 42px; font-weight: bold; color: #9aa5ce;")
        v_layout.addWidget(self.lbl_overall_pct)

        self.lbl_verdict = QLabel("画像を読み込んで解析を実行してください")
        self.lbl_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_verdict.setStyleSheet("font-size: 15px; font-weight: bold;")
        v_layout.addWidget(self.lbl_verdict)
        right_layout.addWidget(verdict_box)

        # 詳細スコアボックス
        detail_box = QGroupBox("詳細スコア内訳")
        d_layout = QVBoxLayout(detail_box)
        self.bar_ai, self.lbl_ai = self._create_score_row(d_layout, "AI生成の可能性")
        self.bar_splice, self.lbl_splice = self._create_score_row(d_layout, "合成・切り貼りリスク")
        self.bar_ela, self.lbl_ela = self._create_score_row(d_layout, "ELA (圧縮誤差異常)")
        self.bar_noise, self.lbl_noise = self._create_score_row(d_layout, "ノイズ不均一性")
        self.bar_fft, self.lbl_fft = self._create_score_row(d_layout, "FFT周波数異常")
        right_layout.addWidget(detail_box)

        # 判定理由・メタデータ
        reason_box = QGroupBox("解析レポート・メタデータ")
        r_layout = QVBoxLayout(reason_box)
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        r_layout.addWidget(self.txt_report)
        right_layout.addWidget(reason_box, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([800, 440])
        main_layout.addWidget(splitter, stretch=1)

        # ステータスバー
        status_bar = QHBoxLayout()
        self.lbl_status = QLabel("待機中")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(220)
        status_bar.addWidget(self.lbl_status, stretch=1)
        status_bar.addWidget(self.progress_bar)
        main_layout.addLayout(status_bar)

    def _create_score_row(self, parent_layout, title: str):
        row = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setFixedWidth(150)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        lbl_val = QLabel("-- %")
        lbl_val.setFixedWidth(55)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl_title)
        row.addWidget(bar, stretch=1)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return bar, lbl_val

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "画像を選択", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        self.current_image_path = path
        self.current_result = None
        img = Image.open(path).convert("RGB")
        self.orig_np = np.asarray(img)
        self.btn_analyze.setEnabled(True)
        self.lbl_status.setText(f"読み込み完了: {path}")
        self.lbl_overall_pct.setText("-- %")
        self.lbl_overall_pct.setStyleSheet("font-size: 42px; font-weight: bold; color: #9aa5ce;")
        self.lbl_verdict.setText("「高精度解析を実行」を押してください")
        self.txt_report.clear()
        self.update_image_display()

    def start_analysis(self):
        if not self.current_image_path:
            return
        self.btn_open.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("解析を開始しています...")

        self.worker = AnalyzeWorker(self.analyzer, self.current_image_path)
        self.worker.progress.connect(self.lbl_status.setText)
        self.worker.finished_result.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_finished(self, result: AnalysisResult):
        self.current_result = result
        self.progress_bar.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.lbl_status.setText("解析が完了しました")

        pct = result.overall_fake_prob
        self.lbl_overall_pct.setText(f"{pct:.1f} %")
        if pct >= 70.0:
            color = "#f7768e"
        elif pct >= 40.0:
            color = "#e0af68"
        else:
            color = "#9ece6a"
        self.lbl_overall_pct.setStyleSheet(f"font-size: 42px; font-weight: bold; color: {color};")
        self.lbl_verdict.setText(result.verdict_label)
        self.lbl_verdict.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {color};")

        self._set_bar(self.bar_ai, self.lbl_ai, result.ai_prob)
        self._set_bar(self.bar_splice, self.lbl_splice, result.splice_prob)
        self._set_bar(self.bar_ela, self.lbl_ela, result.ela_score)
        self._set_bar(self.bar_noise, self.lbl_noise, result.noise_score)
        self._set_bar(self.bar_fft, self.lbl_fft, result.fft_score)

        report_lines = ["【判定理由サマリー】"]
        for r in result.summary_reasons:
            report_lines.append(f"• {r}")
        report_lines.append("\n【メタデータ検査】")
        for f in result.metadata.get("findings", []):
            report_lines.append(f"• {f}")
        self.txt_report.setPlainText("\n".join(report_lines))

        self.update_image_display()

    def on_analysis_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.lbl_status.setText(f"エラー: {err_msg}")
        self.txt_report.setPlainText(f"解析中にエラーが発生しました:\n{err_msg}")

    @staticmethod
    def _set_bar(bar: QProgressBar, lbl: QLabel, val: float):
        bar.setValue(int(round(val)))
        lbl.setText(f"{val:.1f} %")

    def update_image_display(self):
        if self.orig_np is None:
            return

        view_mode = self.combo_view.currentText()
        if self.current_result is None or view_mode == "元画像のみ":
            disp = self.orig_np
        else:
            map_name = self.combo_map.currentText()
            heatmap = self.current_result.heatmaps.get(map_name)
            if heatmap is None:
                disp = self.orig_np
            elif view_mode == "ヒートマップのみ":
                disp = heatmap
            else:
                alpha = self.slider_alpha.value() / 100.0
                disp = cv2.addWeighted(self.orig_np, 1.0 - alpha, heatmap, alpha, 0)

        pix = numpy_to_qpixmap(disp)
        scaled = pix.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()


def _run_self_test(output_log_path: str) -> int:
    """GUI初期化、フォレンジック解析、AI検出モデルの推論を一通り検証してログ出力する。"""
    import torch

    app = QApplication(sys.argv)
    win = MainWindow()
    _ = win  # GUIインスタンス化の確認

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "test_splice.png")
            rng = np.random.default_rng(42)
            base = rng.integers(80, 180, size=(256, 256, 3), dtype=np.uint8)
            # 合成風の切り貼り領域を配置
            base[64:192, 64:192] = np.array([220, 40, 40], dtype=np.uint8)
            Image.fromarray(base).save(tmp_path, format="PNG")

            analyzer = FakeImageAnalyzer()
            result = analyzer.analyze(tmp_path)

            lines = [
                "SELF_TEST_OK",
                f"cuda_available: {torch.cuda.is_available()}",
                f"overall_fake_prob: {result.overall_fake_prob:.2f}",
                f"ai_prob: {result.ai_prob:.2f}",
                f"splice_prob: {result.splice_prob:.2f}",
                f"heatmaps: {list(result.heatmaps.keys())}",
            ]
        with open(output_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return 0
    except Exception:
        with open(output_log_path, "w", encoding="utf-8") as f:
            f.write("SELF_TEST_FAILED\n")
            f.write(traceback.format_exc())
        return 1
    finally:
        app.quit()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--self-test":
        sys.exit(_run_self_test(sys.argv[2]))
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
