from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image

from .ai_detector import AIDetector
from .forensics import (
    analyze_ela,
    analyze_noise_inconsistency,
    analyze_fft_spectrum,
    analyze_metadata,
)


@dataclass
class AnalysisResult:
    overall_fake_prob: float
    ai_prob: float
    splice_prob: float
    ela_score: float
    noise_score: float
    fft_score: float
    metadata: dict
    verdict_label: str
    summary_reasons: list[str]
    heatmaps: dict[str, np.ndarray]  # RGB uint8 (H, W, 3)


def _to_color_heatmap(norm_map: np.ndarray) -> np.ndarray:
    """0.0..1.0 の2D配列を JET カラーマップの RGB uint8 画像に変換する。"""
    u8 = np.clip(norm_map * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FakeImageAnalyzer:
    def __init__(self, progress_callback=None, cache_dir: str | None = None):
        self.progress_callback = progress_callback
        self.ai_detector = AIDetector(progress_callback=progress_callback, cache_dir=cache_dir)

    def set_cache_dir(self, cache_dir: str | None):
        self.ai_detector.set_cache_dir(cache_dir)

    def _report(self, msg: str):
        if self.progress_callback:
            self.progress_callback(msg)

    def analyze(self, image_path: str) -> AnalysisResult:
        self._report("画像を読み込み中...")
        image_pil = Image.open(image_path)
        rgb_pil = image_pil.convert("RGB")
        image_np = np.asarray(rgb_pil)

        # 1. メタデータ解析
        self._report("メタデータ (EXIF / PNGチャンク) を検査中...")
        meta = analyze_metadata(image_pil)

        # 2. ELA (圧縮誤差解析)
        self._report("ELA (圧縮誤差) を解析中...")
        ela_score, ela_map = analyze_ela(rgb_pil)

        # 3. ノイズ不均一性解析
        self._report("局所ノイズ分散の不均一性を解析中...")
        noise_score, noise_map = analyze_noise_inconsistency(image_np)

        # 4. FFT周波数解析
        self._report("2次元FFT周波数スペクトルを解析中...")
        fft_score = analyze_fft_spectrum(image_np)

        # 5. Deep Learning AI生成検出 (全体 + マルチパッチ)
        ai_score, ai_map = self.ai_detector.analyze(rgb_pil)

        # 合成・切り貼りリスクの統合
        splice_prob = float(np.clip(ela_score * 0.55 + noise_score * 0.45, 0.0, 100.0))

        # 総合フェイク確率の算出
        # AI生成と非AI加工のどちらか高い方を軸にしつつ、補助指標を加味
        ai_combined = float(np.clip(ai_score * 0.88 + fft_score * 0.12, 0.0, 100.0))
        base_fake = max(ai_combined, splice_prob * 0.92)
        # 両方が中程度以上に高い場合はリスクを上乗せ
        if ai_combined > 40 and splice_prob > 40:
            base_fake = min(100.0, base_fake + 10.0)

        # メタデータに明確な痕跡がある場合
        if meta["is_ai_tagged"]:
            base_fake = max(base_fake, 95.0)
            ai_score = max(ai_score, 95.0)
        elif meta["is_edited_tagged"]:
            base_fake = max(base_fake, 65.0)
            splice_prob = max(splice_prob, 65.0)

        overall_fake_prob = float(np.clip(base_fake, 0.0, 100.0))

        # 判定ラベル
        if overall_fake_prob >= 70.0:
            verdict = "フェイク画像の可能性が高い"
        elif overall_fake_prob >= 40.0:
            verdict = "加工・AI生成の疑いあり（要注意）"
        else:
            verdict = "本物（未加工）の可能性が高い"

        # 判定理由サマリー
        reasons = []
        if meta["is_ai_tagged"]:
            reasons.append("画像メタデータにAI生成ツールの痕跡が記録されています。")
        if meta["is_edited_tagged"]:
            reasons.append("画像メタデータに編集ソフトの痕跡が記録されています。")
        if ai_score >= 60.0:
            reasons.append(f"AI検出モデルがAI生成特有の特徴を強く検出しました ({ai_score:.1f}%)。")
        if ela_score >= 55.0:
            reasons.append(f"ELA解析で局所的な圧縮率の不整合を検出しました ({ela_score:.1f}%)。")
        if noise_score >= 55.0:
            reasons.append(f"ノイズ解析で周囲と異なるノイズ分布の領域を検出しました ({noise_score:.1f}%)。")
        if fft_score >= 50.0:
            reasons.append(f"周波数解析で人工的な周期パターンを検出しました ({fft_score:.1f}%)。")
        if not reasons:
            reasons.append("AI生成や局所的な切り貼りの明確な痕跡は検出されませんでした。")

        # 総合ヒートマップの構築
        # AIスコアが高ければAIマップ重視、そうでなければELA+ノイズ重視
        ai_w = 0.6 if ai_score >= splice_prob else 0.25
        ela_w = (1.0 - ai_w) * 0.55
        noise_w = (1.0 - ai_w) * 0.45
        combined_norm = np.clip(ai_map * ai_w + ela_map * ela_w + noise_map * noise_w, 0.0, 1.0)

        heatmaps = {
            "総合異常マップ": _to_color_heatmap(combined_norm),
            "AI生成注目領域": _to_color_heatmap(ai_map),
            "ELA (圧縮誤差)": _to_color_heatmap(ela_map),
            "ノイズ不均一性": _to_color_heatmap(noise_map),
        }

        self._report("解析完了")
        return AnalysisResult(
            overall_fake_prob=overall_fake_prob,
            ai_prob=ai_score,
            splice_prob=splice_prob,
            ela_score=ela_score,
            noise_score=noise_score,
            fft_score=fft_score,
            metadata=meta,
            verdict_label=verdict,
            summary_reasons=reasons,
            heatmaps=heatmaps,
        )
