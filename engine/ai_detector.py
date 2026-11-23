import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class AIDetector:
    """
    複数の事前学習済みVision Transformer/Swinモデルをアンサンブルし、
    全体推論＋マルチパッチ推論によって高精度なAI生成確率と空間ヒートマップを算出する。
    """

    MODEL_IDS = [
        "umm-maybe/AI-image-detector",
        "Organika/sdxl-detector",
    ]

    def __init__(self, progress_callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.progress_callback = progress_callback
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return
        for idx, model_id in enumerate(self.MODEL_IDS):
            if self.progress_callback:
                self.progress_callback(
                    f"AI検出モデルを読み込み中 ({idx + 1}/{len(self.MODEL_IDS)}): {model_id}..."
                )
            try:
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device)
                model.eval()
                ai_label_idx = self._find_ai_label_index(model.config.id2label)
                self.models.append({
                    "id": model_id,
                    "processor": processor,
                    "model": model,
                    "ai_idx": ai_label_idx,
                })
            except Exception as e:
                print(f"[Warning] Failed to load {model_id}: {e}")

        if not self.models:
            raise RuntimeError("AI検出モデルの読み込みにすべて失敗しました。ネットワーク接続を確認してください。")
        self._loaded = True

    @staticmethod
    def _find_ai_label_index(id2label: dict) -> int:
        """モデルのラベル辞書から 'artificial' / 'ai' / 'fake' に該当するインデックスを特定する。"""
        for idx, label in id2label.items():
            low = str(label).lower()
            if any(k in low for k in ("artificial", "ai", "fake", "generated")):
                return int(idx)
        return 1 if len(id2label) > 1 else 0

    @torch.inference_mode()
    def _predict_batch(self, images: list[Image.Image]) -> list[float]:
        """複数画像（またはパッチ）に対して全モデルのアンサンブルAI確率(0..1)を返す。"""
        all_probs = np.zeros(len(images), dtype=np.float32)
        for m in self.models:
            inputs = m["processor"](images=images, return_tensors="pt").to(self.device)
            outputs = m["model"](**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, m["ai_idx"]].detach().cpu().numpy()
            all_probs += probs
        all_probs /= len(self.models)
        return all_probs.tolist()

    def analyze(self, image_pil: Image.Image) -> tuple[float, np.ndarray]:
        """
        Returns:
            ai_score (float): 0.0 ~ 100.0 のAI生成確率
            heatmap (np.ndarray): 0.0 ~ 1.0 の空間ヒートマップ (H, W)
        """
        self.load_models()
        rgb = image_pil.convert("RGB")
        w, h = rgb.size

        # 1. 画像全体のグローバル推論
        if self.progress_callback:
            self.progress_callback("AI生成判定: 画像全体を解析中...")
        global_prob = self._predict_batch([rgb])[0]

        # 2. マルチパッチ推論（3x3グリッド + オーバーラップ）による高精度化＆ヒートマップ生成
        if self.progress_callback:
            self.progress_callback("AI生成判定: 局所パッチを詳細解析中...")

        grid_n = 3
        patch_w = int(w * 0.45)
        patch_h = int(h * 0.45)
        patches = []
        coords = []

        if w >= 128 and h >= 128:
            xs = np.linspace(0, max(0, w - patch_w), grid_n, dtype=int)
            ys = np.linspace(0, max(0, h - patch_h), grid_n, dtype=int)
            for y in ys:
                for x in xs:
                    x2 = min(w, x + patch_w)
                    y2 = min(h, y + patch_h)
                    patches.append(rgb.crop((x, y, x2, y2)))
                    coords.append((x, y, x2, y2))

            patch_probs = self._predict_batch(patches)
        else:
            patch_probs = [global_prob]
            coords = [(0, 0, w, h)]

        # 空間ヒートマップの構築
        acc_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2), p in zip(coords, patch_probs):
            acc_map[y1:y2, x1:x2] += p
            weight_map[y1:y2, x1:x2] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        spatial_map = acc_map / weight_map
        # 滑らかなヒートマップにする
        ksize = max(15, (min(w, h) // 10) | 1)
        spatial_map = cv2.GaussianBlur(spatial_map, (ksize, ksize), 0)

        # グローバル推論(60%) + パッチ最大/上位平均(40%)で精度を高める
        top_patch_prob = float(np.percentile(patch_probs, 80))
        final_prob = global_prob * 0.6 + top_patch_prob * 0.4
        ai_score = float(np.clip(final_prob * 100.0, 0.0, 100.0))

        # ヒートマップ正規化（0..1）
        max_v = float(np.max(spatial_map))
        min_v = float(np.min(spatial_map))
        if max_v - min_v > 1e-5:
            heatmap = (spatial_map - min_v) / (max_v - min_v)
            # AI確率自体が低い場合はヒートマップの強度も抑える
            heatmap = heatmap * np.clip(final_prob * 1.3, 0.15, 1.0)
        else:
            heatmap = np.full((h, w), final_prob, dtype=np.float32)

        return ai_score, np.clip(heatmap, 0.0, 1.0).astype(np.float32)
