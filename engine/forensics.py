import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ExifTags


def analyze_ela(image_pil: Image.Image, qualities=(90, 95)) -> tuple[float, np.ndarray]:
    """
    Error Level Analysis (ELA) を複数品質で実行し、
    合成・切り貼りによる局所的な圧縮誤差の不均一性を検出する。
    Returns:
        score (float): 0.0 ~ 100.0 の加工疑いスコア
        heatmap (np.ndarray): 0.0 ~ 1.0 に正規化された空間ヒートマップ (H, W)
    """
    rgb = image_pil.convert("RGB")
    w, h = rgb.size
    orig_arr = np.asarray(rgb, dtype=np.float32)

    diff_maps = []
    for q in qualities:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        recomp_arr = np.asarray(recompressed, dtype=np.float32)
        diff = np.abs(orig_arr - recomp_arr)
        # チャンネル平均
        diff_gray = np.mean(diff, axis=2)
        diff_maps.append(diff_gray)

    combined_diff = np.mean(diff_maps, axis=0)

    # 局所的な異常（周囲との差）を強調するため平滑化
    blurred = cv2.GaussianBlur(combined_diff, (15, 15), 0)
    # 全体のベースラインを引いて、突出した領域を抽出
    median_val = np.median(blurred)
    mad = np.median(np.abs(blurred - median_val)) + 1e-6
    z_map = np.clip((blurred - median_val) / (mad * 3.0), 0.0, 1.0)

    # スコア算出: 上位5%の異常度と全体のばらつきを組み合わせる
    p95 = float(np.percentile(z_map, 95))
    p99 = float(np.percentile(z_map, 99))
    ela_raw = (p95 * 0.4 + p99 * 0.6) * 100.0
    score = float(np.clip(ela_raw, 0.0, 100.0))

    # ヒートマップを0..1に正規化
    max_val = np.max(z_map)
    if max_val > 1e-6:
        heatmap = z_map / max_val
    else:
        heatmap = z_map

    return score, heatmap.astype(np.float32)


def analyze_noise_inconsistency(image_np: np.ndarray) -> tuple[float, np.ndarray]:
    """
    高周波ノイズ残差を抽出し、局所ブロックごとのノイズ分散の不均一性を検出する。
    切り貼りされた領域は周囲とノイズレベルが異なることが多い。
    Returns:
        score (float): 0.0 ~ 100.0
        heatmap (np.ndarray): 0.0 ~ 1.0 (H, W)
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # SRM風の高周波残差抽出（メディアンフィルタとの差分 + ラプラシアン）
    denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    residual = np.abs(gray - denoised)

    # 局所領域(32x32相当)のノイズ標準偏差を計算
    ksize = 31
    local_mean = cv2.boxFilter(residual, -1, (ksize, ksize))
    local_sq_mean = cv2.boxFilter(residual ** 2, -1, (ksize, ksize))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(local_var)

    # エッジの影響を抑えるため、画像の勾配が大きい部分は重みを下げる
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_weight = 1.0 / (1.0 + grad_mag / 50.0)

    weighted_std = local_std * edge_weight
    med = np.median(weighted_std)
    mad = np.median(np.abs(weighted_std - med)) + 1e-6

    # 周囲より極端にノイズが多い、または極端に少ない（塗りつぶし/平滑化）領域を検出
    anomaly = np.abs(weighted_std - med) / (mad * 4.0)
    anomaly = cv2.GaussianBlur(anomaly, (21, 21), 0)
    anomaly = np.clip(anomaly, 0.0, 1.0)

    p95 = float(np.percentile(anomaly, 95))
    p99 = float(np.percentile(anomaly, 99))
    score = float(np.clip((p95 * 0.5 + p99 * 0.5) * 100.0, 0.0, 100.0))

    max_val = np.max(anomaly)
    heatmap = (anomaly / max_val) if max_val > 1e-6 else anomaly
    return score, heatmap.astype(np.float32)


def analyze_fft_spectrum(image_np: np.ndarray) -> float:
    """
    2次元FFTによる周波数領域の異常（AI生成特有の周期的アーティファクトや高周波欠落）を評価。
    Returns:
        score (float): 0.0 ~ 100.0
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 計算安定化のため512x512にリサイズ
    resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # 中心（低周波）を除いた高周波領域の統計
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    high_freq_mask = (r > 60) & (r < 240)
    hf_vals = magnitude[high_freq_mask]
    # 局所的なスパイク（GAN/Diffusion特有のグリッドアーティファクト）を検出
    med = np.median(hf_vals)
    mad = np.median(np.abs(hf_vals - med)) + 1e-6
    spike_ratio = float(np.mean(hf_vals > med + 4.5 * mad))

    score = float(np.clip(spike_ratio * 2500.0, 0.0, 100.0))
    return score


def analyze_metadata(image_pil: Image.Image) -> dict:
    """
    EXIFやPNGテキストチャンクからAI生成ツールや編集ソフトの痕跡を検出する。
    """
    findings = []
    is_ai_tagged = False
    is_edited_tagged = False

    # PNG info chunks (Stable Diffusion, ComfyUI, NovelAI, Midjourney等)
    info = image_pil.info or {}
    ai_keywords = [
        "parameters", "prompt", "workflow", "sd-metadata",
        "Dream", "Midjourney", "Stable Diffusion", "ComfyUI", "DALL-E", "NovelAI"
    ]
    for k, v in info.items():
        k_str = str(k)
        v_str = str(v)[:200]
        for kw in ai_keywords:
            if kw.lower() in k_str.lower() or kw.lower() in v_str.lower():
                is_ai_tagged = True
                findings.append(f"AI生成メタデータ検出: {k_str}")
                break

    # EXIF
    exif_dict = {}
    try:
        raw_exif = image_pil.getexif()
        if raw_exif:
            for tag_id, val in raw_exif.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_dict[tag] = str(val)
    except Exception:
        pass

    edit_software = ["photoshop", "gimp", "lightroom", "affinity", "canva", "picsart", "snapseed"]
    sw = exif_dict.get("Software", "")
    if sw:
        for es in edit_software:
            if es in sw.lower():
                is_edited_tagged = True
                findings.append(f"編集ソフトの痕跡 (Software): {sw}")
                break

    camera_model = exif_dict.get("Model", "")
    if camera_model:
        findings.append(f"撮影機器情報 (Model): {camera_model}")
    elif not is_ai_tagged:
        findings.append("カメラ撮影情報 (EXIF Model) なし")

    return {
        "is_ai_tagged": is_ai_tagged,
        "is_edited_tagged": is_edited_tagged,
        "has_camera_exif": bool(camera_model),
        "findings": findings,
    }
