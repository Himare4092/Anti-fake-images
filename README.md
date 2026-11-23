# Anti-Fake Image Detector

AIで生成されたフェイク画像、および合成・切り貼りなどの非AI加工画像を検出するための Windows 向けデスクトップアプリケーションです。  
処理速度よりも**判定精度と根拠の可視化**を重視して設計されています。

## 主な機能

- **総合フェイク確率（%）の表示**: 複数の解析手法を統合し、0〜100%でリスクを判定
- **怪しい箇所のヒートマップ表示**: 元画像との重ね合わせ表示・透明度調整・マップ種別の切り替えに対応
  - 総合異常マップ
  - AI生成注目領域（マルチパッチ推論による空間ヒートマップ）
  - ELA（Error Level Analysis：JPEG再圧縮誤差）
  - ノイズ不均一性（局所的なノイズ分散の異常領域）
- **AI生成画像検出（高精度アンサンブル）**:
  - 2種類の事前学習済みモデル（`umm-maybe/AI-image-detector` / `Organika/sdxl-detector`）のアンサンブル
  - 画像全体＋マルチパッチ（3×3オーバーラップ領域）の推論による高精度判定
- **非AI加工・フォレンジック解析**:
  - ELA（複数品質でのJPEG再圧縮誤差解析）
  - 高周波ノイズ残差による局所ノイズ分散の不均一性検出
  - 2次元FFTによる周波数スペクトル異常解析
  - EXIF / PNGテキストチャンクからのAI生成ツール・編集ソフト痕跡検査

## 動作環境

- OS: Windows 10 / 11 (64-bit)
- Python: 3.11〜3.13
- GPU: NVIDIA GPU（CUDA対応）推奨 ※CPU環境でも動作します

## セットアップと起動方法

```powershell
# 1. リポジトリをクローン
git clone https://github.com/Himare4092/Anti-fake-images.git
cd Anti-fake-images

# 2. 仮想環境の作成
python -m venv .venv

# 3. PyTorch のインストール（NVIDIA GPU がある場合 / CUDA 12.4）
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. その他の依存ライブラリをインストール
.\.venv\Scripts\pip install -r requirements.txt

# 5. アプリを起動
.\.venv\Scripts\python.exe app.py
```

`run.bat` をダブルクリックして起動することもできます。

## Windows 向け `.exe` ファイルのビルド方法

PyInstaller を用いてフォルダ形式（`--onedir`）の実行ファイルをビルドできます。

```powershell
.\.venv\Scripts\pip install pyinstaller
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean AntiFakeImageDetector.spec
```

ビルド完了後、`dist\AntiFakeImageDetector\AntiFakeImageDetector.exe` が生成されます。  
別の場所へ移動する場合は、同じ階層にある `_internal` フォルダとセットで **`AntiFakeImageDetector` フォルダごと** 移動してください。

## 注意事項

- 本ツールの判定結果は統計的・機械学習的な推定であり、**100%の精度を保証するものではありません**。特にSNSでの強い再圧縮を経た画像や高度なレタッチ画像では判定精度が低下する場合があります。
- 初回解析時のみ、Hugging Face からAI検出モデル（合計数百MB〜1GB程度）が自動ダウンロードされるため、インターネット接続が必要です。2回目以降はローカルキャッシュから読み込まれます。
