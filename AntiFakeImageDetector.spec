# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

def safe_copy_metadata(pkg_name: str):
    try:
        return copy_metadata(pkg_name)
    except Exception:
        return []

def safe_collect_data(pkg_name: str, **kwargs):
    try:
        return collect_data_files(pkg_name, **kwargs)
    except Exception:
        return []

datas = []
metadata_pkgs = [
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "numpy",
    "scipy",
    "pillow",
    "tqdm",
    "regex",
    "pyyaml",
    "packaging",
    "filelock",
    "hf-xet",
    "httpx2",
    "truststore",
    "requests",
    "urllib3",
]
for pkg in metadata_pkgs:
    datas += safe_copy_metadata(pkg)

datas += safe_collect_data("transformers", include_py_files=False)
datas += safe_collect_data("certifi")

hiddenimports = []
hiddenimports += collect_submodules("engine")
hiddenimports += collect_submodules("transformers.models.swin")
hiddenimports += collect_submodules("transformers.models.vit")
hiddenimports += collect_submodules("transformers.models.auto")
hiddenimports += [
    "transformers.image_processing_utils",
    "transformers.image_transforms",
    "transformers.image_utils",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.utils",
    "scipy.special._cdflib",
    "scipy.ndimage",
    "hf_xet",
    "truststore",
]

excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtSql",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AntiFakeImageDetector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AntiFakeImageDetector",
)
