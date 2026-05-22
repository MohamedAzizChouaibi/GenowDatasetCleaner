# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for GenowDatasetCleaner
# Build with:  pyinstaller genowCleaner.spec

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect all torch files (DLLs, data, submodules)
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
torchvision_datas, torchvision_binaries, torchvision_hiddenimports = collect_all('torchvision')
clip_datas, clip_binaries, clip_hiddenimports = collect_all('clip')

a = Analysis(
    ['genowCleaner.py'],
    pathex=[],
    binaries=torch_binaries + torchvision_binaries + clip_binaries,
    datas=[
        # Bundle the logo so _resource_dir() finds it under sys._MEIPASS
        ('genow_logo.jpeg', '.'),
        ('genow_logo.ico', '.'),
        ('embedders.py', '.'),
        ('api_embedders.py', '.'),
    ] + torch_datas + torchvision_datas + clip_datas,
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'PIL._tkinter_finder',
        'cv2',
        'numpy',
        'imagehash',
        'send2trash',
        'scipy',
        'scipy.fft',
        'PyWavelets',
        'ftfy',
        'regex',
        'tqdm',
        'clip',
        'embedders',
        'api_embedders',
    ] + torch_hiddenimports + torchvision_hiddenimports + clip_hiddenimports
      + collect_submodules('torch')
      + collect_submodules('torchvision'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'sklearn',
        'tensorflow',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GenowDatasetCleaner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no black console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='genow_logo.ico',   # exe icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GenowDatasetCleaner',
)
