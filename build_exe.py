# -*- coding: utf-8 -*-
"""แพ็ก TaThip เป็น .exe ด้วย PyInstaller
ใช้:  pip install pyinstaller   แล้ว   python build_exe.py
ผล:   dist/TaThip.exe
"""
import PyInstaller.__main__

SEP = ";"  # Windows ใช้ ; (mac/linux ใช้ :)
PyInstaller.__main__.run([
    "app.py",
    "--name", "TaThip",
    "--windowed",
    "--noconfirm",
    "--clean",
    f"--add-data=index.html{SEP}.",
    f"--add-data=tracks_data.js{SEP}.",
    f"--add-data=route.js{SEP}.",
    f"--add-data=logo_icon.png{SEP}.",
    f"--add-data=logo_black.png{SEP}.",
    f"--add-data=img{SEP}img",
    # ML (ultralytics/torch/easyocr) โหลดแบบ lazy + เป็น dependency ภายนอก (BDI/TATHIP_BDI)
    # ไม่รวมในตัว exe เพื่อให้ไฟล์เล็ก — "ประมวลผลกล้อง"/OCR ต้องรันจาก source (python app.py)
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "ultralytics",
    "--exclude-module", "easyocr",
])
