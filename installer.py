#!/usr/bin/env python3
"""
Script d'installation automatique pour le projet Lunettes Aveugle
Gère les incompatibilités Python 3.13
"""

import subprocess
import sys
import os

print("=" * 60)
print("  INSTALLATION AUTOMATIQUE")
print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("=" * 60)

# Librairies de base (toujours compatibles)
base_packages = [
    "ultralytics",
    "opencv-python",
    "numpy",
]

# TTS — peut échouer sur Python 3.13
tts_packages = [
    "pyttsx3",
]

print("\n[1/3] Installation des librairies de base...")
for pkg in base_packages:
    print(f"     → {pkg}...", end=" ", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅")
    except:
        print("❌")

print("\n[2/3] Tentative d'installation de la synthèse vocale...")
for pkg in tts_packages:
    print(f"     → {pkg}...", end=" ", flush=True)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅")
    except:
        print("❌ (non critique — on utilisera le mode texte)")

print("\n[3/3] Vérification finale...")
try:
    from ultralytics import YOLO
    import cv2
    print("     ✅ ultralytics + opencv OK")
except Exception as e:
    print(f"     ❌ Problème: {e}")
    sys.exit(1)

try:
    import pyttsx3
    print("     ✅ pyttsx3 OK — la voix fonctionnera")
except:
    print("     ⚠️  pyttsx3 indisponible — mode texte activé")
    print("     → Les phrases s'afficheront dans le terminal")

print("\n" + "=" * 60)
print("  INSTALLATION TERMINÉE")
print("=" * 60)
print("\nProchaine étape:")
print("  python test_minimal.py")
