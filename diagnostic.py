#!/usr/bin/env python3
"""
Diagnostic d'environnement pour le projet Lunettes Aveugle
Vérifie Python, les librairies, la webcam et la synthèse vocale
"""

import sys
import subprocess

print("=" * 60)
print("  DIAGNOSTIC ENVIRONNEMENT")
print("=" * 60)
print(f"\nPython version: {sys.version}")
print(f"Plateforme: {sys.platform}")

# 1. Vérifier les librairies critiques
print("\n--- Vérification des librairies ---")
libs = {
    'ultralytics': 'YOLOv8 (détection)',
    'cv2': 'OpenCV (caméra)',
    'numpy': 'NumPy (calculs)',
    'pyttsx3': 'Synthèse vocale (TTS)',
}

manquantes = []
for lib, desc in libs.items():
    try:
        mod = __import__(lib)
        version = getattr(mod, '__version__', 'inconnue')
        print(f"  ✅ {lib:12s} v{version:10s} — {desc}")
    except ImportError:
        print(f"  ❌ {lib:12s} {'MANQUANT':10s} — {desc}")
        manquantes.append(lib)

# 2. Vérifier la webcam
print("\n--- Vérification webcam ---")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  ✅ Webcam détectée: {w}x{h}")
        cap.release()
    else:
        print(f"  ❌ Aucune webcam détectée sur l'index 0")
except Exception as e:
    print(f"  ❌ Erreur webcam: {e}")

# 3. Vérifier la synthèse vocale
print("\n--- Vérification synthèse vocale ---")
try:
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print(f"  ✅ Moteur TTS OK — {len(voices)} voix disponibles")
    for i, v in enumerate(voices[:3]):
        print(f"      Voix {i}: {v.name}")
except Exception as e:
    print(f"  ❌ TTS non disponible: {e}")
    print(f"      → Solution: on utilisera des fichiers audio pré-enregistrés")

# 4. Vérifier le GPU (optionnel)
print("\n--- Vérification accélération ---")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ GPU CUDA détecté: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"  ✅ GPU Apple MPS détecté")
    else:
        print(f"  ⚠️  Pas de GPU — utilisation CPU (plus lent mais fonctionnel)")
except:
    print(f"  ⚠️  PyTorch non installé — utilisation CPU")

# 5. Résumé
print("\n" + "=" * 60)
if manquantes:
    print("  ACTION REQUISE: Installer les librairies manquantes")
    print(f"  Commande: pip install {' '.join(manquantes)}")
else:
    print("  ✅ TOUT EST PRÊT — Tu peux lancer le prototype!")
print("=" * 60)
