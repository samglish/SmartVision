"""
SmartVision - Étape 3 : Test de segmentation sémantique (compréhension de scène)
==================================================================================

Ce script teste SegFormer (pré-entraîné sur ADE20K) pour voir s'il peut
distinguer : mur, sol, porte, escalier, passage, plafond, etc.

Installation requise :
    pip install transformers torch torchvision pillow numpy opencv-python

Usage :
    python test_segmentation.py chemin/vers/image.jpg
"""

import sys
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Classes ADE20K qui nous intéressent particulièrement pour SmartVision
# (index -> nom de la classe dans ADE20K)
# Référence officielle des indices ADE20K (150 classes) :
# https://github.com/CSAILVision/sceneparsing
CLASSES_UTILES = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",
    5: "ceiling",
    8: "windowpane",
    14: "door",
    38: "railing",
    53: "stairs",
    59: "stairway",
    95: "bannister",
}

# Couleurs pour la visualisation (une couleur par classe utile, gris pour le reste)
COULEURS = {
    0: (200, 200, 200),   # wall - gris clair
    1: (150, 100, 100),   # building - brun
    2: (255, 200, 0),     # sky - bleu clair (BGR)
    3: (0, 200, 0),       # floor - vert
    5: (200, 200, 0),     # ceiling - jaune
    8: (255, 0, 255),     # windowpane - magenta
    14: (0, 0, 255),      # door - rouge (BGR)
    38: (0, 255, 255),    # railing - cyan
    53: (255, 0, 0),      # stairs - bleu
    59: (255, 100, 0),    # stairway - bleu foncé
    95: (0, 165, 255),    # bannister - orange
}


def charger_modele():
    print("[1/3] Chargement de SegFormer (ADE20K)...")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    print("     ✅ Modèle chargé")
    return processor, model


def segmenter_image(image_path, processor, model):
    print(f"[2/3] Analyse de l'image : {image_path}")
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # (1, num_classes, H, W)

    # Upsample à la taille originale de l'image
    upsampled = cv2.resize(
        logits.argmax(dim=1)[0].numpy().astype(np.uint8),
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST,
    )
    return image, upsampled


def analyser_resultats(seg_map):
    """Affiche les classes détectées et leur proportion dans l'image."""
    print("[3/3] Classes détectées dans la scène :")
    classes_presentes, counts = np.unique(seg_map, return_counts=True)
    total_pixels = seg_map.size

    resultats = sorted(zip(classes_presentes, counts), key=lambda x: -x[1])

    for classe_id, count in resultats:
        pourcentage = 100 * count / total_pixels
        if pourcentage < 1:
            continue
        nom = CLASSES_UTILES.get(int(classe_id), f"classe_{classe_id}")
        marqueur = "🎯" if int(classe_id) in CLASSES_UTILES else "  "
        print(f"     {marqueur} {nom:12s} : {pourcentage:5.1f}% de l'image")


def creer_visualisation(image, seg_map, sortie="segmentation_resultat.png"):
    """Superpose les classes utiles en couleur sur l'image originale."""
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = image_cv.copy()

    for classe_id, couleur in COULEURS.items():
        masque = seg_map == classe_id
        overlay[masque] = couleur

    resultat = cv2.addWeighted(image_cv, 0.5, overlay, 0.5, 0)
    cv2.imwrite(sortie, resultat)
    print(f"     💾 Visualisation sauvegardée : {sortie}")


def main():
    if len(sys.argv) < 2:
        print("Usage : python test_segmentation.py chemin/vers/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    processor, model = charger_modele()
    image, seg_map = segmenter_image(image_path, processor, model)
    analyser_resultats(seg_map)
    creer_visualisation(image, seg_map)

    print("\n✅ Test terminé. Regarde segmentation_resultat.png pour voir")
    print("   les zones colorées : mur / sol / porte / escalier / etc.")


if __name__ == "__main__":
    main()
