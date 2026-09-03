#!/usr/bin/env python3
"""
SmartVision - Étape 4 : Fusion YOLO (objets) + SegFormer (surfaces)
======================================================================

Combine la détection d'objets (YOLO) et la compréhension de scène
(SegFormer/ADE20K) dans une seule analyse : position + priorité + phrase.

Usage :
    python test_fusion.py chemin/vers/image.jpg

Installation requise :
    pip install ultralytics transformers torch torchvision opencv-python pillow numpy
"""

import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Objets YOLO utiles (nom anglais -> nom français)
OBJETS_UTILES = {
    "person": "personne",
    "car": "voiture",
    "motorcycle": "moto",
    "bus": "bus",
    "truck": "camion",
    "bicycle": "vélo",
    "chair": "chaise",
    "bench": "banc",
    "dog": "chien",
    "cat": "chat",
    "backpack": "sac à dos",
    "suitcase": "valise",
}

# Surfaces SegFormer utiles (indice ADE20K -> nom français)
# Référence des indices : https://github.com/CSAILVision/sceneparsing
SURFACES_UTILES = {
    0: "mur",
    3: "sol",
    8: "fenêtre",
    14: "porte",
    38: "rambarde",
    53: "escalier",
    59: "escalier",
    95: "main courante",
}

# Surfaces qu'on considère comme des obstacles/dangers potentiels
SURFACES_DANGER = {14, 38, 53, 59, 95}  # porte, rambarde, escalier, main courante

# Priorité de base par type (plus haut = plus important à annoncer)
PRIORITE_BASE = {
    "personne": 10,
    "voiture": 10, "moto": 10, "bus": 10, "camion": 10, "vélo": 9,
    "escalier": 8, "main courante": 6,
    "chaise": 5, "banc": 5, "chien": 6, "chat": 4,
    "porte": 4, "rambarde": 3,
    "mur": 2, "fenêtre": 1, "sol": 0,
}


# ------------------------------------------------------------------
# Chargement des modèles
# ------------------------------------------------------------------

def charger_modeles():
    print("[1/4] Chargement de YOLOv8n...")
    yolo = YOLO("yolov8n.pt")
    print("     ✅ YOLO chargé")

    print("[2/4] Chargement de SegFormer (ADE20K)...")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    print("     ✅ SegFormer chargé")

    return yolo, processor, seg_model


# ------------------------------------------------------------------
# Utilitaires : position / distance relative
# ------------------------------------------------------------------

def position_texte(x_center, largeur):
    ratio = x_center / largeur
    if ratio < 0.33:
        return "gauche"
    elif ratio > 0.66:
        return "droite"
    return "centre"


def distance_texte(surface_ratio):
    """surface_ratio = proportion de l'image occupée par l'élément."""
    if surface_ratio > 0.25:
        return "très proche"
    elif surface_ratio > 0.08:
        return "proche"
    elif surface_ratio > 0.015:
        return "à moyenne distance"
    return "loin"


# ------------------------------------------------------------------
# Détection YOLO -> liste de détections unifiées
# ------------------------------------------------------------------

def detecter_objets(yolo, frame):
    h, w = frame.shape[:2]
    results = yolo(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            nom_en = yolo.names[cls_id]

            if nom_en not in OBJETS_UTILES or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            xc = (x1 + x2) / 2
            bw, bh = x2 - x1, y2 - y1
            surface_ratio = (bw * bh) / (w * h)

            nom_fr = OBJETS_UTILES[nom_en]
            pos = position_texte(xc, w)
            dist = distance_texte(surface_ratio)
            priorite = PRIORITE_BASE.get(nom_fr, 3)

            # Bonus de priorité si proche du centre et très proche
            if pos == "centre" and dist in ("très proche", "proche"):
                priorite += 5

            detections.append({
                "type": "objet",
                "nom": nom_fr,
                "position": pos,
                "distance": dist,
                "priorite": priorite,
                "box": (x1, y1, x2, y2),
                "confiance": conf,
            })

    return detections


# ------------------------------------------------------------------
# Segmentation SegFormer -> liste de détections unifiées
# ------------------------------------------------------------------

def detecter_surfaces(processor, seg_model, frame):
    h, w = frame.shape[:2]
    image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = seg_model(**inputs)
    logits = outputs.logits

    seg_map = cv2.resize(
        logits.argmax(dim=1)[0].numpy().astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    detections = []
    for classe_id, nom_fr in SURFACES_UTILES.items():
        masque = seg_map == classe_id
        count = int(masque.sum())
        if count == 0:
            continue

        surface_ratio = count / (w * h)
        if surface_ratio < 0.015:  # ignorer le bruit résiduel
            continue

        # Centroïde de la zone pour déterminer la position
        ys, xs = np.where(masque)
        xc = float(xs.mean())

        pos = position_texte(xc, w)
        priorite = PRIORITE_BASE.get(nom_fr, 1)

        # Les surfaces "danger" (porte/escalier/rambarde) proches du centre
        # et occupant une grande zone gagnent en priorité
        if classe_id in SURFACES_DANGER and surface_ratio > 0.1:
            priorite += 3

        detections.append({
            "type": "surface",
            "nom": nom_fr,
            "position": pos,
            "distance": distance_texte(surface_ratio),
            "priorite": priorite,
            "surface_ratio": surface_ratio,
            "masque": masque,
        })

    return detections


# ------------------------------------------------------------------
# Fusion + génération de la phrase
# ------------------------------------------------------------------

def generer_phrase(det):
    nom = det["nom"]
    pos = det["position"]
    dist = det["distance"]

    urgent = det["type"] == "objet" and det["priorite"] >= 10 and dist in ("très proche", "proche") and pos == "centre"
    if urgent:
        return f"🚨 DANGER — {nom.capitalize()} {dist} devant vous !"

    if det["type"] == "surface" and nom in ("escalier", "rambarde", "main courante"):
        return f"⚠️ {nom.capitalize()} sur votre {pos}, {dist}"

    return f"{nom.capitalize()} {dist} sur votre {pos}"


def fusionner_et_prioriser(objets, surfaces):
    toutes = objets + surfaces
    toutes.sort(key=lambda d: d["priorite"], reverse=True)
    return toutes


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

COULEURS_SURFACES = {
    "mur": (200, 200, 200),
    "sol": (0, 200, 0),
    "fenêtre": (255, 0, 255),
    "porte": (0, 0, 255),
    "rambarde": (0, 255, 255),
    "escalier": (255, 0, 0),
    "main courante": (0, 165, 255),
}


def creer_visualisation(frame, surfaces, objets, sortie="fusion_resultat.png"):
    overlay = frame.copy()

    for det in surfaces:
        couleur = COULEURS_SURFACES.get(det["nom"], (128, 128, 128))
        overlay[det["masque"]] = couleur

    resultat = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    for det in objets:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(resultat, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['nom']} | {det['position']} | {det['distance']}"
        cv2.putText(resultat, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(sortie, resultat)
    print(f"     💾 Visualisation sauvegardée : {sortie}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage : python test_fusion.py chemin/vers/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Impossible de charger l'image : {image_path}")
        sys.exit(1)

    yolo, processor, seg_model = charger_modeles()

    print(f"[3/4] Analyse de l'image : {image_path}")
    objets = detecter_objets(yolo, frame)
    surfaces = detecter_surfaces(processor, seg_model, frame)

    print("[4/4] Fusion et priorisation")
    toutes = fusionner_et_prioriser(objets, surfaces)

    print("\n" + "=" * 60)
    print("  DÉTECTIONS FUSIONNÉES (triées par priorité)")
    print("=" * 60)
    for det in toutes:
        phrase = generer_phrase(det)
        print(f"  [{det['type']:7s}] priorité={det['priorite']:2d} → {phrase}")

    if toutes:
        print("\n🔊 Message qui serait prononcé :")
        print(f"   {generer_phrase(toutes[0])}")

    creer_visualisation(frame, surfaces, objets)
    print("\n✅ Test terminé.")


if __name__ == "__main__":
    main()
