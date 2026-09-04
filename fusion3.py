"""
SmartVision — Pipeline complet
================================
YOLO (objets/personnes/véhicules, à chaque frame)
+ SegFormer (sol/mur/rambarde/main courante, toutes les N frames)
+ fusion par priorité commune
+ synthèse vocale (espeak-ng)

Installation :
    pip install ultralytics opencv-python transformers torch torchvision pillow numpy
"""

from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import subprocess
import shutil
import threading
import queue

print("=" * 60)
print("SMARTVISION — PIPELINE COMPLET (objets + surfaces + voix)")
print("=" * 60)

# ------------------------------------------------------------------
# VOIX — binaire espeak-ng/espeak en ligne de commande, thread séparé
# ------------------------------------------------------------------
print("\n[1/4] Initialisation de la voix...")
BINAIRE_VOIX = shutil.which("espeak-ng") or shutil.which("espeak")
if BINAIRE_VOIX is None:
    print("❌ Aucun moteur vocal trouvé. Installe : sudo apt install espeak-ng")
    exit()

file_attente_vocale = queue.Queue()


def travailleur_vocal():
    while True:
        message = file_attente_vocale.get()
        if message is None:
            break
        subprocess.run(
            [BINAIRE_VOIX, "-v", "fr", "-s", "165", message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        file_attente_vocale.task_done()


thread_voix = threading.Thread(target=travailleur_vocal, daemon=True)
thread_voix.start()


def parler(message, urgent=False):
    if urgent:
        with file_attente_vocale.mutex:
            file_attente_vocale.queue.clear()
    file_attente_vocale.put(message)


print(f"✅ Voix prête ({BINAIRE_VOIX})")

# ------------------------------------------------------------------
# YOLO — objets utiles à la navigation (liste ciblée)
# ------------------------------------------------------------------
print("\n[2/4] Chargement de YOLOv8s-oiv7...")
modele_yolo = YOLO("yolov8s-oiv7.pt")
print("✅ YOLO chargé")

priorite_objets = {
    "Person": ("Personne", 8),
    "Dog": ("Chien", 6),
    "Car": ("Voiture", 8),
    "Truck": ("Camion", 8),
    "Bus": ("Bus", 8),
    "Motorcycle": ("Moto", 8),
    "Bicycle": ("Vélo", 6),
    "Train": ("Train", 8),
    "Taxi": ("Taxi", 8),
    "Van": ("Camionnette", 8),
    "Door": ("Porte", 6),
    "Window": ("Fenêtre", 4),
    "Stairs": ("Escalier", 9),
    "Traffic light": ("Feu de circulation", 7),
    "Traffic sign": ("Panneau de signalisation", 5),
    "Stop sign": ("Panneau stop", 7),
    "Fire hydrant": ("Bouche d'incendie", 3),
    "Street light": ("Lampadaire", 2),
    "Bench": ("Banc", 3),
    "Chair": ("Chaise", 3),
    "Table": ("Table", 3),
    "Waste container": ("Poubelle", 3),
    "Suitcase": ("Valise", 4),
    "Luggage and bags": ("Bagages", 3),
    "Ladder": ("Échelle", 5),
    "Mobile phone": ("Téléphone", 2),
    "Bed": ("Lit", 2),
    "Refrigerator": ("Réfrigérateur", 2),
    "Gas stove": ("Cuisinière à gaz", 5),
    "Laptop": ("Ordinateur", 2),
}

SEUIL_YOLO = 0.30

# ------------------------------------------------------------------
# SEGFORMER — uniquement les surfaces que YOLO ne sait pas détecter.
# Porte/fenêtre/escalier sont volontairement exclus ici : YOLO les
# gère déjà avec une bounding box (position/distance plus fiable
# qu'une simple proportion de pixels).
# ------------------------------------------------------------------
print("\n[3/4] Chargement de SegFormer (ADE20K)...")
processor_seg = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
modele_seg = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
modele_seg.eval()
print("✅ SegFormer chargé")

priorite_surfaces = {
    3: ("Sol", 0),         # floor
    0: ("Mur", 2),         # wall
    38: ("Rambarde", 6),   # railing
    95: ("Main courante", 6),  # bannister
}

SEUIL_SURFACE_PROPORTION = 0.05  # en dessous de 5% de l'image, on ignore (bruit)
INTERVALLE_SEGMENTATION = 15     # ne relance SegFormer que toutes les 15 frames

# ------------------------------------------------------------------
# Fonctions communes position / distance
# ------------------------------------------------------------------


def position_brute(centre_x, largeur_image):
    tiers = largeur_image / 3
    if centre_x < tiers:
        return "gauche"
    elif centre_x < 2 * tiers:
        return "centre"
    else:
        return "droite"


def distance_depuis_proportion(proportion):
    if proportion > 0.35:
        return "très proche"
    elif proportion > 0.15:
        return "proche"
    elif proportion > 0.05:
        return "à moyenne distance"
    else:
        return "loin"


def position_objet_texte(position):
    return {"gauche": "à gauche", "centre": "devant vous", "droite": "à droite"}[position]


def position_surface_texte(position):
    return f"sur votre {position}"


# ------------------------------------------------------------------
# Détection YOLO (chaque frame)
# ------------------------------------------------------------------


def detecter_objets(frame, largeur, hauteur):
    resultats = modele_yolo(frame, verbose=False)
    detections = []

    for result in resultats:
        for box in result.boxes:
            confiance = float(box.conf[0])
            if confiance < SEUIL_YOLO:
                continue
            cls = int(box.cls[0])
            nom_brut = modele_yolo.names[cls]
            if nom_brut not in priorite_objets:
                continue

            nom_fr, base = priorite_objets[nom_brut]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centre_x = (x1 + x2) / 2
            proportion = ((x2 - x1) * (y2 - y1)) / (largeur * hauteur)

            position = position_brute(centre_x, largeur)
            distance = distance_depuis_proportion(proportion)
            priorite = base + {"très proche": 4, "proche": 2, "à moyenne distance": 0, "loin": -2}[distance]

            detections.append({
                "type": "objet", "nom": nom_fr, "distance": distance,
                "position": position, "priorite": priorite,
                "boite": (x1, y1, x2, y2),
            })

    return detections


# ------------------------------------------------------------------
# Détection SegFormer (toutes les N frames)
# ------------------------------------------------------------------


def detecter_surfaces(frame, largeur, hauteur):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor_seg(images=image_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = modele_seg(**inputs)

    seg_map = cv2.resize(
        outputs.logits.argmax(dim=1)[0].numpy().astype(np.uint8),
        (largeur, hauteur),
        interpolation=cv2.INTER_NEAREST,
    )

    detections = []
    for classe_id, (nom_fr, base) in priorite_surfaces.items():
        masque = seg_map == classe_id
        proportion = masque.sum() / masque.size
        if proportion < SEUIL_SURFACE_PROPORTION:
            continue

        ys, xs = np.where(masque)
        centre_x = xs.mean()
        position = position_brute(centre_x, largeur)
        distance = distance_depuis_proportion(proportion)
        priorite = base + {"très proche": 4, "proche": 2, "à moyenne distance": 0, "loin": -2}[distance]

        detections.append({
            "type": "surface", "nom": nom_fr, "distance": distance,
            "position": position, "priorite": priorite, "masque": masque,
        })

    return detections


# ------------------------------------------------------------------
# Boucle principale
# ------------------------------------------------------------------
print("\n[4/4] Ouverture de la webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossible d'ouvrir la webcam")
    exit()

print("✅ Webcam ouverte")
print("\nCommandes : 'q' quitter | 'i' faire le point")
print("=" * 60)

SEUIL_URGENT = 7
derniers_objets = set()
dernieres_surfaces = []
compteur_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Impossible de lire la webcam")
        break

    hauteur, largeur = frame.shape[:2]
    compteur_frame += 1

    # YOLO à chaque frame
    detections_objets = detecter_objets(frame, largeur, hauteur)

    # SegFormer seulement toutes les N frames (coûteux)
    if compteur_frame % INTERVALLE_SEGMENTATION == 0:
        dernieres_surfaces = detecter_surfaces(frame, largeur, hauteur)

    toutes_detections = detections_objets + dernieres_surfaces
    toutes_detections.sort(key=lambda d: d["priorite"], reverse=True)

    # Affichage
    for d in toutes_detections:
        couleur = (0, 0, 255) if d["priorite"] >= 6 else (0, 165, 255) if d["priorite"] >= 3 else (0, 255, 0)
        if d["type"] == "objet":
            x1, y1, x2, y2 = d["boite"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(frame, f"{d['nom']} | {d['distance']}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2)
        else:
            overlay = frame.copy()
            overlay[d["masque"]] = couleur
            frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Annonce des nouveautés (voix + terminal)
    objets_actuels = {(d["nom"], d["position"]) for d in toutes_detections}
    nouveaux = objets_actuels - derniers_objets

    if nouveaux:
        print("\n📍 DÉTECTIONS FUSIONNÉES (triées par priorité)")
        for d in toutes_detections:
            marqueur = "[objet]" if d["type"] == "objet" else "[surface]"
            print(f"  {marqueur} priorité={d['priorite']:2d} → {d['nom']}, {d['distance']}, {d['position']}")
            # On affiche TOUT (même loin), mais on ne parle que si c'est
            # assez proche pour être pertinent (~3m ou moins, approximatif)
            if (d["nom"], d["position"]) in nouveaux and d["distance"] != "loin":
                if d["type"] == "objet":
                    phrase = f"{d['nom']}, {position_objet_texte(d['position'])}, {d['distance']}"
                else:
                    phrase = f"{d['nom']} {position_surface_texte(d['position'])}, {d['distance']}"
                parler(phrase, urgent=(d["priorite"] >= SEUIL_URGENT))

    derniers_objets = objets_actuels

    cv2.imshow("SmartVision - Pipeline complet", frame)

    touche = cv2.waitKey(1) & 0xFF
    if touche == ord("q"):
        break
    elif touche == ord("i"):
        if not toutes_detections:
            message = "Rien de pertinent détecté autour de vous."
        else:
            elements = []
            for d in toutes_detections:
                if d["type"] == "objet":
                    elements.append(f"{d['nom']} {position_objet_texte(d['position'])}, {d['distance']}")
                else:
                    elements.append(f"{d['nom']} {position_surface_texte(d['position'])}, {d['distance']}")
            message = "Autour de vous : " + " ; ".join(elements) + "."
        print(f"\n🔊 {message}")
        parler(message, urgent=True)


cap.release()
cv2.destroyAllWindows()
file_attente_vocale.put(None)
thread_voix.join(timeout=2)

print("\n" + "=" * 60)
print("TEST TERMINÉ")
print("=" * 60)
