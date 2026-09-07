"""
SmartVision — SERVEUR (FastAPI)
==================================
Tourne sur le laptop/PC. Reçoit une image envoyée par le téléphone
(via HTTP), fait tourner YOLO + SegFormer + fusion par priorité,
et renvoie les détections en JSON (le téléphone se charge de parler
avec sa propre synthèse vocale).

Installation :
    pip install fastapi uvicorn python-multipart ultralytics opencv-python transformers torch torchvision pillow numpy

Lancement :
    uvicorn serveur_smartvision:app --host 0.0.0.0 --port 5000

Le serveur écoute sur le port 5000. Depuis le téléphone (sur le
MÊME réseau Wi-Fi que ce laptop), on appellera :
    http://<IP_DU_LAPTOP>:5000/analyser

Doc interactive (pratique pour tester sans le téléphone) :
    http://<IP_DU_LAPTOP>:5000/docs

Pour trouver l'IP du laptop sur le réseau local :
    Linux/Mac : hostname -I   (ou : ip addr)
    Windows   : ipconfig
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import os
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

print("=" * 60)
print("SMARTVISION — SERVEUR (FastAPI)")
print("=" * 60)

# ------------------------------------------------------------------
# Chargement des modèles (une seule fois, au démarrage du serveur)
# ------------------------------------------------------------------
print("\n[1/2] Chargement de YOLOv8m-oiv7...")

CHEMIN_PT = "yolov8m-oiv7.pt"
CHEMIN_OPENVINO = "yolov8m-oiv7_openvino_model"
FICHIER_XML = os.path.join(CHEMIN_OPENVINO, "yolov8m-oiv7.xml")

if os.path.isfile(FICHIER_XML):
    modele_yolo = YOLO(CHEMIN_OPENVINO)
    print("✅ YOLO chargé (OpenVINO)")
else:
    modele_yolo = YOLO(CHEMIN_PT)
    print("✅ YOLO chargé (PyTorch standard)")

print("\n[2/2] Chargement de SegFormer (ADE20K)...")
processor_seg = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
modele_seg = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
modele_seg.eval()
print("✅ SegFormer chargé")

# ------------------------------------------------------------------
# Config identique au pipeline qu'on avait déjà
# ------------------------------------------------------------------
priorite_objets = {
    "Person": ("Personne", 8), "Man": ("Personne", 8), "Woman": ("Personne", 8),
    "Boy": ("Personne", 8), "Girl": ("Personne", 8),
    "Dog": ("Chien", 6),
    "Car": ("Voiture", 8), "Truck": ("Camion", 8), "Bus": ("Bus", 8),
    "Motorcycle": ("Moto", 8), "Bicycle": ("Vélo", 6), "Train": ("Train", 8),
    "Taxi": ("Taxi", 8), "Van": ("Camionnette", 8),
    "Door": ("Porte", 6), "Window": ("Fenêtre", 4), "Stairs": ("Escalier", 9),
    "Traffic light": ("Feu de circulation", 7), "Traffic sign": ("Panneau de signalisation", 5),
    "Stop sign": ("Panneau stop", 7), "Fire hydrant": ("Bouche d'incendie", 3),
    "Street light": ("Lampadaire", 2), "Bench": ("Banc", 3), "Chair": ("Chaise", 3),
    "Table": ("Table", 3), "Waste container": ("Poubelle", 3), "Suitcase": ("Valise", 4),
    "Luggage and bags": ("Bagages", 3), "Ladder": ("Échelle", 5),
    "Mobile phone": ("Téléphone", 2), "Bed": ("Lit", 2), "Refrigerator": ("Réfrigérateur", 2),
    "Gas stove": ("Cuisinière à gaz", 5), "Laptop": ("Ordinateur", 2),
    "Land vehicle": ("Véhicule", 6), "Vehicle": ("Véhicule", 6),
}

priorite_surfaces = {
    3: ("Sol", 0), 0: ("Mur", 2), 38: ("Rambarde", 6), 95: ("Main courante", 6),
    6: ("Route", 5), 11: ("Trottoir", 1),
    21: ("Eau", 6), 26: ("Eau", 6), 60: ("Eau", 6), 109: ("Eau", 6), 128: ("Eau", 6),
}

SEUIL_YOLO = 0.20
SEUIL_SURFACE_PROPORTION = 0.05
BONUS_DISTANCE = {"très proche": 4, "proche": 2, "à moyenne distance": 0, "loin": -2}

# Un objet pile sur le chemin de marche (centre) représente un risque de
# collision direct — il doit primer sur un objet "plus important" en
# général mais situé sur le côté (ex: un PC droit devant vs une personne
# sur le côté).
BONUS_POSITION_CENTRE = 5

# Un objet qui vient d'apparaître (absent à l'image précédente) est un
# danger nouveau et imprévu (ex: quelqu'un qui vient de se mettre sur le
# chemin) — il doit être signalé avant un objet déjà connu et immobile.
BONUS_NOUVEAUTE = 3

# Mémoire simple d'une image à l'autre, pour détecter les nouveaux objets.
# Un seul utilisateur à la fois (téléphone <-> serveur), donc un état
# global suffit ici — pas besoin de le lier à une session.
dernieres_detections_vues = set()

# "Debounce" du conseil de direction : on n'annonce un changement de
# direction que s'il est confirmé sur 2 captures consécutives, pour
# éviter les allers-retours "gauche/droite" erratiques dus au léger
# tremblement de la détection d'une image à l'autre.
dernier_conseil_confirme = None
conseil_en_attente = None
compteur_confirmation = 0


def position_brute(centre_x, largeur_image):
    # Zone centrale volontairement élargie (35%-65% au lieu d'un tiers
    # exact) pour éviter qu'un léger tremblement de détection fasse
    # basculer la position d'une capture à l'autre (gauche <-> centre).
    if centre_x < 0.35 * largeur_image:
        return "gauche"
    elif centre_x < 0.65 * largeur_image:
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


def analyser_image(frame):
    """Fait tourner YOLO + SegFormer sur une image, renvoie la liste
    fusionnée triée par priorité (sans dessiner, juste les données)."""
    hauteur, largeur = frame.shape[:2]
    detections = []

    # --- YOLO ---
    resultats = modele_yolo(frame, verbose=False)
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
            priorite = base + BONUS_DISTANCE[distance]
            if position == "centre":
                priorite += BONUS_POSITION_CENTRE

            detections.append({
                "type": "objet", "nom": nom_fr, "distance": distance,
                "position": position, "priorite": priorite,
            })

    # --- SegFormer ---
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor_seg(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = modele_seg(**inputs)
    seg_map = cv2.resize(
        outputs.logits.argmax(dim=1)[0].numpy().astype(np.uint8),
        (largeur, hauteur), interpolation=cv2.INTER_NEAREST,
    )

    for classe_id, (nom_fr, base) in priorite_surfaces.items():
        masque = seg_map == classe_id
        proportion = masque.sum() / masque.size
        if proportion < SEUIL_SURFACE_PROPORTION:
            continue
        ys, xs = np.where(masque)
        centre_x = xs.mean()
        position = position_brute(centre_x, largeur)
        distance = distance_depuis_proportion(proportion)
        priorite = base + BONUS_DISTANCE[distance]
        if position == "centre":
            priorite += BONUS_POSITION_CENTRE

        detections.append({
            "type": "surface", "nom": nom_fr, "distance": distance,
            "position": position, "priorite": priorite,
        })

    # Bonus de nouveauté : un objet absent à l'image précédente est un
    # danger nouveau et imprévu, donc plus urgent qu'un objet déjà connu.
    global dernieres_detections_vues
    detections_actuelles = set()
    for d in detections:
        cle = (d["nom"], d["position"])
        detections_actuelles.add(cle)
        if cle not in dernieres_detections_vues:
            d["priorite"] += BONUS_NOUVEAUTE
    dernieres_detections_vues = detections_actuelles

    detections.sort(key=lambda d: d["priorite"], reverse=True)
    return detections


def construire_message(detections):
    """Construit la phrase à faire prononcer par le téléphone.
    Reste volontairement très court : un seul élément annoncé, et en
    cas de danger (plusieurs obstacles proches), juste la direction à
    prendre — sans répéter la liste des obstacles."""
    proches = [d for d in detections if d["distance"] != "loin"]

    if not proches:
        return "Rien à signaler."

    positions_occupees = set()
    nombre_tres_proche = 0
    for d in proches:
        positions_occupees.add(d["position"])
        if d["distance"] == "très proche":
            nombre_tres_proche += 1

    proches_tries = sorted(proches, key=lambda d: d["priorite"], reverse=True)
    principal = proches_tries[0]
    if principal["type"] == "objet":
        description = f"{principal['nom']} {position_objet_texte(principal['position'])}, {principal['distance']}."
    else:
        description = f"{principal['nom']} {position_surface_texte(principal['position'])}, {principal['distance']}."

    obstacles_multiples = len(proches) >= 2
    danger_immediat = nombre_tres_proche >= 1

    if obstacles_multiples and danger_immediat:
        toutes_positions = {"gauche", "centre", "droite"}
        positions_libres = toutes_positions - positions_occupees

        if "centre" in positions_libres:
            direction_calculee = "Avancez tout droit."
        elif "gauche" in positions_libres:
            direction_calculee = "Allez à gauche."
        elif "droite" in positions_libres:
            direction_calculee = "Allez à droite."
        else:
            direction_calculee = "Arrêtez-vous."

        # La direction est stabilisée (anti-erratique) mais la description
        # de ce qui est devant l'utilisateur reste toujours à jour, pour un
        # message naturel du type "Personne devant vous, très proche. Allez
        # à gauche." plutôt qu'un simple "Attention, allez à gauche."
        direction_stable = _conseil_stabilise(direction_calculee)
        return f"{description} {direction_stable}"

    if danger_immediat:
        return "Attention. " + description

    return description


def _conseil_stabilise(conseil_calcule):
    """N'annonce un nouveau conseil de direction que s'il est confirmé
    sur 2 captures consécutives, pour éviter les allers-retours erratiques
    gauche/droite dus au tremblement naturel de la détection."""
    global dernier_conseil_confirme, conseil_en_attente, compteur_confirmation

    if conseil_calcule == dernier_conseil_confirme:
        conseil_en_attente = None
        compteur_confirmation = 0
        return dernier_conseil_confirme

    if conseil_calcule == conseil_en_attente:
        compteur_confirmation += 1
    else:
        conseil_en_attente = conseil_calcule
        compteur_confirmation = 1

    if compteur_confirmation >= 2:
        dernier_conseil_confirme = conseil_calcule
        conseil_en_attente = None
        compteur_confirmation = 0
        return dernier_conseil_confirme

    # Pas encore confirmé : on garde le conseil précédent le temps de
    # confirmer le changement, sauf s'il n'y en avait pas encore.
    return dernier_conseil_confirme if dernier_conseil_confirme else conseil_calcule


# ------------------------------------------------------------------
# Serveur FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="SmartVision API")

# Autorise les requêtes venant de n'importe quelle origine (y compris
# les pages HTML ouvertes en local avec file://, qui ont une origine
# "null"). Nécessaire pour que le téléphone/laptop puisse appeler
# /analyser sans être bloqué par la politique CORS du navigateur.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyser")
async def analyser(image: UploadFile = File(...)):
    """Reçoit une image (multipart/form-data, champ 'image'), renvoie
    le JSON des détections + le message à prononcer."""
    donnees = await image.read()
    tableau = np.frombuffer(donnees, dtype=np.uint8)
    frame = cv2.imdecode(tableau, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Image illisible")

    detections = analyser_image(frame)
    message = construire_message(detections)
    priorite_max = max((d["priorite"] for d in detections), default=0)

    return {
        "detections": detections,
        "message": message,
        "urgent": priorite_max >= 7,
    }


@app.get("/")
def accueil():
    return {"statut": "Serveur SmartVision actif. Envoie une image en POST sur /analyser."}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("✅ Serveur prêt. Trouve l'IP de ce laptop avec :")
    print("   Linux/Mac : hostname -I")
    print("   Windows   : ipconfig")
    print("Puis sur le téléphone, utilise : http://<CETTE_IP>:5000/analyser")
    print("Doc interactive : http://<CETTE_IP>:5000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5000)
