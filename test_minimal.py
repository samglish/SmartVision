#!/usr/bin/env python3
"""
TEST MINIMAL — Détection d'obstacles pour aveugles
Fonctionne même si la synthèse vocale n'est pas installée
Auteur: BEIDI DINA SAMUEL
"""

import cv2
import time
import sys

print("=" * 60)
print("  TEST MINIMAL — Détection d'obstacles")
print("=" * 60)

# --- 1. Charger YOLOv8 ---
print("\n[1/4] Chargement de YOLOv8n...")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("     ✅ Modèle chargé (téléchargement auto si 1ère fois)")
except Exception as e:
    print(f"     ❌ ERREUR: {e}")
    print("     → Installe avec: pip install ultralytics")
    sys.exit(1)

# --- 2. Initialiser TTS (optionnel) ---
print("\n[2/4] Initialisation de la synthèse vocale...")
tts_ok = False
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    tts_ok = True
    print("     ✅ Synthèse vocale prête")
except Exception as e:
    print(f"     ⚠️  TTS indisponible: {e}")
    print("     → Mode TEXTE uniquement (les phrases s'afficheront)")
    print("     → Pour installer: pip install pyttsx3")
    engine = None

# --- 3. Ouvrir la webcam ---
print("\n[3/4] Ouverture de la webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("     ❌ Aucune webcam détectée!")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("     ✅ Webcam ouverte: 640x480")

# --- 4. Lancer la détection ---
print("\n[4/4] Démarrage de la détection en temps réel")
print("=" * 60)
print("  Appuie sur 'q' pour quitter")
print("=" * 60)

# Classes utiles pour un aveugle (noms COCO)
CLASSES_UTILES = {
    "person": "personne",
    "car": "voiture",
    "motorcycle": "moto",
    "bus": "bus",
    "truck": "camion",
    "bicycle": "vélo",
    "chair": "chaise",
    "couch": "canapé",
    "bed": "lit",
    "dining table": "table",
    "potted plant": "plante",
    "tv": "télévision",
    "laptop": "ordinateur",
    "refrigerator": "réfrigérateur",
    "microwave": "micro-ondes",
    "oven": "four",
    "sink": "évier",
    "toilet": "toilettes",
    "book": "livre",
    "clock": "horloge",
    "vase": "vase",
    "backpack": "sac à dos",
    "umbrella": "parapluie",
    "handbag": "sac",
    "tie": "cravate",
    "suitcase": "valise",
    "frisbee": "frisbee",
    "skis": "skis",
    "sports ball": "ballon",
    "kite": "cerf-volant",
    "baseball bat": "batte",
    "skateboard": "skateboard",
    "surfboard": "planche de surf",
    "tennis racket": "raquette",
    "bottle": "bouteille",
    "wine glass": "verre",
    "cup": "tasse",
    "fork": "fourchette",
    "knife": "couteau",
    "spoon": "cuillère",
    "bowl": "bol",
    "banana": "banane",
    "apple": "pomme",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "brocoli",
    "carrot": "carotte",
    "hot dog": "hot-dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "gâteau",
    "bench": "banc",
    "bird": "oiseau",
    "cat": "chat",
    "dog": "chien",
    "horse": "cheval",
    "sheep": "mouton",
    "cow": "vache",
    "elephant": "éléphant",
    "bear": "ours",
    "zebra": "zèbre",
    "giraffe": "girafe",
    "teddy bear": "ours en peluche",
    "hair drier": "sèche-cheveux",
    "toothbrush": "brosse à dents",
}

def position_texte(x_center, largeur):
    ratio = x_center / largeur
    if ratio < 0.33:
        return "gauche"
    elif ratio > 0.66:
        return "droite"
    else:
        return "centre"

def distance_texte(box_w, box_h, frame_w, frame_h):
    ratio = (box_w * box_h) / (frame_w * frame_h)
    if ratio > 0.25:
        return "très proche"
    elif ratio > 0.08:
        return "proche"
    elif ratio > 0.015:
        return "à moyenne distance"
    else:
        return "loin"

def generer_phrase(nom_fr, position, distance):
    if position == "centre" and distance in ["très proche", "proche"]:
        if nom_fr == "personne":
            return f"⚠️ ATTENTION — Personne {distance} devant vous!"
        elif nom_fr in ["voiture", "moto", "bus", "camion", "vélo"]:
            return f"🚨 DANGER — {nom_fr.capitalize()} {distance} devant vous!"
        else:
            return f"⚠️ Attention — {nom_fr.capitalize()} {distance} devant vous"

    if nom_fr == "personne":
        return f"👤 Personne {distance} sur votre {position}"
    elif nom_fr in ["voiture", "moto", "bus", "camion", "vélo"]:
        return f"🚗 {nom_fr.capitalize()} {distance} sur votre {position}"
    else:
        return f"📦 {nom_fr.capitalize()} {distance} sur votre {position}"

# Boucle principale
fps_time = time.time()
fps_count = 0
fps_display = 0

dernier_message = ""
dernier_temps = 0
DELAI_MESSAGE = 3  # secondes entre deux messages identiques

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Détection YOLO
    results = model(frame, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            nom_en = model.names[cls_id]

            if nom_en not in CLASSES_UTILES or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            xc = (x1 + x2) / 2
            bw, bh = x2 - x1, y2 - y1

            pos = position_texte(xc, w)
            dist = distance_texte(bw, bh, w, h)
            nom_fr = CLASSES_UTILES[nom_en]
            phrase = generer_phrase(nom_fr, pos, dist)

            detections.append({
                'phrase': phrase,
                'conf': conf,
                'box': (x1, y1, x2, y2),
                'nom': nom_fr,
                'critique': (nom_fr == "personne" and pos == "centre" and dist in ["très proche", "proche"])
            })

            # Couleur selon criticité
            if detections[-1]['critique']:
                couleur = (0, 0, 255)  # Rouge
            elif nom_fr == "personne":
                couleur = (0, 165, 255)  # Orange
            elif nom_fr in ["voiture", "moto", "bus", "camion", "vélo"]:
                couleur = (0, 255, 255)  # Jaune
            else:
                couleur = (0, 255, 0)  # Vert

            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            label = f"{nom_fr} {conf:.2f} | {pos} | {dist}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

    # Prononcer le message le plus critique
    if detections:
        detections.sort(key=lambda d: d['critique'], reverse=True)
        msg = detections[0]['phrase']

        temps_now = time.time()
        if msg != dernier_message or (temps_now - dernier_temps) > DELAI_MESSAGE:
            dernier_message = msg
            dernier_temps = temps_now
            print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

            if tts_ok and engine:
                try:
                    engine.say(msg.replace("⚠️", "").replace("🚨", "").replace("👤", "").replace("🚗", "").replace("📦", ""))
                    engine.runAndWait()
                except:
                    pass

    # FPS
    fps_count += 1
    if time.time() - fps_time >= 1:
        fps_display = fps_count
        fps_count = 0
        fps_time = time.time()

    cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Lunettes Aveugle — Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Session terminée. À bientôt!")
