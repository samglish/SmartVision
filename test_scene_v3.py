from ultralytics import YOLO
import cv2

print("=" * 60)
print("TEST — COMPRÉHENSION DE LA SCÈNE (v2, YOLO oiv7)")
print("=" * 60)

# Charger le modèle — Open Images V7 (601 classes), connaît Door,
# Window, Stairs en plus des classes COCO habituelles.
# Pas de classe "Wall" dans ce dataset (voir note plus bas).
print("\n[1/3] Chargement du modèle YOLOv8s-oiv7...")
model = YOLO("yolov8s-oiv7.pt")
print("✅ Modèle chargé")

# Ouvrir la webcam
print("\n[2/3] Ouverture de la webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la webcam")
    exit()

print("✅ Webcam ouverte")

print("\n[3/3] Détection de la scène")
print("=" * 60)
print("Place devant la caméra :")
print("  🚪 une porte")
print("  🪟 une fenêtre")
print("  🪜 un escalier si possible")
print("\nAppuie sur 'q' pour quitter")
print("=" * 60)


def analyser_position(x1, x2, largeur_image):
    """Détermine si l'objet est à gauche, au centre ou à droite."""
    centre_objet = (x1 + x2) / 2
    tiers = largeur_image / 3

    if centre_objet < tiers:
        return "à gauche"
    elif centre_objet < 2 * tiers:
        return "devant vous"
    else:
        return "à droite"


def analyser_distance(x1, y1, x2, y2, largeur_image, hauteur_image):
    """Estime la distance relative grâce à la taille de l'objet dans l'image."""
    largeur_objet = x2 - x1
    hauteur_objet = y2 - y1

    surface = largeur_objet * hauteur_objet
    surface_image = largeur_image * hauteur_image

    proportion = surface / surface_image

    if proportion > 0.35:
        return "très proche"
    elif proportion > 0.15:
        return "proche"
    elif proportion > 0.05:
        return "à moyenne distance"
    else:
        return "loin"


# Classes que nous voulons surveiller (noms exacts Open Images V7)
# On se concentre uniquement sur les éléments structurels de la scène.
# Note : pas de "Wall" dans Open Images V7 (voir note plus haut).
objets_importants = [
    "Door",
    "Window",
    "Stairs",
]

# Traduction française
noms_francais = {
    "Door": "Porte",
    "Window": "Fenêtre",
    "Stairs": "Escalier",
}

# Pour éviter d'afficher 50 fois le même objet dans le terminal
derniers_objets = set()


while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Impossible de lire la webcam")
        break

    hauteur, largeur = frame.shape[:2]

    # Détection
    results = model(frame, verbose=False)

    objets_detectes = set()

    for result in results:

        for box in result.boxes:

            # Confiance
            confiance = float(box.conf[0])
            cls = int(box.cls[0])
            nom = model.names[cls]

            if confiance < 0.20:
                continue

            if nom not in objets_importants:
                continue

            nom_fr = noms_francais.get(nom, nom)
            objets_detectes.add(nom_fr)

            # Coordonnées
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Position et distance
            position = analyser_position(x1, x2, largeur)
            distance = analyser_distance(x1, y1, x2, y2, largeur, hauteur)

            # Affichage sur l'image (avec position/distance, pas juste la confiance)
            texte = f"{nom_fr} | {distance} | {position}"

            # Couleur selon le type
            if nom_fr == "Porte":
                couleur = (0, 0, 255)      # rouge
            elif nom_fr == "Fenêtre":
                couleur = (0, 165, 255)    # orange
            elif nom_fr == "Escalier":
                couleur = (255, 0, 0)      # bleu
            else:
                couleur = (0, 255, 0)      # vert

            cv2.rectangle(frame, (x1, y1), (x2, y2), couleur, 2)
            cv2.putText(
                frame,
                texte,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                couleur,
                2
            )

    # N'annoncer dans le terminal que les objets NOUVELLEMENT détectés
    # (comme dans test_scene.py), mais avec position + distance en plus
    nouveaux_objets = objets_detectes - derniers_objets

    for nom_fr in nouveaux_objets:
        # On retrouve la première boîte correspondant à ce nom français pour
        # afficher sa position/distance au moment de la première détection
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                nom_brut = model.names[cls]
                if noms_francais.get(nom_brut, nom_brut) == nom_fr:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    position = analyser_position(x1, x2, largeur)
                    distance = analyser_distance(x1, y1, x2, y2, largeur, hauteur)
                    print(f"🔎 Détecté : {nom_fr} — {distance}, {position}")
                    break
            else:
                continue
            break

    derniers_objets = objets_detectes

    # Afficher la fenêtre
    cv2.imshow(
        "SmartVision - Test Scene v2 (oiv7)",
        frame
    )

    # Quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("TEST TERMINÉ")
print("=" * 60)
