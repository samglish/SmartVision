from ultralytics import YOLO
import cv2

# Charger le modèle
print("Chargement de YOLOv8n...")
model = YOLO("yolov8n.pt")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la webcam")
    exit()

print("✅ Webcam ouverte")
print("Appuie sur 'q' pour quitter")


def analyser_position(x1, x2, largeur_image):
    """
    Détermine si l'objet est à gauche, au centre ou à droite.
    """
    centre_objet = (x1 + x2) / 2
    tiers = largeur_image / 3

    if centre_objet < tiers:
        return "à gauche"
    elif centre_objet < 2 * tiers:
        return "devant vous"
    else:
        return "à droite"


def analyser_distance(x1, y1, x2, y2, largeur_image, hauteur_image):
    """
    Estime la distance relative grâce à la taille
    de l'objet dans l'image.
    """

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


while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Impossible de lire la webcam")
        break

    hauteur, largeur = frame.shape[:2]

    # Détection YOLO
    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            # Coordonnées du rectangle
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Classe détectée
            cls = int(box.cls[0])
            nom = model.names[cls]

            # Confiance
            confiance = float(box.conf[0])

            # On ignore les détections faibles
            if confiance < 0.50:
                continue

            # Objets intéressants pour notre prototype
            objets_utiles = [
                "person",
                "car",
                "motorcycle",
                "bus",
                "truck",
                "bicycle",
                "chair",
                "couch",
                "bed"
            ]

            if nom not in objets_utiles:
                continue

            # Position
            position = analyser_position(
                x1,
                x2,
                largeur
            )

            # Distance
            distance = analyser_distance(
                x1,
                y1,
                x2,
                y2,
                largeur,
                hauteur
            )

            # Traduction des noms
            noms_francais = {
                "person": "Personne",
                "car": "Voiture",
                "motorcycle": "Moto",
                "bus": "Bus",
                "truck": "Camion",
                "bicycle": "Vélo",
                "chair": "Chaise",
                "couch": "Canapé",
                "bed": "Lit"
            }

            nom_fr = noms_francais.get(nom, nom)

            # Construire le message
            message = f"{nom_fr} {distance} {position}"

            print("➡️", message)

            # Dessiner le rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Afficher le message sur l'image
            cv2.putText(
                frame,
                message,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Afficher la webcam
    cv2.imshow(
        "Assistant Vision - Position et Distance",
        frame
    )

    # Quitter avec q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Programme terminé.")
