# SmartVision — Assistive Navigation System for Blind and Visually Impaired Users

## 1. Project Overview

SmartVision is an assistive perception system designed to help blind and visually
impaired people navigate their environment safely. It combines **object detection**,
**semantic segmentation**, and **text-to-speech** into a single pipeline that watches
a camera feed, identifies what matters for safe navigation, ranks it by urgency, and
speaks it out loud in French.

The project was developed iteratively, starting from isolated proof-of-concept
scripts (object detection alone, segmentation alone) and progressively merged into a
single fused pipeline, then split into a **client-server architecture** so it can run
on affordable hardware (an existing Android phone) without requiring expensive
dedicated hardware.

This document explains, in full detail, what has been built, why each design
decision was made, how to install and run every component, what its known
limitations are, and what remains to be done.

**Context note:** this project is being developed with the Cameroonian context in
mind (local traffic patterns, tricycles/"keke" vehicles, mixed road surfaces,
intermittent connectivity, tight hardware budgets). Several design decisions
(offline text-to-speech, tight budget hardware plan, specific object classes) are
driven by this context.

---

## 2. Table of Contents

1. Project Overview
2. Table of Contents
3. High-Level Architecture
4. Why a Client-Server Architecture?
5. Core Concepts
   - Object Detection (YOLO)
   - Semantic Segmentation (SegFormer)
   - Fusion and Priority System
   - Distance and Position Estimation
   - Text-to-Speech
6. Repository / File Structure
7. Detailed File-by-File Documentation
8. Installation Guide
9. Usage Guide
10. Network Setup (Phone ↔ Server)
11. Configuration and Tuning
12. Known Limitations
13. Troubleshooting
14. Performance Notes
15. Roadmap / Next Steps
16. Design Decisions and Rationale (FAQ style)
17. Credits and Model Sources

---

## 3. High-Level Architecture

```
┌─────────────────────────┐        Wi-Fi (local network)        ┌──────────────────────────┐
│         PHONE           │ ───────────────────────────────────▶│          SERVER          │
│  (smartvision_telephone │        photo every ~2.5s (HTTP POST)│  (serveur_smartvision.py)│
│         .html)          │                                     │                          │
│                         │ ◀───────────────────────────────────│  YOLOv8 (objects)        │
│  - Live camera preview  │        JSON: detections + message   │  SegFormer (surfaces)    │
│  - Captures frames      │                                     │  Fusion + priority       │
│  - Sends to server      │                                     │  ranking                 │
│  - Speaks result via    │                                     │                          │
│    Web Speech API       │                                     │                          │
└─────────────────────────┘                                     └──────────────────────────┘
```

The system is split into two independent halves:

- **The server** runs the heavy AI models (object detection + segmentation). It is
  meant to run on a laptop or PC — a machine with real CPU power. It exposes a
  simple HTTP API.
- **The phone client** is a plain HTML/JavaScript web page. It does not run any AI
  model itself. Its only jobs are: show a live camera preview, periodically capture
  a frame, send it to the server, and speak the response using the phone's built-in
  text-to-speech engine.

Both devices must be connected to the **same local Wi-Fi network**. No internet
access is required for this to work (only local network connectivity), although the
budget/connectivity questionnaire answered during development indicated that
occasional internet access is acceptable for this project, so this constraint is
soft rather than hard.

---

## 4. Why a Client-Server Architecture?

Early in development, everything (YOLOv8m + SegFormer + fusion + text-to-speech) ran
in a single script directly on a laptop webcam feed. Real-world testing on a
mid-range laptop (HP Folio, Intel Core i5, no dedicated GPU) showed that running two
neural networks per frame is too heavy for smooth real-time use on this class of
hardware. Several optimization attempts were made, in order:

1. Reducing webcam capture resolution (640x480) — modest improvement.
2. Running the segmentation model only every N frames instead of every frame —
   modest improvement.
3. Moving the segmentation inference to a separate background thread so it never
   blocks the main capture/detection loop — architecturally correct, but did not
   solve the underlying CPU throughput ceiling.
4. Exporting the YOLO model to Intel's OpenVINO format for CPU-optimized inference —
   blocked in practice by unreliable internet connectivity during package
   installation (the `openvino` Python package is ~57 MB and repeatedly failed to
   download completely).

After these attempts, the conclusion was that trying to run both models in real time
on this specific laptop was not a productive direction, and that squeezing more
performance out of a phone (generally *not* more powerful than this laptop) would be
even less productive.

Given the project's hardware budget constraint ("very tight, reuse an existing
Android phone"), the most sensible architecture is to **keep all heavy computation
on the most powerful machine available** (the laptop) and use the phone purely as a
camera + speaker + microphone-free remote control. This is a standard and proven
pattern for deploying heavy AI models to lightweight/mobile front-ends.

---

## 5. Core Concepts

### 5.1 Object Detection (YOLO)

We use **YOLOv8**, specifically a variant pretrained on **Open Images V7 (OIV7)**,
which recognizes **601 distinct object classes** (`yolov8m-oiv7.pt`). This is a much
richer vocabulary than the standard COCO-trained YOLO models (80 classes), and
notably includes classes absent from COCO but useful for navigation, such as
`Door`, `Window`, and `Stairs`.

Two model sizes were evaluated:

- `yolov8s-oiv7.pt` (small) — fast, but recall on small/distant objects (motorcycles,
  tricycles, far-away cars) was poor in real street-scene testing.
- `yolov8m-oiv7.pt` (medium) — noticeably better recall (roughly doubled the number
  of correctly detected vehicles on a test image of a busy road), at the cost of
  being heavier to run. This is the model currently used throughout the project.

**Important limitation discovered during testing:** even with the medium model,
detection of **tricycles ("keke"/auto-rickshaws)** and, in some tests, **motorcycles
under riders**, remains weak. This is very likely a **dataset domain gap**: Open
Images V7 was collected predominantly from Western/global sources and under-represents
vehicle types and street scenes typical of West/Central Africa. Increasing the
confidence threshold did not help, which confirms this is a recognition
problem, not a confidence-calibration problem. Solving this properly would require
fine-tuning YOLO on a custom, locally-collected and annotated dataset — this has
been identified but not yet undertaken (see Roadmap).

Rather than exposing all 601 classes to the end user (which was tried and explicitly
rejected — see Section 16), the pipeline filters detections down to a **curated
whitelist** of classes relevant to navigation safety and daily life at home. See
`priorite_objets` in the code for the exact list, which includes:

- **Moving hazards**: Person, Dog
- **Vehicles**: Car, Truck, Bus, Motorcycle, Bicycle, Train, Taxi, Van, plus generic
  `Land vehicle` / `Vehicle` fallback classes added specifically to try to catch
  atypical vehicles (tricycles, etc.) that don't cleanly match a specific class
- **Structural elements**: Door, Window, Stairs
- **Traffic/safety signage**: Traffic light, Traffic sign, Stop sign, Fire hydrant,
  Street light
- **Low obstacles / trip hazards**: Bench, Chair, Table, Waste container, Suitcase,
  Luggage and bags, Ladder
- **Household items** (explicitly requested for the Cameroonian home-use context,
  kept deliberately small): Mobile phone, Bed, Refrigerator, Gas stove (flagged with
  higher priority due to burn risk), Laptop

### 5.2 Semantic Segmentation (SegFormer)

Object detectors like YOLO draw bounding boxes around discrete objects, but they
cannot describe continuous surfaces like floors, walls, roads, or water. For that,
the project uses **SegFormer** (`nvidia/segformer-b0-finetuned-ade-512-512`), a
transformer-based semantic segmentation model pretrained on **ADE20K**, a dataset
with **150 scene classes** covering both objects and surfaces.

Only a **subset of ADE20K classes** relevant to navigation is retained:

| ADE20K class ID | Name (EN) | French label used | Notes |
|---|---|---|---|
| 3 | floor | Sol | Low priority — informational, not a hazard |
| 0 | wall | Mur | Low priority |
| 38 | railing | Rambarde | Higher priority — fall risk |
| 95 | bannister | Main courante | Higher priority — fall risk |
| 6 | road | Route | Added later; moderate priority (traffic zone) |
| 11 | sidewalk | Trottoir | Low priority (normally safe) |
| 21 | water | Eau | Grouped label |
| 26 | sea | Eau | Grouped label |
| 60 | river | Eau | Grouped label |
| 109 | swimming pool | Eau | Grouped label |
| 128 | lake | Eau | Grouped label |

**Deliberate design choice:** `Door`, `Window`, and `Stairs` are intentionally
**excluded** from the segmentation side, even though ADE20K has classes for them,
because YOLO already detects these as discrete objects with bounding boxes, which
give more precise position/distance estimates than a segmentation mask's area
proportion. Keeping them in both systems would cause duplicate, confusing
announcements.

**Verified capability:** testing confirmed that ADE20K's `road` class generalizes
reasonably well even to unpaved/gravel countryside roads, not just to the paved
urban roads it is more commonly associated with. `water` was also correctly
detected on a real puddle photo.

**Known gap:** ADE20K has **no class for potholes or ground-level holes**. This is
a genuinely hard computer vision problem (see Section 12 and Section 15) and is not
solved by this segmentation model or any standard semantic segmentation dataset.
"Corridor ahead" / "open path ahead" is likewise not directly represented as a
class; it would need to be inferred from floor mask geometry, which has not yet been
implemented.

### 5.3 Fusion and Priority System

Detections from YOLO (objects) and SegFormer (surfaces) are merged into a single
list and sorted by a **priority score**. This score is the sum of:

1. A **base priority** per class (0–10 scale), reflecting how dangerous or
   important that class generally is. Examples: Stairs = 9, Person = 8,
   Car/Truck/Bus/Motorcycle/Van/Taxi/Train = 8, Traffic light = 7, Door = 6,
   Water = 6, Railing/Bannister = 6, Floor = 0 (informational only), Sidewalk = 1.
2. A **distance bonus/penalty**, so that the same class of object is prioritized
   higher when it is close and lower when it is far:
   - "très proche" (very close): +4
   - "proche" (close): +2
   - "à moyenne distance" (medium distance): +0
   - "loin" (far): −2

This addresses an early design flaw noted during testing: initially, priority was
based purely on object class, so a railing and a handrail at very different
distances could tie in priority even though the closer one is clearly more urgent.
The distance bonus fixes this.

Detections with priority ≥ 7 are treated as **urgent**: the text-to-speech queue is
cleared immediately so the urgent message is spoken without delay, interrupting
whatever was being said before.

### 5.4 Distance and Position Estimation

**Important limitation, stated plainly:** none of the distance estimates in this
project are true metric distances (meters). There is no depth sensor and no camera
calibration. Distance is estimated purely from **how much of the image frame** a
detected object or surface occupies (its bounding-box or mask area as a proportion
of the total image area):

| Proportion of frame | Label |
|---|---|
| > 35% | très proche (very close) |
| > 15% | proche (close) |
| > 5% | à moyenne distance (medium distance) |
| ≤ 5% | loin (far) |

This is a reasonable and fast heuristic for relative distance, but it is not
calibrated to real-world meters. When the user requested that objects "beyond 3
meters" not be spoken aloud, this was implemented as "don't speak anything
categorized as 'loin'" — an approximation, explicitly flagged as such to the user.
A properly calibrated system would require either a depth-sensing camera or a
calibration procedure (known object size + focal length).

Position is estimated by dividing the image into three horizontal thirds
(left/center/right), based on the horizontal center of the bounding box (for
objects) or the horizontal centroid of the mask (for surfaces).

### 5.5 Text-to-Speech

Two different TTS approaches are used depending on which half of the system is
speaking:

- **On the laptop/server side** (used in the original single-machine prototypes,
  before the client-server split): `espeak-ng`, called directly as a command-line
  subprocess, running in its own background thread with a queue, so speech never
  blocks the video/detection loop. `pyttsx3` was tried first but abandoned after
  hitting a known compatibility bug between `pyttsx3`'s internal driver and
  `espeak-ng` (`ValueError: SetVoiceByName failed ... voice: gmw/en`), which occurs
  when both `espeak` and `espeak-ng` are present on a system. Calling the `espeak-ng`
  binary directly sidesteps this bug entirely and is simpler and more robust,
  including for future deployment on a Raspberry Pi.
- **On the phone side** (current client-server architecture): the browser's native
  **Web Speech API** (`SpeechSynthesis`), which is built into Chrome for Android
  and requires no extra installation. This is generally higher audio quality than
  `espeak-ng`.

In both cases, the speech system supports **interruption**: an urgent, high-priority
message clears any pending/ongoing speech so it is announced immediately, rather
than waiting in a queue behind less important announcements.

---

## 6. Repository / File Structure

```
.
├── segmentation.py                  # Standalone SegFormer test on a single image (original prototype)
├── test_scene_v2.py                 # Early YOLO-only prototype, whitelist too narrow (Door/Window/Stairs/Person/...)
├── test_scene_v5.py                 # YOLO-only, curated navigation-safety whitelist, no speech yet
├── test_scene_v6.py                 # YOLO-only + real text-to-speech via espeak-ng, threaded
├── test_scene_v7_fusion.py          # FULL pipeline: YOLO + SegFormer + fusion + priority + speech, LIVE WEBCAM
├── test_scene_v7_image.py           # Same full pipeline, but on a STATIC IMAGE FILE (for testing without a live scene)
├── serveur_smartvision.py           # FastAPI server: exposes the full pipeline over HTTP for the phone client
├── smartvision_telephone.html       # Phone-side web client: camera capture + upload + native TTS
└── README.md                        # This file
```

Note: `test_scene_v3.py` and `test_scene_v4.py` were intermediate iterations
(full 601-class vocabulary mode, then a hybrid continuous/on-demand mode) that were
explicitly superseded and removed after user feedback that a full-vocabulary
announcement stream would be confusing and unsafe for a blind user in motion (see
Section 16 for the full rationale).

---

## 7. Detailed File-by-File Documentation

### 7.1 `segmentation.py`

The original standalone segmentation prototype. Loads
`nvidia/segformer-b0-finetuned-ade-512-512`, runs it on a single image file passed
as a command-line argument, prints the percentage of the image occupied by each
class of interest (`wall`, `floor`, `door`, `stairs`, `railing`, `bannister`,
`windowpane`, `ceiling`, `sky`, `building`), and saves a color-overlaid visualization
to `segmentation_resultat.png`.

Usage:
```bash
python segmentation.py path/to/image.jpg
```

### 7.2 `test_scene_v5.py`

A YOLO-only pipeline (no segmentation, no speech) using a deliberately restricted
whitelist of ~22 navigation-relevant classes (see Section 5.1). Draws bounding
boxes color-coded by priority tier and prints a "state of the scene" summary to the
terminal when the `i` key is pressed. This was the version where the "full 601-class
vocabulary is not appropriate for continuous navigation use" decision was finalized.

### 7.3 `test_scene_v6.py`

Adds real, working text-to-speech to the `test_scene_v5.py` pipeline, using
`espeak-ng` via subprocess in a dedicated background thread with a message queue.
Urgent detections (priority ≥ 7) clear the queue and interrupt ongoing speech.

### 7.4 `test_scene_v7_fusion.py`

The most complete single-machine pipeline: combines YOLO object detection (every
frame) with SegFormer surface segmentation (every 30 frames, since it is much more
computationally expensive), merges and prioritizes both, draws everything on the
video feed, announces new detections via speech (skipping anything categorized as
"loin"/far), and displays a live FPS counter.

Key implementation detail: SegFormer inference runs in **its own background
thread**, decoupled from the main capture/YOLO/display loop via a lock-protected
shared variable, specifically to avoid the periodic multi-hundred-millisecond
freezes that occurred when segmentation was called synchronously inline (even only
every 30th frame).

Supports two interaction modes:
- **Continuous mode**: automatically announces newly-appeared, close-enough
  detections as they occur.
- **On-demand mode** (`i` key): immediately announces everything currently deemed
  relevant in the frame, regardless of whether it was already announced, and
  regardless of distance filtering (an explicit user request bypasses the "don't
  announce far things" rule, since the user is actively asking).

Usage:
```bash
python test_scene_v7_fusion.py
```
Controls: `q` to quit, `i` to get a full status announcement of the current scene.

### 7.5 `test_scene_v7_image.py`

Functionally identical detection/fusion/priority logic to
`test_scene_v7_fusion.py`, but applied once to a static image file instead of a
live webcam feed. This was built specifically so that scenarios not physically
available at testing time (e.g., a busy road, or a puddle of water) could still be
tested using a representative photo. Saves an annotated output image to
`fusion_resultat_image.png`.

Usage:
```bash
python test_scene_v7_image.py path/to/photo.jpg
```

### 7.6 `serveur_smartvision.py`

A **FastAPI** server exposing the fusion pipeline (identical detection/priority
logic to `test_scene_v7_fusion.py`, minus any drawing/display code, since there is
no local screen to draw to) over HTTP. Loads both models once at startup (not per
request, which would be far too slow).

Endpoints:
- `POST /analyser` — accepts a `multipart/form-data` request with an `image` field
  (a JPEG frame), returns JSON:
  ```json
  {
    "detections": [ { "type": "objet"|"surface", "nom": "...", "distance": "...", "position": "...", "priorite": 0 }, ... ],
    "message": "Autour de vous : ...",
    "urgent": true|false
  }
  ```
- `GET /` — basic health-check message.
- `GET /docs` — automatic interactive API documentation (Swagger UI), provided free
  by FastAPI; useful for testing the pipeline with any image directly from a
  browser, without needing the phone client.

Automatically prefers a pre-exported OpenVINO version of the YOLO model
(`yolov8m-oiv7_openvino_model/`) if a valid `.xml` file is present, and falls back
to the standard PyTorch `.pt` weights otherwise.

Run with either:
```bash
python serveur_smartvision.py
```
or
```bash
uvicorn serveur_smartvision:app --host 0.0.0.0 --port 5000
```

### 7.7 `smartvision_telephone.html`

A self-contained, dependency-free HTML/JavaScript page meant to be opened in
**Chrome on Android**. Responsibilities:

1. Requests access to the phone's rear-facing camera (`getUserMedia`) and displays
   a live preview.
2. Every 2.5 seconds (configurable via `DELAI_ENTRE_CAPTURES_MS`), silently
   captures a JPEG frame from the video feed and `POST`s it to the configured
   server address's `/analyser` endpoint.
3. On receiving a JSON response, speaks the `message` field using the browser's
   native `SpeechSynthesis` API (French voice, `fr-FR`), cancelling any speech
   already in progress so messages never queue up and fall behind real time.
4. Persists the server address in `localStorage` between sessions.

The server address must be entered manually in the page (e.g.
`http://192.168.1.10:5000`) — see Section 10 for how to find it.

---

## 8. Installation Guide

### 8.1 Server-side (laptop/PC) dependencies

```bash
pip install fastapi uvicorn python-multipart
pip install ultralytics opencv-python
pip install transformers torch torchvision pillow numpy
```

Optional, for CPU-optimized YOLO inference on Intel processors:
```bash
pip install openvino
```
(Note: this package is relatively large, ~57 MB. If your internet connection is
unstable, downloads may time out — see Section 13, Troubleshooting.)

### 8.2 Text-to-speech engine (only needed for single-machine scripts, not the
phone client, which uses the browser's built-in speech)

On Debian/Ubuntu/Kali/Parrot-based Linux:
```bash
sudo apt install espeak-ng
```

**Known issue:** if both `espeak` and `espeak-ng` are installed simultaneously,
Python bindings libraries such as `pyttsx3` can throw:
```
ValueError: SetVoiceByName failed with unknown return code -1 for voice: gmw/en
```
This project avoids the issue entirely by calling the `espeak-ng` (or `espeak`)
binary directly via `subprocess`, rather than using `pyttsx3`'s internal driver
bindings.

### 8.3 Phone-side

No installation needed beyond a modern version of **Chrome for Android**. Transfer
`smartvision_telephone.html` to the phone by any means (email to self, USB cable,
Bluetooth, cloud storage) and open it directly in Chrome.

**Important caveat:** browsers generally require a secure context (HTTPS) for
camera access via `getUserMedia`, with an exception for `localhost`. Opening the
file directly from local storage (`file://...`) may or may not be permitted
depending on the Chrome version and Android security settings. If camera access is
blocked, the recommended fix is to serve the HTML file over a simple local web
server (even `python -m http.server`, run on the same laptop as the API server, on
a different port) instead of opening it as a raw local file.

---

## 9. Usage Guide

### 9.1 Quick single-machine test (webcam, everything on one computer)

```bash
python test_scene_v7_fusion.py
```
Press `q` to quit, `i` to request a full announcement of everything currently
detected as relevant.

### 9.2 Quick single-machine test (static image)

```bash
python test_scene_v7_image.py path/to/photo.jpg
```
Check the generated `fusion_resultat_image.png` and the terminal output.

### 9.3 Full client-server setup (recommended deployment: phone + laptop)

1. On the laptop, start the server:
   ```bash
   python serveur_smartvision.py
   ```
   Wait for `✅ Serveur prêt.` in the terminal.
2. Find the laptop's local IP address (see Section 10).
3. On the phone, open `smartvision_telephone.html` in Chrome.
4. Enter the server address, e.g. `http://192.168.1.10:5000`.
5. Tap **Démarrer** (Start). Grant camera permission when prompted.
6. The phone will now periodically capture frames, send them to the laptop, and
   speak the results aloud.
7. Tap **Arrêter** (Stop) to end the session.

---

## 10. Network Setup (Phone ↔ Server)

Both devices **must be on the same local Wi-Fi network**. To find the laptop's
local IP address:

- **Linux/macOS:**
  ```bash
  hostname -I
  ```
  or
  ```bash
  ip addr
  ```
- **Windows:**
  ```bash
  ipconfig
  ```
  (look for the "IPv4 Address" under your active Wi-Fi adapter)

The server listens on port `5000` by default. The full address to enter on the
phone will look like `http://192.168.X.X:5000`.
Server test : https://smartvision-ifs5.onrender.com

If the phone cannot reach the server, check:
- Both devices are on the same Wi-Fi network (not one on Wi-Fi and one on mobile
  data).
- The laptop's firewall is not blocking incoming connections on port 5000.
- The server terminal shows it is actually running and listening.

---

## 11. Configuration and Tuning

All of the following are defined near the top of `serveur_smartvision.py` (and
mirrored in `test_scene_v7_fusion.py` / `test_scene_v7_image.py` for the
single-machine scripts) and can be adjusted without touching the rest of the logic:

| Variable | Meaning | Current value |
|---|---|---|
| `SEUIL_YOLO` | Minimum confidence for a YOLO detection to be kept | 0.20 |
| `SEUIL_SURFACE_PROPORTION` | Minimum fraction of the image a segmentation class must cover to be reported | 0.05 (5%) |
| `INTERVALLE_SEGMENTATION` | How many frames between segmentation runs (single-machine scripts only) | 30 |
| `priorite_objets` | Dict mapping YOLO class name → (French label, base priority) | see Section 5.1 |
| `priorite_surfaces` | Dict mapping ADE20K class ID → (French label, base priority) | see Section 5.2 |
| `BONUS_DISTANCE` | Distance-based priority adjustment | +4 / +2 / 0 / −2 |
| `SEUIL_URGENT` | Priority threshold above which speech is treated as urgent (interrupts queue) | 7 |
| `DELAI_ENTRE_CAPTURES_MS` (phone HTML) | Time between camera captures sent to the server | 2500 ms |

**Tuning guidance:**
- If you get too many false positives, raise `SEUIL_YOLO` and/or
  `SEUIL_SURFACE_PROPORTION`.
- If important objects are being missed, lower these thresholds — but be aware this
  was already tested (0.30 → 0.20) with negligible effect on missed tricycles/
  motorcycles, confirming that threshold tuning alone cannot fix dataset-level
  recognition gaps (see Section 12).
- If speech feels laggy relative to the phone's capture rate, this most likely
  reflects genuine server processing time (model inference speed), not a queuing
  bug — see Section 14 (Performance Notes).

---

## 12. Known Limitations

This section is intentionally explicit and honest about what does **not** work yet,
so that expectations are correctly calibrated for anyone continuing this project.

1. **No true metric distance.** All distances ("close", "far", etc.) are estimated
   from the proportion of the camera frame an object/surface occupies. This is not
   calibrated to real-world meters. A depth sensor or a calibrated
   monocular-depth model would be needed for genuine metric distance.

2. **No pothole/ground-hole detection.** Neither YOLO (object-based) nor SegFormer/
   ADE20K (which has no "hole" or "pothole" class) can detect a hole or sudden drop
   in the ground. This is a recognized, unresolved gap and one of the hardest
   remaining problems (see Roadmap).

3. **No dedicated "open corridor/path ahead" detection.** There is no direct class
   for "the way ahead is clear." This would need to be inferred (e.g., from
   contiguous floor-mask geometry extending forward) — not yet implemented.

4. **Weak detection of tricycles ("keke") and, in some tests, motorcycles with
   riders.** This is very likely due to a training-data domain gap in Open Images
   V7 (see Section 5.1). Confirmed not to be a confidence-threshold issue — lowering
   the threshold from 0.30 to 0.20 did not meaningfully improve recall on these
   vehicle types.

5. **Server-side compute is heavy for typical laptop hardware.** On a mid-range
   Intel Core i5 laptop with no dedicated GPU, running both YOLOv8m and SegFormer
   was found to be noticeably slow even after: reducing capture resolution,
   throttling segmentation frequency, moving segmentation to a background thread,
   and attempting an OpenVINO export (which could not be completed due to
   unreliable internet during package installation). This directly motivated the
   move to a client-server architecture that keeps computation on the most capable
   available machine, rather than trying to run everything on a phone.

6. **No secure/authenticated connection between phone and server.** The current
   `/analyser` endpoint accepts unauthenticated requests from anyone on the same
   local network. This is acceptable for a personal-use prototype on a private
   home/local network but should not be exposed to the open internet without adding
   authentication and HTTPS.

7. **No automated tests.** All validation so far has been manual (visual inspection
   of annotated images/video, manual reading of terminal logs, and manual listening
   to spoken output).

---

## 13. Troubleshooting

**"❌ Impossible de lire l'image : path/to/file.jpg"**
The image file path given to `test_scene_v7_image.py` does not exist at that
location, or is misspelled. Double check with `ls` (Linux/macOS) or `dir`
(Windows) in the folder you expect the file to be in.

**`ValueError: SetVoiceByName failed with unknown return code -1 for voice:
gmw/en`** (when using `pyttsx3`)
This is a known conflict between `espeak` and `espeak-ng` both being present on
the system, affecting `pyttsx3`'s internal driver. This project's scripts avoid
`pyttsx3` entirely and call `espeak-ng`/`espeak` directly via `subprocess` for this
reason. If you see this error, you are likely running an older version of a script
or a custom modification that still imports `pyttsx3`.

**`ModuleNotFoundError: No module named 'openvino'`**
The `openvino` Python package is not installed. Install it with
`pip install openvino`. If the download repeatedly times out (common on slower or
unstable connections, since the package is ~57 MB), try:
```bash
pip install --default-timeout=1000 --retries 5 openvino
```
or download the wheel file manually with a resumable download tool (e.g.
`curl -C - -O <url>`) and install it from the local file with
`pip install ./openvino-<version>.whl`.

**`StopIteration` inside `ultralytics/nn/backends/openvino.py` (`w =
next(w.glob("*.xml"))`)**
This means an OpenVINO export folder exists (e.g. `yolov8m-oiv7_openvino_model/`)
but is incomplete/empty — typically because a previous export attempt ran before
the `openvino` package was successfully installed. Fix: delete the incomplete
folder and re-run so the export happens again now that the dependency is present:
```bash
rm -rf yolov8m-oiv7_openvino_model
python serveur_smartvision.py
```
The scripts in this project check for the specific `.xml` file inside the export
folder (not just the folder's existence) specifically to avoid silently reusing a
broken export in the future.

**Camera access blocked on the phone / `getUserMedia` fails**
Browsers generally require HTTPS (or `localhost`) for camera access. If opening the
HTML file directly (`file://...`) is blocked by Chrome, serve it via a simple local
web server instead (e.g. `python -m http.server 8080` from the folder containing
the HTML file, then visit `http://<laptop-ip>:8080/smartvision_telephone.html` from
the phone).

**Phone shows "❌ Impossible de joindre le serveur"**
Check that: (1) the phone and server are on the same Wi-Fi network, (2) the IP
address entered on the phone is correct and current (it can change if the laptop
reconnects to Wi-Fi), (3) the server terminal shows it is running, (4) no firewall
on the laptop is blocking port 5000.

**Many visible objects (vehicles, etc.) are not detected**
First check the confidence threshold and model size (see Section 11 and Section
5.1). If detections remain poor for a specific vehicle type even at a low
threshold and with the medium model, this is likely a genuine dataset gap (see
Section 12, point 4) rather than something fixable by configuration alone.

---

## 14. Performance Notes

- `yolov8s-oiv7.pt` is fast but has poor recall on small/distant/atypical objects.
- `yolov8m-oiv7.pt` has meaningfully better recall (roughly 2x more correctly
  detected vehicles in one real-world test image) but is heavier to run.
- SegFormer (`segformer-b0`, the smallest variant in its family) is still
  considerably more expensive per inference than YOLO, which is why it is only run
  periodically (every 30 frames) rather than every frame, and why it runs in its
  own background thread rather than blocking the main loop.
- On the tested hardware (Intel Core i5 laptop, no dedicated GPU), even with all of
  the above optimizations, real-time performance running both models on a single
  machine was judged insufficient for a smooth user experience. This is the primary
  reason for the client-server split.
- The phone-side capture interval (2.5 seconds) is intentionally matched to
  roughly the pace at which the server can realistically process and respond,
  rather than attempting to stream every camera frame (which would be wasteful of
  both bandwidth and server compute, since the server cannot keep up with true
  video framerates on this hardware anyway).

---

## 15. Roadmap / Next Steps

In rough priority order, based on project discussions so far:

1. **Validate the phone-server architecture end-to-end in real-world outdoor
   conditions** (not just local network tests) — check latency, Wi-Fi range if
   applicable, and battery drain on the phone from keeping the camera and network
   active continuously.
2. **Ground-hole / pothole detection.** Proposed approach: monocular depth
   estimation (e.g., a lightweight MiDaS variant) restricted to the region already
   identified as "floor" by the segmentation model, looking for localized sudden
   depth discontinuities. This is acknowledged to be experimental and will need
   careful validation before being trusted for a safety-critical alert — the
   project intentionally deferred this rather than shipping an unreliable hole
   detector.
3. **"Open path/corridor ahead" detection**, likely inferable from the geometry of
   the floor segmentation mask (e.g., measuring how far a contiguous floor region
   extends vertically in front of the user) rather than requiring a new model.
4. **Custom fine-tuning of YOLO on locally-collected data** to address the
   tricycle/motorcycle detection gap — would require collecting and annotating a
   local dataset of Cameroonian street scenes.
5. **Security hardening** of the server API (authentication, HTTPS) before any
   deployment beyond a private local network.
6. **Physical form factor** considerations (mounting, battery life for extended
   phone use, durability against dust/rain relevant to the target environment) —
   not yet addressed.
7. **Revisit OpenVINO (or an alternative CPU-optimization path)** if/when a more
   reliable internet connection is available, since it remains a promising
   unexplored lever for improving server-side inference speed on Intel hardware.

---

## 16. Design Decisions and Rationale (FAQ style)

**Q: Why not just announce everything the model detects (all 601 YOLO classes /
150 segmentation classes)?**
This was actually built and tested (see the now-removed `test_scene_v3.py` /
`test_scene_v4.py` prototypes), and explicitly rejected. Continuously narrating
irrelevant objects (an accordion, a wine glass, a teddy bear, etc.) while someone is
walking creates auditory noise that can mask genuinely important safety
information and cause "alert fatigue," where the user starts tuning out the system
entirely. The system now uses a small, deliberately curated whitelist for
continuous navigation, calibrated specifically around what matters for a blind
user's safety and daily life (see Section 5.1). A broader on-demand
"identify everything" mode was prototyped but ultimately also removed in favor of
keeping even the explicit query mode focused on the same curated, relevant list —
the user specifically requested this narrowing.

**Q: Why estimate distance from frame-occupancy percentage instead of something
more accurate?**
Because no depth sensor or camera calibration is currently part of the system. This
is a deliberate, low-cost, "good enough for relative urgency ranking" heuristic,
not a claim of metric accuracy. It has been explicitly flagged to the user as an
approximation each time it came up (e.g., when implementing the "don't announce
things farther than ~3 meters" request).

**Q: Why exclude Door/Window/Stairs from the segmentation side, given ADE20K has
classes for them?**
To avoid duplicate detections/announcements for the same real-world object from two
different subsystems. YOLO's bounding-box-based estimate is more precise for these
specific discrete, well-defined objects; segmentation is reserved for things YOLO
fundamentally cannot represent (continuous surfaces: floor, wall, road, water,
railings).

**Q: Why FastAPI instead of Flask for the server?**
Purely a stated developer preference. Flask was used in an earlier draft/version of
this same server and was fully functional; FastAPI was chosen for the final version
because the developer explicitly requested it, and because it additionally provides
automatic interactive API documentation (`/docs`), which is a convenient bonus for
testing the pipeline without needing the phone client.

**Q: Why the phone's native browser speech (Web Speech API) instead of sending
audio from the server, or using espeak-ng on the phone?**
The phone already has a good-quality, zero-installation, offline-capable
text-to-speech engine built into its browser/OS. Using it avoids having to stream
audio data over the network (adding latency and bandwidth usage) and avoids trying
to install/run `espeak-ng` in a mobile browser context, which is not straightforward.
The server only needs to send a short text string, which is far cheaper.

**Q: Why keep two nearly-identical pipeline scripts
(`test_scene_v7_fusion.py` for webcam, `test_scene_v7_image.py` for static images)
instead of one?**
They serve different purposes: the webcam version is the closest thing to the real
end-user product, useful for real-time behavior/performance testing; the static-image
version is a fast, convenient debugging tool for testing specific scenarios (a busy
road, a puddle of water) that may not be physically available to the developer at
testing time. Both are kept manually synchronized whenever detection logic changes.

---

## 17. Credits and Model Sources

- **YOLOv8 (Open Images V7 variants)** — [Ultralytics](https://github.com/ultralytics/ultralytics)
  (`yolov8s-oiv7.pt`, `yolov8m-oiv7.pt`), trained on
  [Google's Open Images V7 dataset](https://storage.googleapis.com/openimages/web/index.html)
  (601 object classes).
- **SegFormer** — [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
  via Hugging Face `transformers`, fine-tuned on the
  [ADE20K scene parsing dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
  (150 scene classes).
- **espeak-ng** — open-source offline text-to-speech engine, used server-side for
  the single-machine prototype scripts.
- **Web Speech API (`SpeechSynthesis`)** — browser-native text-to-speech, used on
  the phone client.
- **FastAPI** / **Uvicorn** — Python web framework and ASGI server used for the
  HTTP API between phone and server.
- **OpenCV**, **PyTorch**, **Hugging Face Transformers**, **NumPy**, **Pillow** —
  supporting libraries for image processing and model inference.

---

*This README reflects the state of the project as of the most recent development
session. Sections 12 (Known Limitations) and 15 (Roadmap) should be treated as the
most current source of truth on what still needs work.*
