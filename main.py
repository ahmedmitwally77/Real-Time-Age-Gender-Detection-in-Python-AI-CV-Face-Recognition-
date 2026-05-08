import os
import cv2
import time
import queue
import warnings
import threading
import numpy as np

from deepface import DeepFace
from datetime import datetime
from collections import deque, Counter

# =========================================================
# ENVIRONMENT SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DEEPFACE_HOME"] = "models"

warnings.filterwarnings("ignore")

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# =========================================================
# APP SETTINGS
# =========================================================

WINDOW_NAME = "AI Surveillance Pro"

# Window size
WIDTH = 1280
HEIGHT = 720

# FPS limiter
FPS_LIMIT = 60

# Analyze every X frames
# Lower = faster updates but more CPU usage
ANALYZE_EVERY = 5

# AI input resolution
ANALYZE_WIDTH = 320
ANALYZE_HEIGHT = 180

# Minimum face size
MIN_FACE_SIZE = 100

# Recognition threshold
SIMILARITY_THRESHOLD = 0.42

# How long to keep recognition cache
RECOGNITION_COOLDOWN = 4.0

# How long to freeze age/gender values
FACE_DATA_HOLD_TIME = 5.0

# Screenshot directory
SCREENSHOTS_DIR = "screenshots"

# Known faces directory
KNOWN_FACES_DIR = "known_faces"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# =========================================================
# COLORS
# =========================================================

CYAN = (255, 255, 0)
GREEN = (0, 255, 120)
RED = (0, 80, 255)
ORANGE = (0, 170, 255)

WHITE = (245, 245, 245)
GRAY = (170, 170, 170)

BLACK = (0, 0, 0)
DARK = (22, 22, 22)

# =========================================================
# FONT
# =========================================================

FONT = cv2.FONT_HERSHEY_DUPLEX

# =========================================================
# EMOJIS
# =========================================================

EMOJI = {
    "happy": ":)",
    "sad": ":'(",
    "neutral": ":|",
    "angry": ">:(",
    "surprise": ":O",
    "fear": ":/",
    "disgust": "XD"
}

# =========================================================
# GLOBAL VARIABLES
# =========================================================

running = True

frame_counter = 0

latest_results = []

save_flash = 0

fps_queue = deque(maxlen=30)

# Thread queue
frame_queue = queue.Queue(maxsize=1)

# Emotion smoothing
emotion_history = {}

# Recognition memory
face_memory = {}

# Known face embeddings
known_faces = []

# =========================================================
# LOAD KNOWN FACES
# =========================================================

print("[INFO] Loading known faces...")

for file in os.listdir(KNOWN_FACES_DIR):

    path = os.path.join(KNOWN_FACES_DIR, file)

    if not os.path.isfile(path):
        continue

    try:

        embedding = DeepFace.represent(
            img_path=path,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False
        )[0]["embedding"]

        known_faces.append({

            "name": os.path.splitext(file)[0],
            "embedding": np.array(embedding)

        })

        print(f"[LOADED] {file}")

    except Exception as e:

        print("[LOAD ERROR]", e)

print(f"[INFO] Loaded {len(known_faces)} faces")

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Lower camera latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print("[ERROR] Camera not opened")
    exit()

# =========================================================
# HELPERS
# =========================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def valid_face(w, h):

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    ratio = h / w

    return 0.75 <= ratio <= 1.55


def smooth_emotion(face_id, emotion):

    if face_id not in emotion_history:

        emotion_history[face_id] = deque(maxlen=6)

    emotion_history[face_id].append(emotion)

    return Counter(
        emotion_history[face_id]
    ).most_common(1)[0][0]

# =========================================================
# FACE RECOGNITION
# =========================================================

def recognize_face(face_crop):

    try:

        embedding = DeepFace.represent(
            face_crop,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False
        )[0]["embedding"]

        embedding = np.array(embedding)

        best_score = -1
        best_name = "Unknown"

        for person in known_faces:

            score = cosine_similarity(
                embedding,
                person["embedding"]
            )

            if score > best_score:

                best_score = score
                best_name = person["name"]

        if best_score >= SIMILARITY_THRESHOLD:
            return best_name

        return "Unknown"

    except:
        return "Unknown"

# =========================================================
# AI WORKER
# =========================================================

def analyze_worker():

    global latest_results

    while running:

        try:

            frame = frame_queue.get(timeout=1)

        except:
            continue

        try:

            # Resize frame for AI
            small = cv2.resize(
                frame,
                (ANALYZE_WIDTH, ANALYZE_HEIGHT)
            )

            # =================================================
            # AI ANALYSIS
            # =================================================

            results = DeepFace.analyze(
                small,
                actions=["age", "gender", "emotion"],
                detector_backend="yunet",
                enforce_detection=False,
                silent=True
            )

            if not isinstance(results, list):
                results = [results]

            sx = frame.shape[1] / ANALYZE_WIDTH
            sy = frame.shape[0] / ANALYZE_HEIGHT

            faces = []

            for face in results:

                region = face["region"]

                x = int(region["x"] * sx)
                y = int(region["y"] * sy)
                w = int(region["w"] * sx)
                h = int(region["h"] * sy)

                if not valid_face(w, h):
                    continue

                # Prevent overflow
                x = max(0, x)
                y = max(0, y)

                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                crop = frame[y:y+h, x:x+w]

                if crop.size == 0:
                    continue

                # =================================================
                # FACE TRACKING ID
                # =================================================

                face_id = f"{x//35}_{y//35}"

                current_time = time.time()

                # =================================================
                # RECOGNITION CACHE
                # =================================================

                if face_id in face_memory:

                    memory = face_memory[face_id]

                    # Keep stable data
                    if current_time - memory["last_update"] < FACE_DATA_HOLD_TIME:

                        age = memory["age"]
                        gender = memory["gender"]
                        name = memory["name"]

                    else:

                        age = int(face.get("age", 0))
                        gender = face.get(
                            "dominant_gender",
                            "Unknown"
                        )

                        name = recognize_face(crop)

                        memory["age"] = age
                        memory["gender"] = gender
                        memory["name"] = name
                        memory["last_update"] = current_time

                else:

                    age = int(face.get("age", 0))

                    gender = face.get(
                        "dominant_gender",
                        "Unknown"
                    )

                    name = recognize_face(crop)

                    face_memory[face_id] = {

                        "age": age,
                        "gender": gender,
                        "name": name,
                        "last_update": current_time

                    }

                # =================================================
                # EMOTION SMOOTHING
                # =================================================

                emotion = face["dominant_emotion"].lower()

                emotion = smooth_emotion(
                    face_id,
                    emotion
                )

                # =================================================
                # STORE RESULT
                # =================================================

                faces.append({

                    "name": name,

                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,

                    "age": age,
                    "gender": gender,

                    "emotion": emotion,
                    "emoji": EMOJI.get(emotion, ":|")

                })

            latest_results = faces

        except Exception as e:

            print("[AI ERROR]", e)

# =========================================================
# DRAW HELPERS
# =========================================================

def draw_text(
    img,
    text,
    pos,
    color=WHITE,
    scale=0.5,
    thickness=1
):

    x, y = pos

    cv2.putText(
        img,
        text,
        (x, y),
        FONT,
        scale,
        BLACK,
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (x, y),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def panel(img, x1, y1, x2, y2):

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        DARK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.78,
        img,
        0.22,
        0,
        img
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (55, 55, 55),
        1
    )


def draw_face_box(img, x, y, w, h):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        CYAN,
        2
    )

# =========================================================
# DRAW HUD
# =========================================================

def draw_hud(img, fps, people):

    # Top left
    panel(img, 15, 15, 240, 80)

    draw_text(
        img,
        "AI SURVEILLANCE",
        (28, 38),
        CYAN,
        0.62,
        2
    )

    draw_text(
        img,
        f"FPS : {fps:.1f}",
        (28, 65),
        GREEN,
        0.46,
        1
    )

    draw_text(
        img,
        f"PEOPLE : {people}",
        (125, 65),
        WHITE,
        0.46,
        1
    )

    # Controls
    panel(img, 15, 95, 170, 145)

    draw_text(
        img,
        "[Q] EXIT",
        (28, 125),
        RED,
        0.45,
        2
    )

    draw_text(
        img,
        "[S] SHOT",
        (105, 125),
        GREEN,
        0.45,
        2
    )

    # Time
    panel(img, WIDTH - 120, 15, WIDTH - 15, 48)

    draw_text(
        img,
        datetime.now().strftime("%H:%M:%S"),
        (WIDTH - 107, 36),
        WHITE,
        0.42,
        2
    )

# =========================================================
# START AI THREAD
# =========================================================

threading.Thread(
    target=analyze_worker,
    daemon=True
).start()

# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.resizeWindow(
    WINDOW_NAME,
    WIDTH,
    HEIGHT
)

print("[INFO] System Started")

# =========================================================
# MAIN LOOP
# =========================================================

while running:

    start = time.time()

    ok, frame = cap.read()

    if not ok:
        break

    output = frame.copy()

    frame_counter += 1

    # =====================================================
    # FPS
    # =====================================================

    fps_queue.append(time.time())

    fps = 0

    if len(fps_queue) > 1:

        fps = (
            len(fps_queue) - 1
        ) / (
            fps_queue[-1] - fps_queue[0]
        )

    # =====================================================
    # SEND FRAME TO AI
    # =====================================================

    if frame_counter % ANALYZE_EVERY == 0:

        if frame_queue.empty():

            frame_queue.put(frame.copy())

    # =====================================================
    # DRAW RESULTS
    # =====================================================

    for face in latest_results:

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        draw_face_box(
            output,
            x,
            y,
            w,
            h
        )

        # =================================================
        # INFO PANEL POSITION
        # =================================================

        info_y = y + h + 8

        if info_y + 100 > HEIGHT:
            info_y = y - 105

        # =================================================
        # INFO PANEL
        # =================================================

        panel(
            output,
            x,
            info_y,
            x + 185,
            info_y + 95
        )

        # Name
        draw_text(
            output,
            face["name"],
            (x + 10, info_y + 22),
            WHITE,
            0.52,
            2
        )

        # Gender + Age
        draw_text(
            output,
            f"{face['gender']} | {face['age']}Y",
            (x + 10, info_y + 48),
            GREEN,
            0.44,
            1
        )

        # Emotion
        draw_text(
            output,
            f"{face['emotion'].upper()} {face['emoji']}",
            (x + 10, info_y + 74),
            ORANGE,
            0.44,
            1
        )

    # =====================================================
    # SCREENSHOT FLASH
    # =====================================================

    if save_flash > 0:

        flash = np.full_like(output, 255)

        cv2.addWeighted(
            flash,
            0.10,
            output,
            0.90,
            0,
            output
        )

        save_flash -= 1

    # =====================================================
    # DRAW HUD
    # =====================================================

    draw_hud(
        output,
        fps,
        len(latest_results)
    )

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(WINDOW_NAME, output)

    key = cv2.waitKey(1) & 0xFF

    # =====================================================
    # EXIT
    # =====================================================

    if key == ord("q"):

        running = False
        break

    # =====================================================
    # SCREENSHOT
    # =====================================================

    elif key == ord("s"):

        try:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = f"shot_{timestamp}.png"

            path = os.path.join(
                SCREENSHOTS_DIR,
                filename
            )

            success = cv2.imwrite(path, output)

            if success:

                save_flash = 2

                print(f"[SAVED] {path}")

        except Exception as e:

            print("[SCREENSHOT ERROR]", e)

    # =====================================================
    # FPS LIMITER
    # =====================================================

    elapsed = time.time() - start

    delay = max(1 / FPS_LIMIT - elapsed, 0)

    time.sleep(delay)

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("[INFO] System Closed")