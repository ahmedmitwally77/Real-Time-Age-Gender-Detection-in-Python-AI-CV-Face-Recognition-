import os
import cv2
import time
import threading
import warnings
import numpy as np

from deepface import DeepFace
from datetime import datetime
from collections import deque

# =========================================================
# ENVIRONMENT
# =========================================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["DEEPFACE_HOME"] = "models"

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS
# =========================================================

WINDOW_NAME = "AI Surveillance Pro"

# شاشة أكبر ومحترمة
WIDTH = 1380
HEIGHT = 780

# أسرع
FPS_LIMIT = 90
ANALYZE_EVERY = 22

# تقليل استهلاك التحليل
ANALYZE_WIDTH = 480
ANALYZE_HEIGHT = 270

SCREENSHOTS_DIR = "screenshots"
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

WHITE = (255, 255, 255)
GRAY = (170, 170, 170)

BLACK = (0, 0, 0)
DARK = (15, 15, 15)

# =========================================================
# EMOJIS
# =========================================================

EMOJI = {
    "happy": ":)",
    "sad": ":(",
    "neutral": ":|",
    "angry": ">:(",
    "surprise": ":O",
    "fear": ":/",
    "disgust": "XD"
}

# =========================================================
# GLOBALS
# =========================================================

latest_results = []
analyzing = False
running = True

frame_counter = 0

fps_queue = deque(maxlen=30)

known_faces = []

save_flash = 0

# =========================================================
# LOAD KNOWN FACES
# =========================================================

print("[INFO] Loading known faces...")

for file in os.listdir(KNOWN_FACES_DIR):

    path = os.path.join(KNOWN_FACES_DIR, file)

    if os.path.isfile(path):

        name = os.path.splitext(file)[0]

        known_faces.append({
            "name": name,
            "path": path
        })

print(f"[INFO] Loaded {len(known_faces)} faces")

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():

    print("[ERROR] Camera not opened")
    exit()

# =========================================================
# FACE FILTER
# =========================================================

def valid_face(x, y, w, h):

    if w < 140 or h < 140:
        return False

    ratio = h / w

    if ratio < 0.9 or ratio > 1.5:
        return False

    return True

# =========================================================
# REMOVE DUPLICATES
# =========================================================

def duplicate(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center1 = (x1 + w1//2, y1 + h1//2)
    center2 = (x2 + w2//2, y2 + h2//2)

    distance = np.sqrt(
        (center1[0] - center2[0])**2 +
        (center1[1] - center2[1])**2
    )

    return distance < 110

# =========================================================
# FACE RECOGNITION
# =========================================================

def recognize_face(face):

    try:

        for person in known_faces:

            result = DeepFace.verify(
                face,
                person["path"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if result["verified"]:
                return person["name"]

        return "Unknown"

    except:
        return "Unknown"

# =========================================================
# ANALYZE FRAME
# =========================================================

def analyze_frame(frame):

    global latest_results
    global analyzing

    try:

        # تصغير للتحسين السريع
        small = cv2.resize(
            frame,
            (ANALYZE_WIDTH, ANALYZE_HEIGHT)
        )

        results = DeepFace.analyze(
            small,
            actions=['age', 'gender', 'emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True
        )

        if not isinstance(results, list):
            results = [results]

        sx = frame.shape[1] / ANALYZE_WIDTH
        sy = frame.shape[0] / ANALYZE_HEIGHT

        faces = []
        added = []

        for face in results:

            region = face["region"]

            x = int(region["x"] * sx)
            y = int(region["y"] * sy)
            w = int(region["w"] * sx)
            h = int(region["h"] * sy)

            if not valid_face(x, y, w, h):
                continue

            skip = False

            for old in added:

                if duplicate((x, y, w, h), old):
                    skip = True
                    break

            if skip:
                continue

            added.append((x, y, w, h))

            crop = frame[y:y+h, x:x+w]

            name = "Unknown"

            # تسريع التعرف
            if crop.size > 0 and len(known_faces) > 0:

                try:
                    name = recognize_face(crop)
                except:
                    name = "Unknown"

            emotion = face["dominant_emotion"].lower()

            emoji = EMOJI.get(emotion, ":|")

            faces.append({

                "name": name,

                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "gender": face["dominant_gender"],
                "age": int(face["age"]),

                "emotion": emotion.upper(),
                "emoji": emoji

            })

        latest_results = faces

    except Exception as e:

        print("[ERROR]", e)

    analyzing = False

# =========================================================
# TEXT
# =========================================================

def draw_text(
    img,
    text,
    pos,
    color=WHITE,
    scale=0.8,
    thickness=2
):

    x, y = pos

    cv2.putText(
        img,
        text,
        (x+3, y+3),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        BLACK,
        thickness + 4,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

# =========================================================
# PANELS
# =========================================================

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
        0.82,
        img,
        0.18,
        0,
        img
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        CYAN,
        2
    )

# =========================================================
# FACE BOX
# =========================================================

def draw_face_box(img, x, y, w, h):

    color = CYAN

    line = 35
    t = 3

    cv2.line(img, (x, y), (x+line, y), color, t)
    cv2.line(img, (x, y), (x, y+line), color, t)

    cv2.line(img, (x+w, y), (x+w-line, y), color, t)
    cv2.line(img, (x+w, y), (x+w, y+line), color, t)

    cv2.line(img, (x, y+h), (x+line, y+h), color, t)
    cv2.line(img, (x, y+h), (x, y+h-line), color, t)

    cv2.line(img, (x+w, y+h), (x+w-line, y+h), color, t)
    cv2.line(img, (x+w, y+h), (x+w, y+h-line), color, t)

# =========================================================
# HUD
# =========================================================

def draw_hud(img, fps, people):

    # تكبير البوكس العلوي
    panel(img, 20, 20, 520, 185)

    draw_text(
        img,
        "AI SURVEILLANCE PRO",
        (45, 85),
        CYAN,
        1.45,
        3
    )

    draw_text(
        img,
        "REAL-TIME AI DETECTION SYSTEM",
        (45, 135),
        GRAY,
        0.75,
        2
    )

    draw_text(
        img,
        f"FPS : {fps:.1f}",
        (45, 178),
        GREEN,
        1.0,
        3
    )

    # CONTROLS
    panel(img, 20, 220, 360, 380)

    draw_text(
        img,
        "[ Q ] EXIT",
        (45, 295),
        RED,
        1.05,
        3
    )

    draw_text(
        img,
        "[ S ] SCREENSHOT",
        (45, 360),
        GREEN,
        0.95,
        3
    )

    # RIGHT PANEL
    panel(img, WIDTH - 320, 20, WIDTH - 20, 240)

    draw_text(
        img,
        "DETECTION LOG",
        (WIDTH - 285, 85),
        CYAN,
        0.95,
        3
    )

    draw_text(
        img,
        f"PEOPLE : {people}",
        (WIDTH - 285, 150),
        GREEN,
        1.0,
        3
    )

    draw_text(
        img,
        datetime.now().strftime("%H:%M:%S"),
        (WIDTH - 285, 210),
        WHITE,
        0.9,
        2
    )

# =========================================================
# BOTTOM BAR
# =========================================================

def bottom_bar(img):

    panel(
        img,
        0,
        HEIGHT - 55,
        WIDTH,
        HEIGHT
    )

    draw_text(
        img,
        "STATUS : RUNNING",
        (30, HEIGHT - 18),
        GREEN,
        0.72,
        2
    )

    draw_text(
        img,
        "AI MODEL : DEEPFACE",
        (530, HEIGHT - 18),
        CYAN,
        0.72,
        2
    )

    draw_text(
        img,
        "PRESS Q TO EXIT",
        (1090, HEIGHT - 18),
        RED,
        0.72,
        2
    )

# =========================================================
# MAIN LOOP
# =========================================================

print("[INFO] AI System Started")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# يخليها Full HD ومحترمة
cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

while running:

    start = time.time()

    ok, frame = cap.read()

    if not ok:
        break

    frame_counter += 1

    output = frame.copy()

    fps_queue.append(time.time())

    fps = 0

    if len(fps_queue) > 1:

        fps = (
            len(fps_queue) - 1
        ) / (
            fps_queue[-1] - fps_queue[0]
        )

    # =====================================================
    # ANALYZE
    # =====================================================

    if frame_counter % ANALYZE_EVERY == 0:

        if not analyzing:

            analyzing = True

            threading.Thread(
                target=analyze_frame,
                args=(frame.copy(),),
                daemon=True
            ).start()

    # =====================================================
    # DRAW FACES
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

        # زودنا ارتفاع البوكس تحت
        info_y1 = y + h + 10
        info_y2 = y + h + 155

        # منع الخروج خارج الشاشة
        if info_y2 > HEIGHT - 70:

            info_y1 = y - 165
            info_y2 = y - 10

        panel(
            output,
            x,
            info_y1,
            x + 270,
            info_y2
        )

        draw_text(
            output,
            face["name"],
            (x + 25, y - 15),
            WHITE,
            0.9,
            3
        )

        draw_text(
            output,
            f"{face['gender']}",
            (x + 25, info_y1 + 45),
            CYAN,
            0.82,
            2
        )

        draw_text(
            output,
            f"{face['age']} YEARS",
            (x + 25, info_y1 + 90),
            GREEN,
            0.82,
            2
        )

        draw_text(
            output,
            f"{face['emotion']} {face['emoji']}",
            (x + 25, info_y1 + 135),
            ORANGE,
            0.78,
            2
        )

    # =====================================================
    # HUD
    # =====================================================

    draw_hud(
        output,
        fps,
        len(latest_results)
    )

    bottom_bar(output)

    # =====================================================
    # FLASH EFFECT
    # =====================================================

    if save_flash > 0:

        flash = np.full_like(output, 255)

        cv2.addWeighted(
            flash,
            0.20,
            output,
            0.80,
            0,
            output
        )

        save_flash -= 1

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(WINDOW_NAME, output)

    key = cv2.waitKey(1) & 0xFF

    # =====================================================
    # EXIT
    # =====================================================

    if key == ord('q'):

        running = False
        break

    # =====================================================
    # SCREENSHOT
    # =====================================================

    elif key == ord('s'):

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

            else:

                print("[ERROR] Screenshot failed")

        except Exception as e:

            print("[ERROR]", e)

    # =====================================================
    # FPS LIMIT
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