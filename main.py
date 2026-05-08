"""
Real-Time Age & Gender Detection — Enhanced Version
====================================================
Controls:
    Q        → Quit
    S        → Save screenshot  (saved in /screenshots folder)
    + / =    → Raise confidence threshold by 0.05
    -        → Lower confidence threshold by 0.05
"""

import cv2
import os
import time
from collections import defaultdict, deque, Counter
from datetime import datetime

# ─────────────────────────────────────────────
#  Model paths  (keep your models/ folder as-is)
# ─────────────────────────────────────────────
FACE_PROTO   = "models/opencv_face_detector.pbtxt"
FACE_MODEL   = "models/opencv_face_detector_uint8.pb"
AGE_PROTO    = "models/age_deploy.prototxt"
AGE_MODEL    = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

MODEL_MEAN   = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST     = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST  = ['Male', 'Female']

# ─────────────────────────────────────────────
#  Load models  (with clear error messages)
# ─────────────────────────────────────────────
try:
    face_net   = cv2.dnn.readNet(FACE_MODEL,   FACE_PROTO)
    age_net    = cv2.dnn.readNet(AGE_MODEL,    AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("[OK] All 3 models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    print("       Make sure the 'models/' folder is in the same directory as this script.")
    exit(1)

# ─────────────────────────────────────────────
#  Open camera
# ─────────────────────────────────────────────
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("[ERROR] Cannot open camera.")
    print("       Check that your webcam is connected and not used by another app.")
    exit(1)

# ─────────────────────────────────────────────
#  Config  (tweak these freely)
# ─────────────────────────────────────────────
PADDING           = 20      # pixels added around face before age/gender inference
CONFIDENCE_THRESH = 0.7     # initial face-detection confidence threshold
SMOOTHING_WINDOW  = 10      # how many past frames to average predictions over
FRAME_SKIP        = 2       # run heavy detection only every N frames (speeds things up)
GRID_CELL         = 80      # pixel size of grid cell used to "track" each face by position

SCREENSHOTS_DIR   = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# Colors  (BGR)
COL_MALE   = (210, 140, 40)    # warm blue / teal
COL_FEMALE = (180, 50, 210)    # purple-pink
COL_GREEN  = (50,  210, 100)
COL_YELLOW = (30,  210, 210)
COL_GRAY   = (180, 180, 180)
COL_DARK   = (18,  18,  18)

# ─────────────────────────────────────────────
#  Per-face history store  (for smoothing)
# ─────────────────────────────────────────────
face_history = defaultdict(lambda: {
    "gender":      deque(maxlen=SMOOTHING_WINDOW),
    "age":         deque(maxlen=SMOOTHING_WINDOW),
    "gender_conf": deque(maxlen=SMOOTHING_WINDOW),
})

last_boxes = []    # cached bounding boxes from the last detection frame
frame_idx  = 0
fps_ring   = deque(maxlen=30)   # rolling timestamp window for FPS


# ═════════════════════════════════════════════
#  Helper functions
# ═════════════════════════════════════════════

def make_face_key(x1, y1, x2, y2):
    """
    Map a face's centre position to a coarse grid cell.
    Faces that stay roughly in the same area share the same key,
    so we can accumulate predictions across frames without complex tracking.
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx // GRID_CELL, cy // GRID_CELL)


def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, arm=22):
    """
    Draw stylish corner-only bounding box instead of a full rectangle.
    Much less cluttered when multiple faces are on screen.
    """
    corners = [
        (x1, y1,  1,  1),   # top-left
        (x2, y1, -1,  1),   # top-right
        (x1, y2,  1, -1),   # bottom-left
        (x2, y2, -1, -1),   # bottom-right
    ]
    for px, py, dx, dy in corners:
        cv2.line(img, (px, py), (px + dx * arm, py),        color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px,            py + dy * arm), color, thickness, cv2.LINE_AA)


def draw_label(img, text, x, y, color):
    """
    Draw a text label with a semi-transparent dark background and a
    coloured top-border accent line.  Automatically stays inside the frame.
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thick = 1
    pad   = 5

    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    H, W = img.shape[:2]

    # Text anchor (clamped so label never goes off screen)
    fx = max(pad, min(x, W - tw - pad * 2))
    fy = max(th + pad * 2, min(y, H - pad))

    # Background rectangle coordinates
    rx0 = max(0, fx - pad)
    ry0 = max(0, fy - th - pad)
    rx1 = min(W, fx + tw + pad)
    ry1 = min(H, fy + pad)

    if rx1 <= rx0 or ry1 <= ry0:
        return   # nothing to draw (edge case)

    # Blend a dark rect onto just the label region
    roi = img[ry0:ry1, rx0:rx1]
    dark = roi.copy()
    dark[:] = COL_DARK
    cv2.addWeighted(dark, 0.65, roi, 0.35, 0, roi)
    img[ry0:ry1, rx0:rx1] = roi

    # Coloured top border accent
    cv2.line(img, (rx0, ry0), (rx1, ry0), color, 2, cv2.LINE_AA)

    # Text
    cv2.putText(img, text, (fx, fy), font, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, fps_val, n_faces):
    """
    Draw the info panel in the top-left corner:
    FPS, face count, current threshold, keyboard hints.
    """
    H, W = img.shape[:2]
    px0, py0, px1, py1 = 8, 8, 222, 102

    # semi-transparent dark background for panel
    roi = img[py0:py1, px0:px1]
    dark = roi.copy()
    dark[:] = COL_DARK
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
    img[py0:py1, px0:px1] = roi

    # Panel border
    cv2.rectangle(img, (px0, py0), (px1, py1), (70, 70, 70), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"FPS :  {fps_val:5.1f}",          (18, 30), font, 0.54, COL_GREEN,  1, cv2.LINE_AA)
    cv2.putText(img, f"Faces: {n_faces}",                (18, 54), font, 0.54, COL_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(img, f"Thresh: {CONFIDENCE_THRESH:.2f}", (18, 76), font, 0.47, COL_GRAY,   1, cv2.LINE_AA)
    cv2.putText(img, "[Q]uit [S]ave  [+/-] thresh",      (18, 96), font, 0.33, (110,110,110), 1, cv2.LINE_AA)


def smoothed_prediction(key):
    """
    Return the majority-vote gender, most-common age, and average confidence
    from the stored history for a given face key.
    Returns (None, None, 0) if no history yet.
    """
    h = face_history[key]
    if not h["gender"]:
        return None, None, 0.0

    gender      = Counter(h["gender"]).most_common(1)[0][0]
    age         = Counter(h["age"]).most_common(1)[0][0]
    gender_conf = sum(h["gender_conf"]) / len(h["gender_conf"])
    return gender, age, gender_conf


# ═════════════════════════════════════════════
#  Main loop
# ═════════════════════════════════════════════
print("\n[INFO] Detection running.")
print("       Q → quit   |   S → screenshot   |   + / - → adjust threshold\n")

while True:
    ok, frame = video.read()
    if not ok:
        print("[WARN] Failed to read frame from camera — stopping.")
        break

    frame_idx += 1
    fps_ring.append(time.time())
    fps_val = (
        (len(fps_ring) - 1) / (fps_ring[-1] - fps_ring[0])
        if len(fps_ring) > 1 else 0.0
    )

    H, W = frame.shape[:2]
    out  = frame.copy()   # we draw on this copy, never on the raw frame

    # ── Detection pass (only every FRAME_SKIP frames) ──────────────
    if frame_idx % FRAME_SKIP == 0:

        # 1) Face detection
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False
        )
        face_net.setInput(blob)
        dets = face_net.forward()

        new_boxes = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < CONFIDENCE_THRESH:
                continue

            x1 = max(0,     int(dets[0, 0, i, 3] * W))
            y1 = max(0,     int(dets[0, 0, i, 4] * H))
            x2 = min(W - 1, int(dets[0, 0, i, 5] * W))
            y2 = min(H - 1, int(dets[0, 0, i, 6] * H))

            fk = make_face_key(x1, y1, x2, y2)

            # 2) Crop face region (with padding, clamped to frame)
            roi = frame[
                max(0, y1 - PADDING) : min(H, y2 + PADDING),
                max(0, x1 - PADDING) : min(W, x2 + PADDING),
            ]
            if roi.size == 0:
                continue

            # 3) Age & gender inference
            face_blob = cv2.dnn.blobFromImage(
                roi, 1.0, (227, 227), MODEL_MEAN, swapRB=False
            )

            gender_net.setInput(face_blob)
            gp = gender_net.forward()
            gi = int(gp[0].argmax())
            face_history[fk]["gender"].append(GENDER_LIST[gi])
            face_history[fk]["gender_conf"].append(float(gp[0][gi]))

            age_net.setInput(face_blob)
            ap = age_net.forward()
            ai = int(ap[0].argmax())
            face_history[fk]["age"].append(AGE_LIST[ai])

            new_boxes.append((x1, y1, x2, y2, fk))

        last_boxes = new_boxes   # cache for skipped frames

    # ── Draw results ────────────────────────────────────────────────
    for (x1, y1, x2, y2, fk) in last_boxes:
        gender, age, g_conf = smoothed_prediction(fk)
        if gender is None:
            continue

        color = COL_MALE if gender == "Male" else COL_FEMALE
        tag   = "M" if gender == "Male" else "F"

        draw_corner_box(out, x1, y1, x2, y2, color)

        label   = f"{tag} {gender} {g_conf * 100:.0f}% | Age {age}"
        label_y = (y1 - 6) if y1 > 45 else (y2 + 36)
        draw_label(out, label, x1, label_y, color)

    # ── HUD overlay ─────────────────────────────────────────────────
    draw_hud(out, fps_val, len(last_boxes))

    cv2.imshow("Age & Gender Detection — Enhanced", out)

    # ── Keyboard controls ────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[INFO] Quit.")
        break

    elif key == ord("s"):
        path = os.path.join(
            SCREENSHOTS_DIR,
            f"shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        cv2.imwrite(path, out)
        print(f"[SAVED] {path}")

    elif key in (ord("+"), ord("=")):
        CONFIDENCE_THRESH = min(0.99, round(CONFIDENCE_THRESH + 0.05, 2))
        print(f"[THRESH] Raised to {CONFIDENCE_THRESH:.2f}")

    elif key == ord("-"):
        CONFIDENCE_THRESH = max(0.10, round(CONFIDENCE_THRESH - 0.05, 2))
        print(f"[THRESH] Lowered to {CONFIDENCE_THRESH:.2f}")

# ─────────────────────────────────────────────
video.release()
cv2.destroyAllWindows()
print("[DONE] Camera released.")