import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2
import time
from difflib import get_close_matches

# ----------------------------
# Helper: Order corner points
# ----------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ----------------------------
# Helper: Perspective transform
# ----------------------------
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# ----------------------------
# Initialize Pi Camera
# ----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()
picam2.set_controls({"AfMode": 2})  # Continuous autofocus
time.sleep(2)  # Give autofocus time to settle

# ----------------------------
# Capture one frame
# ----------------------------
frame = picam2.capture_array()
original = frame.copy()

# ----------------------------
# Convert to HSV for color filtering (orange display)
# ----------------------------
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_orange = np.array([5, 150, 150])
upper_orange = np.array([25, 255, 255])

mask = cv2.inRange(hsv, lower_orange, upper_orange)

# ----------------------------
# Morphology to clean mask
# ----------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# ----------------------------
# Find contours
# ----------------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

display_contour = None
max_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 3000:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            display_contour = approx
            max_area = area

# ----------------------------
# Process detected display
# ----------------------------
if display_contour is not None:
    warped = four_point_transform(frame, display_contour.reshape(4, 2))

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #warped_gray = cv2.GaussianBlur(warped_gray, (3, 3), 0)

    # Resize before thresholding
    warped_gray = cv2.resize(
        warped_gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Otsu threshold
    _, thresh = cv2.threshold(
        warped_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ----------------------------
    # ROI for HYBRID text
    # ----------------------------
    h, w = thresh.shape

    roi = thresh[
        int(h * 0.00):int(h * 0.14),
        int(w * 0.2):int(w * 0.8)
    ]

    # Invert so text is black on white
    roi_inv = cv2.bitwise_not(roi)

    # Resize for OCR
    roi_resized = cv2.resize(
        roi_inv,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # OCR config
    config_tess = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ:"
    raw_text = pytesseract.image_to_string(roi_resized, config=config_tess)

    # Clean OCR output
    text = raw_text.strip().upper()

    # ----------------------------
    # Match OCR result to known words
    # ----------------------------
    possible_modes = [
        "MODE: HYBRID",
        "MODE: HYBRID PLUS",
        "MODE: HEAT PUMP",
        "MODE: ELECTRIC",
        "MODE: VACATION"
    ]

    match = get_close_matches(text, possible_modes, n=1, cutoff=0.5)

    if match:
        final_mode = match[0]
    else:
        final_mode = "UNKNOWN"

    # ----------------------------
    # Save debug images
    # ----------------------------
    cv2.imwrite("original3.jpg", original)
    cv2.imwrite("mask3.jpg", mask)
    cv2.imwrite("warped3.jpg", warped)
    cv2.imwrite("thresh3.jpg", thresh)
    cv2.imwrite("roi3.jpg", roi_resized)

    print("Raw OCR Result:", text)
    print("Detected Mode:", final_mode)

else:
    print("No display contour detected.")
    cv2.imwrite("original3.jpg", original)
    cv2.imwrite("mask3.jpg", mask)