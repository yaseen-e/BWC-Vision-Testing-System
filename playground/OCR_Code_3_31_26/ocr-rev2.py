import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2
import time

# ----------------------------
# Helper: Order points
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
# Perspective transform
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
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# ----------------------------
# Initialize Camera
# ----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)

picam2.start()
picam2.set_controls({"AfMode": 2})  # Continuous autofocus
time.sleep(2)

# ----------------------------
# Capture Frame
# ----------------------------
frame = picam2.capture_array()
original = frame.copy()

# ----------------------------
# BLUE DISPLAY SEGMENTATION
# ----------------------------
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

mask = cv2.inRange(hsv, lower_white, upper_white)

# Clean mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.medianBlur(mask, 5)

cv2.imwrite("mask.jpg", mask)

# ----------------------------
# DISTANCE-INVARIANT DETECTION
# ----------------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

display_contour = None
best_score = 0

image_area = frame.shape[0] * frame.shape[1]

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 1000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    print(f"Area: {area:.0f}, Width: {w}, Height: {h}, Aspect: {aspect_ratio:.2f}")

    rect_area = w * h
    extent = area / rect_area

    relative_area = area / image_area

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if (
        2.0 < aspect_ratio < 6.0 and
        extent > 0.5 and
        relative_area > 0.01 and
        4 <= len(approx) <= 6
    ):
        if area > best_score:
            display_contour = approx
            best_score = area

# ----------------------------
# PROCESS DISPLAY
# ----------------------------
if display_contour is not None:

    warped = four_point_transform(frame, display_contour.reshape(-1, 2))

    # OCR preprocessing
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    roi_mask = cv2.inRange(
        warped_hsv,
        np.array([0, 0, 200]),
        np.array([180, 40, 255])
    )

    roi = cv2.bitwise_not(roi_mask)

    # Optional crop (tune if needed)
    h, w = roi.shape
    roi = roi[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # OCR
    config = "--psm 7 -c tessedit_char_whitelist=0123456789.-"
    text = pytesseract.image_to_string(roi, config=config)

    # Save debug images
    cv2.imwrite("original2.jpg", original)
    cv2.imwrite("warped2.jpg", warped)
    cv2.imwrite("roi2.jpg", roi)

    print("OCR Result:", text.strip())

else:
    print("No display detected.")
    cv2.imwrite("original.jpg", original)