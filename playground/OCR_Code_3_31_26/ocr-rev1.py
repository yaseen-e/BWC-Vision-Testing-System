import cv2
import numpy as np
import pytesseract
from picamera2 import Picamera2
import time

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
        [0, maxHeight - 1]], dtype="float32")
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
time.sleep(2)  # Give AF time to settle

# ----------------------------
# Capture one frame
# ----------------------------
frame = picam2.capture_array()
original = frame.copy()

# Convert to grayscale and blur
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection and contours
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

if display_contour is not None:
    # Perspective correction
    warped = four_point_transform(frame, display_contour.reshape(4, 2))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    _, thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define ROI for OCR
    h, w = thresh.shape
    roi = thresh[int(h * 0.1):int(h * 0.42), int(w * 0.3):int(w * 0.63)]

    # OCR
    config = "--psm 7"
    text = pytesseract.image_to_string(roi, config=config)

    # Save images for reference
    cv2.imwrite("original.jpg", original)
    cv2.imwrite("warped.jpg", warped)
    cv2.imwrite("roi.jpg", roi)

    print("OCR Result:", text.strip())

else:
    print("No display contour detected.")
    cv2.imwrite("original.jpg", original)