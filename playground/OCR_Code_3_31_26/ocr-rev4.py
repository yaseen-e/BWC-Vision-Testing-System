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
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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

config = picam2.create_preview_configuration(
    main={"size": (1280, 720)}
)
picam2.configure(config)

picam2.start()
picam2.set_controls({"AfMode": 2})
time.sleep(2)

# ----------------------------
# Capture frame
# ----------------------------
frame = picam2.capture_array()
original = frame.copy()

# ----------------------------
# Convert to grayscale
# ----------------------------
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# ----------------------------
# Detect dark screen regions
# ----------------------------
thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.dilate(thresh, kernel, iterations=1)

# ----------------------------
# Find contours
# ----------------------------
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

display_contour = None
max_area = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 15000:
        continue

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    box_w = rect[1][0]
    box_h = rect[1][1]

    if box_w == 0 or box_h == 0:
        continue

    aspect_ratio = max(box_w, box_h) / min(box_w, box_h)

    # Thermostat screen aspect ratio
    if 1.3 < aspect_ratio < 2.2:
        rect_area = box_w * box_h

        if rect_area > max_area:
            max_area = rect_area
            display_contour = box

# ----------------------------
# Process display if found
# ----------------------------
if display_contour is not None:

    cv2.drawContours(original, [display_contour], 0, (0, 255, 0), 3)

    # Perspective correction
    warped = four_point_transform(
        frame,
        display_contour.astype("float32")
    )

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.GaussianBlur(warped_gray, (3, 3), 0)

    # OCR threshold
    warped_thresh = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    # ----------------------------
    # Temperature ROI only
    # ----------------------------
    wh, ww = warped_thresh.shape

    temp_roi = warped_thresh[
        int(wh * 0.15):int(wh * 0.55),
        int(ww * 0.20):int(ww * 0.75)
    ]

    # Invert for OCR
    temp_roi = cv2.bitwise_not(temp_roi)

    # Resize for OCR
    temp_roi = cv2.resize(
        temp_roi,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # Small blur to smooth jagged edges
    temp_roi = cv2.GaussianBlur(temp_roi, (3, 3), 0)

    # OCR config for temperature only
    temp_config = "--psm 7 -c tessedit_char_whitelist=0123456789."

    temp_text = pytesseract.image_to_string(
        temp_roi,
        config=temp_config
    )

    print("Temperature:", temp_text.strip())

    # Save debug images
    cv2.imwrite("0_original.jpg", frame)
    cv2.imwrite("1_thresh.jpg", thresh)
    cv2.imwrite("2_detected_display.jpg", original)
    cv2.imwrite("3_warped.jpg", warped)
    cv2.imwrite("4_warped_thresh.jpg", warped_thresh)
    cv2.imwrite("5_temp_roi.jpg", temp_roi)

else:
    print("No display contour detected.")

    cv2.imwrite("0_original.jpg", frame)
    cv2.imwrite("1_thresh.jpg", thresh)