1. Upgrade or Ensemble the OCR Engine
Tesseract often struggles with digital/segment LCD fonts and glare artifacts commonly found on water heater displays.

Replace or Pair with Deep Learning OCRs: Integrate PaddleOCR or EasyOCR. PaddleOCR consistently achieves significantly lower Character Error Rates (CER) on emulated or LCD displays compared to Tesseract without requiring extensive manual binarization.

OCR Ensemble Strategy: Run multiple engines concurrently (e.g., Tesseract + PaddleOCR) on each variant. Combine their results using character confidence weights or simple majority voting.

Fine-Tuned LCD/7-Segment Models: If sticking with Tesseract, supply custom .traineddata models trained specifically on 7-segment digital displays (e.g., letsgodigital or ssd) rather than standard eng.

2. Upgrade Image Preprocessing and Unwarping
Small geometric errors in corner detection lead to subtle perspective distortion, degrading down-stream OCR.

Sub-Pixel Corner Refinement: After identifying display corners in _find_display_contour, run cv2.cornerSubPix() around the polygon corners. Sub-pixel accuracy yields sharper, unwarped target ROI crops.

Sauvola / Niblack Adaptive Binarization: Replace or supplement standard Otsu and Gaussian thresholding with Sauvola Binarization (available in scikit-image as skimage.filters.threshold_sauvola). Sauvola handles uneven display backlighting, regional glare, and gradient degradation far better than Otsu or global adaptive thresholding.

Deep Learning Super-Resolution (DLSR): Instead of resizing small crops using cv2.resize(..., interpolation=cv2.INTER_CUBIC), use OpenCV’s cv2.dnn_superres module with pre-trained models such as EDSR or ESPCN. These upscale low-resolution display text while sharpening pixelated boundaries.

3. Implement Multi-Frame Temporal Aggregation
Single-frame OCR reads are susceptible to backlight flicker, camera sensor noise, and frame drop artifacts.

Sequential Frame Voting: Capture 3 to 5 sequential camera frames in capture_and_read_display.

Confidence Fusion: Calculate OCR for all frames and pick the result that appears most frequently (majority vote) or sums the highest overall confidence across the sequence.