"""Hardware and calibration-target constants for the Waveshare AR0144 Stereo USB Camera (A).

Spec source: https://www.waveshare.com/ar0144-stereo-usb-camera-a.htm
The camera exposes a SINGLE UVC video device that outputs one combined
frame containing both eyes side by side (left | right), already
row-synchronized in hardware ("frame-synchronized stereo output").
"""

# --- Video device ---
# Run `v4l2-ctl --list-devices` and adjust if the stereo camera doesn't land on index 0.
CAMERA_INDEX = 0
FOURCC = "MJPG"          # only MJPG reaches 30 FPS at full 2560x720; YUY2 caps at 5 FPS there

# Combined frame as produced by the sensor (left+right side by side).
FRAME_WIDTH = 2560
FRAME_HEIGHT = 720
FRAME_FPS = 30

SINGLE_WIDTH = FRAME_WIDTH // 2     # 1280 px per eye
SINGLE_HEIGHT = FRAME_HEIGHT        # 720 px per eye
IMAGE_SIZE = (SINGLE_WIDTH, SINGLE_HEIGHT)  # (width, height), OpenCV convention

# --- Optical / mechanical specs (datasheet + product page) ---
BASELINE_MM = 52.0                  # distance between the two lens centers
SENSOR_PIXEL_SIZE_MM = 0.003        # AR0144 pixel pitch, 3.0 um
FOCAL_LENGTH_MM = 2.88              # fixed-focus lens EFL
FOV_DIAGONAL_DEG = 74.0
FOV_HORIZONTAL_DEG = 65.0
FOV_VERTICAL_DEG = 43.0
RATED_DISTORTION_PCT = 0.2          # "<0.2%" per datasheet -> lenses are nearly distortion-free

# Rough pinhole estimate from the physical specs (focal length / pixel pitch).
# Only good enough to seed a "sample" calibration file before real calibration is run.
ESTIMATED_FOCAL_PX = FOCAL_LENGTH_MM / SENSOR_PIXEL_SIZE_MM   # ~960 px

# --- Checkerboard calibration target ---
# Counts are INNER corners (rows x columns), not squares. Update SQUARE_SIZE_MM
# to whatever you actually print/measure -- it sets the real-world scale of
# every downstream measurement (baseline, depth, triangulated points).
CHESSBOARD_ROWS = 6
CHESSBOARD_COLUMNS = 9
SQUARE_SIZE_MM = 25.0

# --- Calibration file paths ---
CALIB_DIR = "calibration_data"
SAMPLE_CALIB_FILE = f"{CALIB_DIR}/sample_calibration.yaml"
CALIB_FILE = f"{CALIB_DIR}/stereo_calibration.yaml"
