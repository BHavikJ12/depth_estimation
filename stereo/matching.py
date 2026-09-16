"""Dense stereo matching: Semi-Global Matching (SGM) + WLS edge-aware smoothing.

cv.StereoSGBM implements Hirschmuller's Semi-Global Matching -- the same
family of algorithm the Teledyne embedded-stereo guide specifies. MODE_HH
runs the full 8-direction cost aggregation (closest to the original SGM
paper); MODE_SGBM_3WAY trades some accuracy for speed by aggregating over
only 3 directions, useful if full SGM is too slow on your hardware.
"""

import cv2 as cv


def make_matchers(mode=cv.STEREO_SGBM_MODE_HH, num_disp=16 * 8, block_size=5):
    """Left matcher does the real work; right matcher is only needed so the
    WLS filter can cross-check left-vs-right disparity and reject anything
    that disagrees (the main source of dense-matching speckle noise)."""
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        preFilterCap=63,
        speckleWindowSize=100,
        speckleRange=32,
        mode=mode,
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    return left_matcher, right_matcher


def make_wls_filter(left_matcher):
    """Edge-aware smoothing: fuses left+right disparity, guided by the left
    image's edges, so flat/textureless regions get smoothed heavily while
    real depth boundaries stay sharp."""
    wls = cv.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)
    return wls


def compute_filtered_disparity(left_matcher, right_matcher, wls_filter, gray_l, gray_r, rect_l):
    """Returns (disparity_px, confidence) as float32 arrays, same shape as gray_l."""
    disp_l = left_matcher.compute(gray_l, gray_r)
    disp_r = right_matcher.compute(gray_r, gray_l)
    filtered = wls_filter.filter(disp_l, rect_l, disparity_map_right=disp_r)
    disparity_px = filtered.astype("float32") / 16.0
    confidence = wls_filter.getConfidenceMap()
    return disparity_px, confidence
