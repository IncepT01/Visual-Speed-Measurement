# Multi-Camera BEV Stitching & Optical Flow Pipeline

This project implements a computer vision pipeline using OpenCV and
NumPy to:

-   Perform multi-camera image stitching using homography
-   Generate a Bird's Eye View (BEV) transformation
-   Compute dense optical flow (Farneback method)
-   Estimate motion speed from optical flow magnitude
-   Generate a combined visualization video output

The system processes synchronized image sequences from two cameras and
produces a stitched, perspective-transformed, and motion-annotated
output video.

Google drive link to video:
https://drive.google.com/file/d/1kF3WorgQQzJlVGUG7lQVk01uzuFJQNvJ/view?usp=sharing

One frame:
![Pipeline Output](assets/ExampleFrame.png)

------------------------------------------------------------------------

## Features

### 1. Camera Calibration & Homography

-   Computes homography between two camera views using manually selected
    corresponding points
-   Uses RANSAC for robustness
-   Warps images into a shared coordinate system

### 2. Bird's Eye View (BEV) Transformation

-   Applies perspective transformation to both camera feeds
-   Generates top-down projections for motion analysis
-   Customizable destination resolution

### 3. Optical Flow (Motion Estimation)

-   Uses Farneback dense optical flow
-   Computes pixel-wise motion vectors between consecutive frames
-   Converts flow to HSV visualization
-   Filters low-magnitude noise
-   Estimates average motion speed (pixels/frame)

### 4. Visualization Output

For each frame, the pipeline generates a combined image including: -
Stitched multi-camera view - BEV projection from camera 1 - BEV
projection from camera 2 - Optical flow visualization - Motion speed
overlay

All frames are compiled into an `.mp4` video.

------------------------------------------------------------------------

## Technologies Used

-   Python 3
-   OpenCV (cv2)
-   NumPy

------------------------------------------------------------------------

## Motion Estimation

Average speed is calculated as:

mean(optical_flow_magnitude)

This provides a relative pixel/frame motion estimate.

------------------------------------------------------------------------

## Notes

-   Homography points are manually defined.
-   Optical flow is sensitive to lighting and texture conditions.
-   Speed estimation is relative (pixel-based), not real-world
    calibrated.

------------------------------------------------------------------------

## Future Improvements

-   Automatic feature matching (ORB/SIFT instead of manual points)
-   Real-world speed calibration
-   GPU acceleration
-   Better blending for stitched images
-   Real-time processing support
