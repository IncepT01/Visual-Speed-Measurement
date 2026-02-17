from turtle import st
import cv2
import numpy as np
import os

# Clear output directory first
for filename in os.listdir('./output'):
    file_path = os.path.join('./output', filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

original_resolution = (1920, 1200)

# Top left is (0,0)
# My order:
# 3 6  
# 2 5
# 1 4
dev0_pts = np.float32([[198, 708], [255, 666], [296, 642], [604, 701], [576, 664], [560,640]])
dev3_pts = np.float32([[1180, 750], [1222, 716], [1252, 693], [1561, 762], [1525, 723], [1507,696]])

dev0_pts = dev0_pts.reshape(-1, 1, 2)
dev3_pts = dev3_pts.reshape(-1, 1, 2)
H, mask = cv2.findHomography(dev0_pts, dev3_pts, cv2.RANSAC, 5.0)
print("H:\n", H)

bev_indices = [0, 3, 5, 2]
bev_pts_dev0 = dev0_pts[bev_indices]
bev_pts_dev3 = dev3_pts[bev_indices]
y_offset = -700
dst_width_bev, dst_height_bev = 600, original_resolution[1]
dst_pts = np.float32([[0,dst_height_bev + y_offset],[dst_width_bev,dst_height_bev + y_offset],[dst_width_bev,0],[0,0]])

H_bev_dev0 = cv2.getPerspectiveTransform(bev_pts_dev0, dst_pts)
H_bev_dev3 = cv2.getPerspectiveTransform(bev_pts_dev3, dst_pts)

def show_image(window_name, image, width=original_resolution[0], height=original_resolution[1]):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

prev_dev3_bev = None
image_count = 200
start_idx = 500
speed_history = []
smoothing_window = 10  # Number of frames to average over
for i in range(start_idx,start_idx + image_count):
    # Image stitching
    # Right camera
    img_path_dev0 = f'./pictures/Dev0_Image_w1920_h1200_fn{i}.jpg' 

    # Left camera
    img_path_dev3 = f'./pictures/Dev3_Image_w1920_h1200_fn{i}.jpg' 

    img_dev0 = cv2.imread(img_path_dev0)

    img_dev3 = cv2.imread(img_path_dev3)

    h, w = img_dev3.shape[:2]

    out_w = w + img_dev0.shape[1]
    out_h = h

    warped_dev0 = cv2.warpPerspective(img_dev0,H,(out_w, out_h))

    stitched = warped_dev0.copy()
    stitched[0:h, 0:w] = img_dev3

    # BEV 
    dev0_warped_bev = cv2.warpPerspective(img_dev0, H_bev_dev0, (dst_width_bev, dst_height_bev))
    dev3_warped_bev = cv2.warpPerspective(img_dev3, H_bev_dev3, (dst_width_bev, dst_height_bev))

    # Optical flow
    gray_curr = cv2.cvtColor(dev3_warped_bev, cv2.COLOR_BGR2GRAY)

    flow_img = np.zeros_like(dev3_warped_bev)

    if prev_dev3_bev is not None:
        gray_prev = cv2.cvtColor(prev_dev3_bev, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(dev3_warped_bev)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2][mag < 1.0] = 0

        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply a threshold to consider only significant motion
        motion_threshold = 2.0
        significant_motion = mag[mag > motion_threshold]

        if significant_motion.size > 0:
            avg_speed = np.mean(significant_motion)
        else:
            avg_speed = 0

        # Add the current speed to the history and smooth it
        speed_history.append(avg_speed)
        if len(speed_history) > smoothing_window:
            speed_history.pop(0)

        smoothed_speed = np.mean(speed_history) * 3
        speed_text = f"Speed: {smoothed_speed:.2f} km/h"

        # Overlay the speed text on the bottom-left corner of the flow image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)
        thickness = 2
        text_x = 10
        text_y = stitched.shape[0] - 10

        cv2.putText(stitched, speed_text, (text_x, text_y), font, font_scale, font_color, thickness)

    prev_dev3_bev = dev3_warped_bev.copy()

    # Cobmining
    combined_width = stitched.shape[1] + dev3_warped_bev.shape[1] + dev0_warped_bev.shape[1] + flow_img.shape[1]

    combined = np.zeros((stitched.shape[0], combined_width, 3),dtype=np.uint8)

    combined[0:original_resolution[1], 0:stitched.shape[1]] = stitched

    combined[0:original_resolution[1], stitched.shape[1]:stitched.shape[1] + dev3_warped_bev.shape[1]] = dev3_warped_bev
    combined[0:original_resolution[1], stitched.shape[1] + dev3_warped_bev.shape[1] : stitched.shape[1] + dev3_warped_bev.shape[1] + dev0_warped_bev.shape[1]] = dev0_warped_bev

    combined[0:original_resolution[1], stitched.shape[1] + dev3_warped_bev.shape[1] + dev0_warped_bev.shape[1] :] = flow_img

    if(i - 10 < 10):
        output_path = f'./output/00{i-10}.png'
    elif(i - 10 < 100):
        output_path = f'./output/0{i-10}.png'
    else:
        output_path = f'./output/{i-10}.png'

    cv2.imwrite(output_path, combined)

video_path = './output/final.mp4'
fps = 6

image_files = sorted([f for f in os.listdir('./output') if f.endswith('.png')])

first_frame = cv2.imread(os.path.join('./output', image_files[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for img_file in image_files:
    frame = cv2.imread(os.path.join('./output', img_file))
    video_writer.write(frame)

video_writer.release()
