import cv2
import numpy as np
import os
import csv
import argparse

# -----------------------------
# HSV color ranges for red, yellow, green
# -----------------------------
HSV_RANGES = {
    "red1": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "yellow": ((15, 100, 100), (35, 255, 255)),
    "green": ((36, 80, 70), (90, 255, 255)),
}

LOG_FILE = "detections_log.csv"

def build_mask(hsv, color):
    if color == "red":
        lower1, upper1 = HSV_RANGES["red1"]
        lower2, upper2 = HSV_RANGES["red2"]
        mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        return cv2.bitwise_or(mask1, mask2)
    else:
        lower, upper = HSV_RANGES[color]
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

def detect_traffic_light(hsv):
    masks = {
        "RED": build_mask(hsv, "red"),
        "YELLOW": build_mask(hsv, "yellow"),
        "GREEN": build_mask(hsv, "green"),
    }

    state = "UNKNOWN"
    max_area = 0
    chosen_bbox = None

    for color, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # filter noise
                x, y, w, h = cv2.boundingRect(cnt)
                if area > max_area:
                    max_area = area
                    state = color
                    chosen_bbox = (x, y, w, h)

    return state, chosen_bbox

def log_detection(log_writer, frame_id, state, bbox):
    if bbox is None:
        log_writer.writerow([frame_id, state, -1, -1, -1, -1])
    else:
        x, y, w, h = bbox
        log_writer.writerow([frame_id, state, x, y, w, h])

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {input_path}")
    print(f"[INFO] Annotated video -> {output_path}")
    print(f"[INFO] Log file -> {LOG_FILE}")

    with open(LOG_FILE, "w", newline="") as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["frame/image_id", "state", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            state, bbox = detect_traffic_light(hsv)

            if bbox is not None:
                x, y, w, h = bbox
                color_map = {
                    "RED": (0, 0, 255),
                    "YELLOW": (0, 255, 255),
                    "GREEN": (0, 255, 0),
                    "UNKNOWN": (255, 255, 255),
                }
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[state], 2)
                cv2.putText(frame, state, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[state], 2)

            log_detection(log_writer, frame_id, state, bbox)
            out.write(frame)

            # Show live feed
            cv2.imshow("Traffic Light Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing finished.")

def process_image(input_path, output_path):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    state, bbox = detect_traffic_light(hsv)

    if bbox is not None:
        x, y, w, h = bbox
        color_map = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0),
            "UNKNOWN": (255, 255, 255),
        }
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[state], 2)
        cv2.putText(frame, state, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[state], 2)

    cv2.imwrite(output_path, frame)
    print(f"[INFO] Annotated image -> {output_path}")

    # Log detection
    with open(LOG_FILE, "w", newline="") as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["frame/image_id", "state", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
        log_detection(log_writer, os.path.basename(input_path), state, bbox)

    print(f"[INFO] Log saved as {LOG_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Light Detection (Video/Image)")
    parser.add_argument("--input", required=True, help="Path to input video or image")
    parser.add_argument("--output", default=None, help="Path to save annotated output (video/image)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        if output_path is None:
            output_path = "traffic_light_annotated.mp4"
        process_video(input_path, output_path)
    else:
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = base + "_annotated.jpg"
        process_image(input_path, output_path)
