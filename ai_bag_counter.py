# counts number of loaded and unloaded bags individually 

# Bag comes from left side → crosses middle → goes right → Loaded +1
# Bag comes from right side → crosses middle → goes left → Unloaded +1

# ai_bag_counter.py

import cv2
import sys
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def process_video(source=0, frame_callback=None):

    cap = cv2.VideoCapture(source)

    loaded_count = 0
    unloaded_count = 0
    track_history = {}
    counted_ids = set()

    line_x = None  # auto adjust

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if line_x is None:
            line_x = frame.shape[1] // 2

        results = model.track(frame, persist=True)
        annotated = results[0].plot()

        if results[0].boxes.id is not None:

            for box, track_id in zip(results[0].boxes,
                                     results[0].boxes.id):

                cls = int(box.cls[0])
                label = model.names[cls]
                track_id = int(track_id)

                if label in ["backpack", "handbag", "suitcase"]:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)

                    if track_id in track_history:
                        prev_x = track_history[track_id]

                        if track_id not in counted_ids:

                            # Left → Right
                            if prev_x < line_x and center_x >= line_x:
                                loaded_count += 1
                                counted_ids.add(track_id)

                            # Right → Left
                            elif prev_x > line_x and center_x <= line_x:
                                unloaded_count += 1
                                counted_ids.add(track_id)

                    track_history[track_id] = center_x

        # Draw counting line
        cv2.line(annotated,
                 (line_x, 0),
                 (line_x, frame.shape[0]),
                 (255, 0, 0), 2)

        cv2.putText(annotated, f"Loaded: {loaded_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.putText(annotated, f"Unloaded: {unloaded_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # If using Streamlit later
        if frame_callback:
            frame_callback(annotated, loaded_count, unloaded_count)
        else:
            # Standalone mode
            cv2.imshow("AI Bag Counter", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Standalone Execution Support
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        process_video(video_path)
    else:
        process_video(0)