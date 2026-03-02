# counts the area of the wall that needs to be painted

# import cv2
# import numpy as np
# from ultralytics import YOLO

# model = YOLO("yolov8n-seg.pt")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     annotated = results[0].plot()

#     wall_area_pixels = 0
#     total_pixels = frame.shape[0] * frame.shape[1]

#     if results[0].masks is not None:

#         for mask, cls in zip(results[0].masks.data, results[0].boxes.cls):

#             label = model.names[int(cls)]

#             if label == "person":  # placeholder
#                 mask_np = mask.cpu().numpy()

#                 # Resize mask to frame size
#                 mask_resized = cv2.resize(
#                     mask_np,
#                     (frame.shape[1], frame.shape[0])
#                 )

#                 wall_area_pixels += np.sum(mask_resized > 0.5)

#     wall_percentage = (wall_area_pixels / total_pixels) * 100

#     cv2.putText(
#         annotated,
#         f"Paint Area: {wall_percentage:.2f}%",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2
#     )

#     cv2.imshow("Wall Area Estimation", annotated)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import sys

def process_frame(frame):

    annotated = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 100, 200)

    # Smooth edges a bit
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    total_pixels = frame.shape[0] * frame.shape[1]

    edge_pixels = np.sum(edges > 0)

    # Brick = more edges
    # Paint = less edges

    unpaint_percentage = (edge_pixels / total_pixels) * 100
    paint_percentage = 100 - unpaint_percentage

    # Highlight brick areas (edges)
    overlay = frame.copy()
    overlay[edges > 0] = (0, 0, 255)

    annotated = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    cv2.putText(
        annotated,
        f"Painted: {paint_percentage:.2f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Unpainted (Brick): {unpaint_percentage:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return annotated

# -------------------------------------------------
# CHECK IF IMAGE PATH IS PROVIDED
# -------------------------------------------------

if len(sys.argv) > 1:
    # Image mode
    image_path = sys.argv[1]
    frame = cv2.imread(image_path)

    if frame is None:
        print("Error loading image.")
        sys.exit()

    result = process_frame(frame)

    cv2.imshow("Wall Paint Estimation - Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Webcam mode
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)

        cv2.imshow("Wall Paint Estimation - Live", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()