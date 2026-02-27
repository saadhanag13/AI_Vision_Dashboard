# counts the area of the wall - issues raised

# import cv2
# from ultralytics import YOLO

# model = YOLO("yolov8n-seg.pt")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     annotated = results[0].plot()

#     cv2.imshow("Wall Segmentation", annotated)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------
# counts the area of the wall that needs to be painted

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    wall_area_pixels = 0
    total_pixels = frame.shape[0] * frame.shape[1]

    if results[0].masks is not None:

        for mask, cls in zip(results[0].masks.data, results[0].boxes.cls):

            label = model.names[int(cls)]

            if label == "person":  # placeholder
                mask_np = mask.cpu().numpy()

                # Resize mask to frame size
                mask_resized = cv2.resize(
                    mask_np,
                    (frame.shape[1], frame.shape[0])
                )

                wall_area_pixels += np.sum(mask_resized > 0.5)

    wall_percentage = (wall_area_pixels / total_pixels) * 100

    cv2.putText(
        annotated,
        f"Paint Area: {wall_percentage:.2f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Wall Area Estimation", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()