import cv2
import numpy as np
from tqdm import tqdm

# ——— CONFIG ———
INPUT_PATH   = "input.mp4"
OUTPUT_PATH  = "output.mp4"
YOLO_CFG     = "yolo/yolov3.cfg"
YOLO_WEIGHTS = "yolo/yolov3.weights"
COCO_NAMES   = "yolo/coco.names"
TARGET_CLASS = "person"  # Track this class; you’ll get all detected persons

CONF_THRESH = 0.5
NMS_THRESH  = 0.4

# ——— HELPER: create tracker across OpenCV versions ———
def make_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    raise RuntimeError("No suitable tracker available in this OpenCV build")

# ——— LOAD YOLO ———
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f]

raw_out = net.getUnconnectedOutLayers()
out_idxs = raw_out.flatten() if hasattr(raw_out, "flatten") else raw_out
layer_names   = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in out_idxs]

# ——— SET UP VIDEO I/O ———
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video {INPUT_PATH}")

fps          = cap.get(cv2.CAP_PROP_FPS)
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ——— STORAGE FOR MULTI-TRACKERS ———
trackers = []
detections_done = False

# ——— PROCESS FRAMES WITH PROGRESS BAR ———
with tqdm(total=total_frames, desc="Processing frames") as pbar:
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if not detections_done:
            # Run YOLO to detect all persons in first frame
            blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes, confidences = [], []
            for output in outputs:
                for det in output:
                    scores     = det[5:]
                    class_id   = int(np.argmax(scores))
                    conf       = float(scores[class_id])
                    if conf > CONF_THRESH and classes[class_id] == TARGET_CLASS:
                        cx, cy = int(det[0] * width), int(det[1] * height)
                        w, h   = int(det[2] * width), int(det[3] * height)
                        x, y   = cx - w // 2, cy - h // 2
                        boxes.append([x, y, w, h])
                        confidences.append(conf)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
            # Flatten indices
            if indices is None:
                idxs = []
            elif isinstance(indices, (list, tuple)):
                idxs = [i[0] if hasattr(i, '__iter__') else i for i in indices]
            else:
                idxs = indices.flatten().tolist()

            # Initialize a tracker per detection
            for i in idxs:
                x, y, w, h = boxes[i]
                tr = make_tracker()
                tr.init(frame, (x, y, w, h))
                trackers.append(tr)

            detections_done = True

        else:
            # Update and draw all trackers
            for tr in trackers:
                success, bbox = tr.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, TARGET_CLASS, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        pbar.update(1)

# ——— CLEANUP ———
cap.release()
writer.release()
print(f"Finished processing. Saved to {OUTPUT_PATH}")