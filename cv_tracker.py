import cv2
import numpy as np
from ultralytics import YOLO

# Centroid Tracker Class
class CentroidTracker:
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = dict()
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or path to your weights

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

tracker = CentroidTracker(maxDisappeared=30)
prev_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    rects = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rects.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Update tracker
    objects = tracker.update(rects)
    current_ids = set(objects.keys())

    # Detect new and missing objects
    new_ids = current_ids - prev_ids
    missing_ids = prev_ids - current_ids

    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        color = (0, 255, 0)
        if objectID in new_ids:
            text += " (NEW)"
            color = (0, 255, 255)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, tuple(centroid), 4, color, -1)

    for objectID in missing_ids:
        print(f"Object ID {objectID} missing!")

    prev_ids = current_ids

    cv2.imshow("Real-Time Object Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()