import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(BASE_DIR, "yolo", "yolov3.weights")
config_path  = os.path.join(BASE_DIR, "yolo", "yolov3.cfg")
names_path   = os.path.join(BASE_DIR, "yolo", "coco.names")
image_path   = os.path.join(BASE_DIR, "test.jpg")

net = cv2.dnn.readNet(weights_path, config_path)

with open(names_path, "r") as f:
    classes = f.read().splitlines()

img = cv2.imread(image_path)
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detect in output:
        scores = detect[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), font, 0.6, (0, 255, 0), 2)

cv2.imshow("YOLO Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
