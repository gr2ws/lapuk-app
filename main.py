from sympy.physics.units import force

import img_preprocess
from ultralytics import YOLO

# img_preprocess.structure_dataset('trash_data')

img_classifier = YOLO("models/yolo11s-cls.pt")

metrics = img_classifier.train(data="dataset_structured/trash_data", epochs=100)

img_classifier.export()

print(metrics)