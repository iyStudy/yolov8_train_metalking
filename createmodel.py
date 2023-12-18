from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# 実行時の「mAP50-95」が評価値
model.train(data="dataset.yaml", epochs=10, batch=8, workers=4, degrees=90.0)
model.save("yolov8n_metal_king.pt")
