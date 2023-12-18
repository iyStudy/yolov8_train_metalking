from ultralytics import YOLO
# モデルを選択
# model = YOLO("yolov8n.pt")
# model = YOLO("yolov8x.pt")
#model = YOLO("yolov8x-seg.pt")
model = YOLO("yolov8n_metalking.pt")



# model.to("cuda")

# 検出対象ファイル指定
#results = model("https://ultralytics.com/images/bus.jpg", save=True)
#results = model("./images/test2.jpg", save=True)
# results = model("./images/test2.mp4", save=True)

# WEBカメラからリアルタイム検出
# results = model(0 , show=True)
results = model(0, show=True,  conf=0.65)
for i in enumerate(results):
    print(i)