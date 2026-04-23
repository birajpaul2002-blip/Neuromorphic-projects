from ultralytics import YOLO

model = YOLO("/work/biraj/neuromorphic/runs/detect/26N_offline_reviewkd/weights/best.pt")

results = model.val(data="/work/biraj/neuromorphic/testdataset.yaml",imgsz=1024, batch=1, conf=0.1,plots=True,device=0 )

print("mAP50-95:", results.box.map)
print("mAP50:", results.box.map50)
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("Speed:", results.speed)
