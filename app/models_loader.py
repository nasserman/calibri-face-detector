import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("/home/nikan/Downloads/yolov8n-face.pt")

facenet_model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)
