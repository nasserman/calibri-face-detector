import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# مدل FaceNet
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# پیش‌پردازش
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def extract_embedding(image):
    """
    image: PIL Image
    خروجی: numpy array با shape=(512,) و L2-normalized
    """
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img)

    # تبدیل به numpy
    emb = embedding.cpu().numpy().flatten()

    # L2 normalization (خیلی مهم برای دقت)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb

    emb = emb / norm
    return emb
