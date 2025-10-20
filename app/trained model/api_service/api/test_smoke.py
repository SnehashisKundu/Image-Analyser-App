import requests
import io
from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:8002"

def check_health():
    r = requests.get(f"{BASE}/health")
    print('/health', r.status_code, r.text)

def check_labels():
    r = requests.get(f"{BASE}/labels")
    print('/labels', r.status_code)
    try:
        print(r.json() if r.status_code==200 else r.text)
    except Exception:
        print(r.text)

def check_advice(prediction='Tomato___Late_blight'):
    r = requests.post(f"{BASE}/advice", json={"prediction": prediction})
    print('/advice', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

def check_predict():
    # Create a small synthetic green image
    img = Image.new('RGB', (224,224), color=(34,139,34))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    files = {'file': ('leaf.jpg', buf, 'image/jpeg')}
    r = requests.post(f"{BASE}/predict", files=files)
    print('/predict', r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)

if __name__ == '__main__':
    check_health()
    check_labels()
    check_advice()
    check_predict()
