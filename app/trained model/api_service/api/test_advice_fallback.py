import requests
import json

URL = "http://127.0.0.1:8002/advice"

def post_advice(prediction):
    payload = {"prediction": prediction}
    r = requests.post(URL, json=payload)
    print(r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

if __name__ == '__main__':
    post_advice("Tomato___Late_blight")
    post_advice("Unknown_Plant___Weird_disease")
