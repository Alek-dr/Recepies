# Некогда объяснять

### Кодирование декодирование изображений

#### Отправить бинарное изображение, получить ndarray

```
from flask import Flask, request, jsonify
```

```
curl POST -kv --data-binary @"cat.jpg"  http://127.0.0.1:5000/read_image
```

```
@app.route('/read_image', methods=['POST'])
def read_image():
    byte_img = request.stream.read()
    res = cv2.imdecode(np.frombuffer(byte_img, dtype=np.uint8), 1)
    return jsonify({"Ok": "Read image successfully"}), HTTPStatus.OK
```

byte_img имеет вид:
b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00....

#### Отправить json

```
curl -H "Content-Type: application/json" --data @example.json http://127.0.0.1:5000/read_json
```

```
@app.route('/read_json', methods=['POST'])
def read_json():
    data = request.json
    return jsonify({"Ok": "Read json successfully"}), HTTPStatus.OK
```

#### Отправить multipart

Отправляем бинарное изображение, json и параметр name

```
curl -X POST -H "Content-Type: multipart/form-data" -F "img=@cat.jpg" -F "name=Tom" -F "data=@example.json" http://127.0.0.1:5000/read_multipart
```

```
@app.route('/read_multipart', methods=['POST'])
def read_multipart():
    byte_img = request.files['img'].stream.read()
    json_data_binary = request.files['data'].stream.read()
    json_data = json.loads(json_data_binary)
    return jsonify({"Ok": "All data read successfully"}), HTTPStatus.OK
```

#### Кодирование в base64

```
import base64
from io import BytesIO
from PIL import Image

buffered = BytesIO()
format = "PNG"
img = Image.open(img_path)
img.save(buffered, format=format)
buffered.seek(0)
img_byte = buffered.getvalue()
encoded_string = base64.b64encode(img_byte).decode("utf-8")
```