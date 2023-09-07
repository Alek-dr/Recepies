import copy
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import InterpolationMode

MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
STD = np.array([0.229, 0.224, 0.225])
SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@profile
def numpy_preprocess(images: Iterable[np.ndarray]) -> np.ndarray:
    preprocessed_images = []
    for img in images:
        img = cv2.resize(np.array(img), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocessed_images.append(img)
    images = np.array(preprocessed_images)
    preprocessed_images.clear()
    images = images / 255.0
    images = (images - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    images = np.transpose(images, (0, 3, 1, 2))
    images = np.ascontiguousarray(images, dtype=np.float32)
    return images


@profile
def cv_blob_preprocess(images: Iterable[np.ndarray]) -> np.ndarray:
    input_blob = cv2.dnn.blobFromImages(
        images=images,
        scalefactor=1.0 / 255.0,
        size=(SIZE, SIZE),  # img target size
        mean=MEAN,
        swapRB=False,  # BGR -> RGB
        crop=False,  # center crop
    )
    input_blob /= STD.reshape((3, 1, 1))
    return input_blob


@profile
def pytorch_transform(images: Iterable[np.ndarray]):
    preprocess = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    res = []
    for img in images:
        img = img.astype(np.float32)
        img /= 255.0
        tensor = torch.from_numpy(img).to(device)
        tensor = torch.permute(tensor, (2, 0, 1))
        processed = preprocess(tensor)
        res.append(processed)
    return torch.stack(res).cpu().numpy()


def equal_transforms(images: Iterable[np.ndarray]):
    functions = [numpy_preprocess, cv_blob_preprocess]
    input_data = copy.deepcopy(images)
    res = functions[0](input_data)
    results = [res]
    equals = []
    for f in functions[1:]:
        input_data = copy.deepcopy(images)
        res = f(input_data)
        eq = np.array_equal(results[-1].round(3), res.round(3))
        if not eq:
            print(f"{f} gives different results")
        results.append(res)
        equals.append(eq)
    return all(equals)


"""
Все три функции выполняют одинаковый препроцессинг изображений. Однако,
препроцессинг с помощью функции pytorch_transform дает незначительно отличиающийся результат на этапе 
изменения размера - для любого алгоритма ресайза.

Выполнить профилировние кода с помощью line-profiler:
kernprof -l image_preprocessing.py 
python -m line_profiler image_preprocessing.py.lprof

По результатам профилирования выигрывает cv_blob_preprocess, который быстрее 
препроцессинга с помощью numpy примерно в 2.5 раза. Преобразования с помощью pytorch_transform
медленнее cv_blob_preprocess в 90 раз с использованием gpu. Однако, наверняка можно сделать batch-преобразование 
и сделать warmup.
"""

if __name__ == "__main__":
    img_path = Path("img_for_test/0")
    batch = []
    for file in img_path.glob("*.jpg"):
        image = cv2.imread(str(file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch.append(image)

    for _ in range(10):
        numpy_preprocess(copy.deepcopy(batch))
        cv_blob_preprocess(copy.deepcopy(batch))
        pytorch_transform(copy.deepcopy(batch))
