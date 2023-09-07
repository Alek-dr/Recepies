import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn


@dali.pipeline_def(batch_size=1, num_threads=1, device_id=0)
def pipe():
    """
    Примерный эквивалент  cv2.dnn.blobFromImages, разница на этапе ресайза
    """
    images, _ = dali.fn.readers.file(file_root="img_for_test")
    images = fn.decoders.image(
        images, device="cpu", output_type=dali.types.DALIImageType.RGB
    )

    images = dali.fn.resize(
        images,
        resize_x=224,
        resize_y=224,
        antialias=True,
        interp_type=dali.types.DALIInterpType.INTERP_CUBIC,
    )

    images = dali.fn.crop_mirror_normalize(
        images,
        scale=1.0 / 255.0,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229, 0.224, 0.225],
    )
    return images


if __name__ == "__main__":
    p = pipe()
    p.build()
    for batch in p.run():
        for img in batch:
            res = np.array(img)
            res = np.expand_dims(res, 0).astype(np.float32)
