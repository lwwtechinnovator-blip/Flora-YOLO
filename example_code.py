import os
import torch
from ultralytics import YOLO
def train_with_pretrained_augmented():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
    model_yaml =
    data_yaml =
    output_project =
    output_name =
    pt_path=
    model = YOLO(model_yaml).load(pt_path)
    results = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        project=output_project,
        name=output_name,
        lr0=0.001,
        amp=False,
        optimizer='auto',
        patience=50,
        save=True,
        exist_ok=True,
        verbose=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        hsv_v=0.4,
        iou_type="FSIoU",
        Inner_iou=True,
        Focaler=False

    )

    return results

if __name__ == "__main__":
    train_with_pretrained_augmented()
