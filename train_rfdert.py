from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="models/rf-detr-medium.pth")

if __name__ == '__main__':  # ✅ Windows 必须要有这行
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # ✅ 防止 spawn 错误

    model.train(
        dataset_dir="data/car_coco",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="checkpoints"
    )
