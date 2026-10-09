from anomalib.data import MVTecAD


def get_mvtec_dataloaders(
    category: str = "bottle",
    batch_size: int = 32,
    root: str = "./datasets/MVTecAD",
):

    datamodule = MVTecAD(
        root=root,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    # automatically handles download and setup
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule
