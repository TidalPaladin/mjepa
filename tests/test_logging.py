import pandas as pd
import torch

from mjepa.logging import CSVLogger, SaveImage


def test_csv_logger_writes_and_reads(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(path, interval=1, accumulate_grad_batches=1)

    logger.log(epoch=0, step=0, microbatch=0, loss=1.0)
    logger.log(epoch=0, step=1, microbatch=1, loss=2.0)

    df = logger.get_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["loss"]) == [1.0, 2.0]
    assert list(df["step"]) == [0, 1]


def test_csv_logger_respects_interval_and_accumulation(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(path, interval=2, accumulate_grad_batches=2)

    logger.log(epoch=0, step=0, microbatch=0, loss=1.0)
    logger.log(epoch=0, step=1, microbatch=0, loss=2.0)
    logger.log(epoch=0, step=1, microbatch=1, loss=3.0)

    df = logger.get_df()

    assert len(df) == 1
    assert float(df.loc[0, "loss"]) == 3.0


def test_save_image_writes_file(tmp_path):
    output = tmp_path / "output.png"
    save_image = SaveImage(output, max_save_images=2)

    images = torch.rand(4, 3, 8, 8)
    save_image(images)

    assert output.exists()
