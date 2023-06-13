import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .dataset import MovieLensTrainDataset
from .model import NCF


def main():
    ratings = pd.read_csv("data/ml-20m/ratings.csv")

    num_users = ratings["userId"].max() + 1
    num_items = ratings["movieId"].max() + 1

    all_movieIds = ratings["movieId"].unique()

    model = NCF(num_users, num_items)
    dataloader = DataLoader(MovieLensTrainDataset(ratings, all_movieIds), batch_size=32768, num_workers=8)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=TensorBoardLogger("logs/", name="movielens-ncf"),
        callbacks=[ModelCheckpoint(dirpath="trained_models", monitor="train/loss", save_last=True, mode="min")],
    )

    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
