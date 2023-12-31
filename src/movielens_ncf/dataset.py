import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class MovieLensTrainDataset(Dataset):
    """MovieLens Pytorch Dataset for Training
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    """

    def __init__(self, ratings, all_movieIds):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_movieIds):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings["userId"], ratings["movieId"]))

        num_negatives = 10
        for u, i in tqdm(user_item_set):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(u)
                labels.append(0)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
