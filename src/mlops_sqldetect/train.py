from mlops_sqldetect.model import Model
from mlops_sqldetect.data import MyDataset


def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here


if __name__ == "__main__":
    train()
