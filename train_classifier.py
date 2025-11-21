import torch
import torchvision.transforms as v2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from models.image_classifier import Classifier


def main():
    """
    Train test and evaluate the image classifier
    training framework came from https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
    """

    config = OmegaConf.load("config/classifier.yaml")

    img_size = config.vit.embed.img_size
    transform = v2.Compose([v2.Resize(img_size), v2.ToTensor()])

    train_set = MNIST(root="data", train=True, download=True, transform=transform)
    test_set = MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=config.train.batch_size
    )
    test_loader = DataLoader(
        test_set, shuffle=False, batch_size=config.train.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using {device}")

    classifier = Classifier(config, device)
    classifier.train_model(train_loader)
    res = classifier.evaluate(test_loader)
    print(f"Classifier accuracy: {res}")


if __name__ == "__main__":
    main()
