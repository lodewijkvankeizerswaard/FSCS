from torchvision.datasets import CelebA

def get_celebA(root="data"):
    train = CelebA(root=root, split="train", download=True)
    val = CelebA(root=root, split="val", download=True)
    test = CelebA(root=root, split="test", download=True)
    return (train, val, test)