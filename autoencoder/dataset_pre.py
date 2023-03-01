# https://pytorch.org/vision/stable/datasets.html

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 防止下载数据集报错
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def minst(variance='MNIST', batch_size=16, root='../datasets', download=True, resize=28, num_workers=8):
    """MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST

    Args:
        variance (str, optional):   数据集名称, [MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST]. Defaults to 'DTD'.
        batch_size (int, optional): batch_size. Defaults to 16.
        root (str, optional):       存放路径. Defaults to './dataset'.
        download (bool, optional):  是否下载数据集. Defaults to True.
        resize (int, optional):     图片大小. Defaults to 28.
        num_workers (int, optional):并行数. Defaults to 8.

    Returns:
        train_dataset, val_dataset, train_dataloader, val_dataloader
    """
    assert variance in ['MNIST', 'EMNIST', 'FashionMNIST', 'KMNIST', 'QMNIST'], \
        "variance should in CIFAR10, EMNIST, CIFAR100, FashionMNIST, KMNIST and QMNIST"

    dataset = {
            'MNIST':        datasets.MNIST,
            'EMNIST':       datasets.EMNIST,
            'FashionMNIST': datasets.FashionMNIST,
            'KMNIST':       datasets.KMNIST,
            'QMNIST':       datasets.QMNIST,
            }[variance]

    transform = transforms.Compose([transforms.Resize(resize, antialias=True),
                                    transforms.ToTensor(),
                                    # transforms.Normalize([0.485], [0.229]),
                                    ])

    train_dataset    = dataset(root, train=True, transform=transform, download=download)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset      = dataset(root, train=False, transform=transform, download=download)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    print(f"train images: {train_length}, val images: {val_length}")

    return train_dataset, val_dataset, train_dataloader, val_dataloader


#-------------------#
#   显示1通道的图片
#-------------------#
def show_1_channel():
    import matplotlib.pyplot as plt
    import numpy as np
    _, _, train_dataloader, _ = minst(root='./datasets')
    # 获取一批数据
    image_target = next(iter(train_dataloader))
    print(len(image_target))            # 2             images+targets
    print(len(image_target[0]))         # 1024          images

    # 均值，标准差
    mean, std = 0.485, 0.229

    # 24张图片
    target_list = image_target[1][:24]                          # target
    target_list = [target.numpy() for target in target_list]    # target tensor to numpy
    image_list  = image_target[0][:24]                          # image
    img_list    = []
    for image in image_list:
        image = image.numpy()
        # 反标准化  *标准差 + 均值
        image = image * std + mean
        # c,h,w -> h,w,c
        img_list.append(np.transpose(image, (1, 2, 0)))

    print(img_list[0].shape)    # (32, 32, 3)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(18, 9), dpi=100)
    for i in range(24):
        row = i // 8    # 0 1 2
        col = i  % 8    # 0 1 2 3 4 5 6 7
        axes[row][col].set_title(target_list[i])
        axes[row][col].imshow(img_list[i])
        # axes[row][col].imshow(img_list[i], cmap='gray')   # 灰度图像
        # axes[row][col].set_xlim(0, 31)
        # axes[row][col].set_ylim(31, 0)    31 0 不然图片上下颠倒
    plt.show()


if __name__ == "__main__":
    _, _, train_dataloader, val_dataloader = minst()
    x, _ = next(iter(train_dataloader))
    print(x.size()) # [16, 1, 28, 28]
    x = x.reshape(-1, 784)
    print(x.size()) # [16, 784]
