import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    losses_path = "../checkpoint/swat_fps/normal_mean.pt"
    losses = torch.load(losses_path)
    means = losses["mean"]
    std = losses["std"]

    fig, axes = plt.subplots(1,2)
    axes[0].hist(means)
    axes[1].hist(std)
    axes[0].set_title("Mean loss")
    axes[1].set_title("Std loss")
    plt.savefig("losses.png")