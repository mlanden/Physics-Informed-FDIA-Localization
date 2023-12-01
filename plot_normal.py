import matplotlib.pyplot as plt
import torch

checkpoint = "grid_14_reduction_lr"
profile_path = f"./checkpoint/{checkpoint}/model.pt"
profile = torch.load(profile_path)
mean = profile["mean"].numpy()
std = profile["std"].numpy()

fig, ax = plt.subplots(1, 2)
ax[0].hist(mean, bins=20)
ax[1].hist(std, bins=20)
ax[0].set(title="Mean normal loss",
          xlabel="Mean Physics Loss",
          ylabel="Count")
ax[1].set(title="Standard deviation normal loss",
          xlabel="Standard Deviation Physics Loss",
          ylabel="Count")
plt.tight_layout()
# plt.show()
plt.savefig("normal_profile.png")