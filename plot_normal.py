import matplotlib.pyplot as plt
import torch

checkpoint = "grid_14_continuous"
profile_path = f"./checkpoint/{checkpoint}/model.pt"
profile = torch.load(profile_path)
mean = profile["mean"].numpy()
print(mean)
std = profile["std"].numpy()

fig, ax = plt.subplots(1, 2)
ax[0].hist(mean, bins=20)
ax[1].hist(std, bins=20)
ax[0].set_title("Mean normal state loss")
ax[1].set_title("Standard deviation of normal state loss")
plt.tight_layout()
# plt.show()
plt.savefig("normal_profile.png")