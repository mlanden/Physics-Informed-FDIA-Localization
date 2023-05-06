import matplotlib.pyplot as plt
import torch

profile_path = "./checkpoint/swat_2_equ/normal_mean.pt"
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
plt.savefig("normal_profile_regularized.png")