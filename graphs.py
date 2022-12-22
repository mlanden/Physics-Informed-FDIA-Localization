import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd

conf_path = "configs/swat.yaml"
with open(conf_path, "r") as fd:
    conf = yaml.safe_load(fd)
normal = pd.read_csv(conf["data"]["normal"], skiprows=0, header=1).to_numpy()
attack = pd.read_csv(conf["data"]["attack"], skiprows=0, header=1).to_numpy()

i = 2
end = 100000
norm = np.argwhere(attack[:end, -1] == "Normal")#, attack.iloc[:end, i])
attck = np.argwhere(attack[:end, -1] == "Attack")#, attack.iloc[:end, i])
plt.scatter(norm, attack[norm, i], color="green")
plt.scatter(attck, attack[attck, i], color="red")
plt.xticks(np.arange(0, end, end // 10))
plt.yticks(np.arange(attack[:, i].min(), attack[:, i].max(), 50))
plt.show()
# plt.savefig("attack.png")
