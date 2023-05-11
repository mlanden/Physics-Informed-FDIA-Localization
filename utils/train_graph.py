import json

import matplotlib.pyplot as plt

files = ["equation_val", "prediction_val"]
names = ["With equation", "Without equation"]
fix, ax = plt.subplots()
for file, name in zip(files, names):
    with open(f"../results/{file}.json", "r") as fd:
        data = json.load(fd)

    xvalsues = []
    yvalues = []
    for step in data:
        xvalsues.append(step[1])
        yvalues.append(step[2])
    ax.plot(xvalsues, yvalues, label=name)
ax.set(xlabel="Training Batch",
       ylabel="Validation Loss")
plt.show()