import pandas as pd
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    name = "14"
    base_profile = f"../../data/grid/{name}_base.csv"
    base_data = pd.read_csv(base_profile)
    n_profiles = 5000 // 3
    load_idxs = np.arange(len(base_data))

    for col in base_data:
        profiles = defaultdict(list)
        for percent in [.1, .2, .3]:
            for i, load in enumerate(base_data[col]):
                new_loads = np.random.normal(load, abs(percent * load), n_profiles)
                profiles[i].extend(new_loads)

                # validate
                std = np.std(new_loads)
                variability = std / load * 100
                print("variability", variability)
        profiles = pd.DataFrame(profiles).sample(frac=1)
        profiles.to_csv(f"../../data/grid/{name}_{col}.csv", index=False)