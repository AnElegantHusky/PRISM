"""Small executable demonstration without an LLM or optimization task."""

import numpy as np

from prism import PRISM


rng = np.random.default_rng(0)
responses = rng.normal(size=(128, 10))
prism = PRISM(128, k=5, feedback="cpi_augmented")
for algorithm in range(10):
    prism.observe_initial(algorithm, responses[:, algorithm])

print("selected instances:", prism.selected_indices)
print("formal-search fitness shape:", prism.transform_selected(rng.normal(size=5)).shape)
