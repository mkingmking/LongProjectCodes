import matplotlib.pyplot as plt
import numpy as np

# Labels for each bar
labels = [
    "Quantum Fisher Info (QFI)",
    "Classical Fisher Info (Z-basis)",
    "Classical Fisher Info (X⊗Z⊗Z)",
    "Classical Fisher Info (Y-basis)"
]

# Replace these values with your actual constant values
values = [9.0, 0.0, 9.0, 9.0]

# Colors and hatches for differentiation
colors = ['orange', 'red', 'blue', 'magenta']
hatches = ['', '', '', '']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, values, color=colors)

# Add hatching to differentiate bars
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Labels and title
ax.set_ylabel("Fisher Information")
ax.set_title("QFI vs. CFI for GHZ State Phase Estimation")
plt.xticks(rotation=15, ha='right')

# Show values on top of bars
for i, v in enumerate(values):
    ax.text(i, v + 0.1, f"{v}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
