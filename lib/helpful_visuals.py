import matplotlib.pyplot as plt
import numpy as np

# Define time steps and nodes
time_steps = ['t-2', 't-1', 't', 't+1']
nodes = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'End-effector', 'Error']
colors = plt.cm.Paired(np.linspace(0, 1, len(nodes)))

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot time steps for each node
for i, node in enumerate(nodes):
    y = np.full(len(time_steps), i)
    ax.scatter(time_steps, y, label=node, color=colors[i], s=200)
    for j, step in enumerate(time_steps):
        ax.text(step, i, f'Feature @ {step}', va='center', ha='center', color='black')

# Highlight the target at t+1
ax.scatter('t+1', len(nodes) - 1, color='red', s=300, label='Target (Pose Deviation) @ t+1')
ax.text('t+1', len(nodes) - 1, 'Pose Deviation', va='center', ha='center', color='white')

# Formatting
ax.set_yticks(range(len(nodes)))
ax.set_yticklabels(nodes)
ax.set_xticks(time_steps)
ax.set_xticklabels(time_steps)
ax.set_title('Sequence Creation Process for Input Features and Target at t+1')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Sizes for the different data splits (adjust based on your specific configuration)
train_size = 0.65  # Proportion of training data
val_size = 0.15   # Proportion of validation data
test_size = 0.20  # Proportion of test data

# Create the figure
fig, ax = plt.subplots(figsize=(14, 2))

# Plot bars to represent the data splits
ax.barh(0, train_size, color='green', label='Training Set', left=0)
ax.barh(0, val_size, color='orange', label='Validation Set', left=train_size)
ax.barh(0, test_size, color='red', label='Test Set', left=train_size + val_size)

# Formatting
ax.set_xlim(0, 1)
ax.set_xticks([0, train_size, train_size + val_size, 1])
ax.set_xticklabels(['0%', f'{train_size*100:.0f}%', f'{(train_size + val_size)*100:.0f}%', '100%'])
ax.set_yticks([])
ax.set_title('Visualization of Train, Validation, and Test Split')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.show()
