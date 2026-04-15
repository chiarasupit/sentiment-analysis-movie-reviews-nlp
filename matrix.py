import matplotlib.pyplot as plt
import numpy as np

# Updated confusion matrix values
cm = np.array([[10728, 1772],
               [2266, 10234]])

plt.figure()
plt.imshow(cm)

# Titles and labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Axis labels
labels = ["Positive", "Negative"]
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

# Add numbers inside boxes
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

# Color bar (like your example)
plt.colorbar()

# Save figure
plt.savefig("/Users/frangkysupit/Documents/confusion_matrix_updated.png")

# Show figure
plt.show()