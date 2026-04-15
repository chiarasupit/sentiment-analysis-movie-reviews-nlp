import matplotlib.pyplot as plt

# Updated data
sizes = [5000, 25000]
accuracy = [0.82, 0.84]

plt.figure(figsize=(6,5))

plt.plot(sizes, accuracy, marker='o')

plt.title("Accuracy vs Dataset Size")
plt.xlabel("Dataset Size")
plt.ylabel("Accuracy")

plt.grid()

# Save to Documents
plt.savefig("/Users/frangkysupit/Documents/accuracy_vs_dataset_size.png")

plt.show()