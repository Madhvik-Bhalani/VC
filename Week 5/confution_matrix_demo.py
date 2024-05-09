import matplotlib.pyplot as plt
import seaborn as sns

# replace this matrix with your actual matrix
conf_matrix = [[93,14],[1829]]

# Labels for classes
class_labels = ['0', '1']

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Confusion Matrix')
plt.show()