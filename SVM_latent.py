import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

#npz data intergrate
def select_files(img_dir, subfolders):
    selected_files = []
    for subfolder in subfolders:
        file_pattern = os.path.join(img_dir, subfolder)
        file = os.path.join(file_pattern, 'projected_w.npz')
        selected_files.append(file)
    return selected_files

def list_subfolders(folder_path):
    subfolder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolder_names.append(item)
    return subfolder_names

def merge_npz_files(npz_files):
    all_features = []
    all_labels = []
    for file in npz_files:
        data = np.load(file)
        features = data['features']
        labels = data['labels']
        all_features.append(features)
        all_labels.append(labels)
    return all_features, all_labels


data_dir = 'frackles'
img_dir = os.path.join(data_dir,'anti-latent')
img_dir_frackles = os.path.join(data_dir,'frackles-latent')

subfolders_frackle = list_subfolders(img_dir_frackles)
subfolders_anti = list_subfolders(img_dir)
npz_files_anti = select_files(img_dir, subfolders_anti)
npz_files_frackle = select_files(img_dir_frackles, subfolders_frackle) 
#print(npz_files_frackle[0])           #print data path
#print(npz_files_anti[0])        #print data path

'''
# print all the names
print("Arrays in the npz file:", list(data.keys()))

# print the content inside
for array_name in data:
    print(f"\nArray name: {array_name}")
    print("Array content:\n", data[array_name])
'''

data_class1_features,  data_class1_labels = merge_npz_files(npz_files_frackle)
data_class2_features,  data_class2_labels = merge_npz_files(npz_files_anti)

# get the features and labels
features_class1 = data_class1_features
labels_class1 = data_class1_labels

features_class2 = data_class2_features
labels_class2 = data_class2_labels

# merge features and labels
all_features = np.concatenate((features_class1, features_class2), axis=0)
all_labels = np.concatenate((labels_class1, labels_class2), axis=0)

train_features, test_features, train_labels, test_labels = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# list to numpy
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

model = svm.SVC(kernel='linear')
model.fit(all_features, all_labels)


predictions = model.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#Visualization
plt.scatter(all_features[:, 0], all_features[:, 1], c=all_labels, cmap='viridis')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

legend_labels = ['Class 0', 'Class 1', 'Decision Boundary']
plt.legend(legend_labels)
plt.savefig('SVM_for_frackle.png')
