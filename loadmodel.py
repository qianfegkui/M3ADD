import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE

from dataprc_eeg_fun import loading_data, save_model, load_model
from dataprc_emg_fun import loading_data2, load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import os
import matplotlib.pyplot as plt
import torch
from model2 import MULTAV_CLASSFICATIONModel1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

(x_train_z_eeg, x_train_b_eeg, x_train_l_eeg, x_train_t_eeg, x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, y_train_eeg, y_test_eeg, y_train_data_Ex_eeg, y_test_data_Ex_eeg, y_train_data_Ag_eeg,
y_test_data_Ag_eeg, y_train_data_Co_eeg, y_test_data_Co_eeg, y_train_data_Ne_eeg, y_test_data_Ne_eeg, y_train_data_Op_eeg, y_test_data_Op_eeg) = loading_data()

(x_train_z_emg, x_train_b_emg, x_train_l_emg, x_train_t_emg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg, y_train_emg, y_test_emg, y_train_data_Ex_emg, y_test_data_Ex_emg, y_train_data_Ag_emg,
y_test_data_Ag_emg, y_train_data_Co_emg, y_test_data_Co_emg, y_train_data_Ne_emg, y_test_data_Ne_emg, y_train_data_Op_emg, y_test_data_Op_emg) = loading_data2()


x_train_z_eeg = torch.tensor(x_train_z_eeg, dtype=torch.float32).to(device)
x_test_z_eeg = torch.tensor(x_test_z_eeg, dtype=torch.float32).to(device)
x_train_b_eeg = torch.tensor(x_train_b_eeg, dtype=torch.float32).to(device)
x_test_b_eeg = torch.tensor(x_test_b_eeg, dtype=torch.float32).to(device)
x_train_l_eeg = torch.tensor(x_train_l_eeg, dtype=torch.float32).to(device)
x_test_l_eeg = torch.tensor(x_test_l_eeg, dtype=torch.float32).to(device)
x_train_t_eeg = torch.tensor(x_train_t_eeg, dtype=torch.float32).to(device)
x_test_t_eeg = torch.tensor(x_test_t_eeg, dtype=torch.float32).to(device)

x_train_z_emg = torch.tensor(x_train_z_emg, dtype=torch.float32).to(device)
x_test_z_emg = torch.tensor(x_test_z_emg, dtype=torch.float32).to(device)
x_train_b_emg = torch.tensor(x_train_b_emg, dtype=torch.float32).to(device)
x_test_b_emg = torch.tensor(x_test_b_emg, dtype=torch.float32).to(device)
x_train_l_emg = torch.tensor(x_train_l_emg, dtype=torch.float32).to(device)
x_test_l_emg = torch.tensor(x_test_l_emg, dtype=torch.float32).to(device)
x_train_t_emg = torch.tensor(x_train_t_emg, dtype=torch.float32).to(device)
x_test_t_emg = torch.tensor(x_test_t_emg, dtype=torch.float32).to(device)

model = MULTAV_CLASSFICATIONModel1().to(device)

saved_model = torch.load(
        r"best_model_iteration_10_epoch_2.pt")


state_dict = saved_model.state_dict()


model_dict = model.state_dict()

matched_state_dict = {k: v for k, v in state_dict.items() if
                    k in model_dict and v.size() == model_dict[k].size()}

model_dict.update(matched_state_dict)

model.load_state_dict(model_dict)
for name, param in model.named_parameters():
    if name in matched_state_dict:
        param.requires_grad = False
    else:
        param.requires_grad = False


for name, param in model.named_parameters():
    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

model.eval()

with torch.no_grad():
    y_test_pred_depression, y_test_pred_Ex, y_test_pred_Ag, y_test_pred_Co, y_test_pred_Ne, y_test_pred_Op, x = model(x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg)
    print("output_depression", y_test_pred_depression.shape)
    print("x", x.shape)
    print("label", y_test_eeg.shape)
    visualize(x, y_test_eeg, folder="test", filename="test", dpi=800)
    y_test_pred_classes = torch.argmax(y_test_pred_depression, dim=1)


print("Classification Report:")
report = classification_report(y_test_eeg, y_test_pred_classes.cpu().numpy(), output_dict=True)
print(classification_report(y_test_eeg, y_test_pred_classes.cpu().numpy()))


accuracy = (y_test_pred_classes.cpu().numpy() == y_test_eeg).mean()
print("Accuracy:", accuracy)

f1 = f1_score(y_test_eeg, y_test_pred_classes.cpu().numpy(), average='weighted')
print("F1 Score (Weighted):", f1)

conf_matrix_test = confusion_matrix(y_test_eeg, y_test_pred_classes.cpu().numpy())
conf_matrix_test_normalized = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(max(6, len(conf_matrix_test)*0.7), max(6, len(conf_matrix_test)*0.7)))
sns.heatmap(conf_matrix_test_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()