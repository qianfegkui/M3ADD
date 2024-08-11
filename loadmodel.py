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
# 检查GPU是否可用，如果不可用，将使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"可用的GPU数量: {num_gpus}")
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"当前GPU: {current_gpu_name}")
else:
    print("GPU不可用。使用CPU。")






# def visualize(features, labels, folder="nogenderticks", filename="tsne_visualization", dpi=800):
#     def plot_tsne(features, labels, ax, color_dict):
#         from sklearn.manifold import TSNE
#         from matplotlib.colors import ListedColormap
#         tsne = TSNE(n_components=2, random_state=42)
#         tsne_results = tsne.fit_transform(features)
#         cmap = ListedColormap([color_dict[0], color_dict[1], color_dict[2]])
#         scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.9)
#         #ax.tick_params(axis='both', which='major', labelsize=23)  # 设置刻度大小
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # 创建图例
#         handles = [
#             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[0], markersize=10, label='0'),
#             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[1], markersize=10, label='1'),
#             plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[2], markersize=10, label='2')
#         ]
#         ax.legend(handles=handles, title='Categories')
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     color_dict = {0: '#9D9ECD', 1: '#E68D3D', 2: '#4DBD33'}
#     fig, ax = plt.subplots(figsize=(5, 5))
#
#     # 检查数据类型并适当转换
#     if isinstance(features, torch.Tensor):
#         features = features.detach().cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.detach().cpu().numpy()
#
#     plot_tsne(features, labels, ax, color_dict)
#     plt.tight_layout()
#
#     # 保存为 JPG 和 PDF 格式
#     plt.savefig(os.path.join(folder, f"{filename}.jpg"), dpi=dpi)
#     plt.savefig(os.path.join(folder, f"{filename}.pdf"), dpi=dpi)
#     plt.show()


def visualize(features, labels, folder="nogenderticks", filename="tsne_visualization", dpi=800):
    def plot_tsne(features, labels, ax, color_dict):
        # 调整 t-SNE 参数，降低 perplexity 以增强聚拢效果
        tsne = TSNE(n_components=2, perplexity=6, learning_rate=500, n_iter=1000, random_state=42)
        tsne_results = tsne.fit_transform(features)

        unique_labels = np.unique(labels)

        # 为每个类别添加凸包并填充背景色
        def plot_convex_hull(points, color):
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5)
            plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.3)

        for label in unique_labels:
            points = tsne_results[labels == label]
            plot_convex_hull(points, color_dict[label])

        # 绘制类别数据点的散点图，放在凸包之后，以确保颜色清晰
        cmap = ListedColormap([color_dict[0], color_dict[1], color_dict[2]])
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.9, s=50, edgecolor='black')

        # 设置背景色和边界
        ax.set_facecolor('#f7f7f7')  # 背景色
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # 隐藏刻度
        ax.set_xticks([])
        ax.set_yticks([])

        # 创建图例
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[0], markersize=10, label='0'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[1], markersize=10, label='1'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[2], markersize=10, label='2')
        ]
        ax.legend(handles=handles, title='Categories')

    if not os.path.exists(folder):
        os.makedirs(folder)

    color_dict = {0: '#9D9ECD', 1: '#E68D3D', 2: '#4DBD33'}
    fig, ax = plt.subplots(figsize=(5, 5))

    # 检查数据类型并适当转换
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    plot_tsne(features, labels, ax, color_dict)
    plt.tight_layout()

    # 保存为 JPG 和 PDF 格式
    plt.savefig(os.path.join(folder, f"{filename}.jpg"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(folder, f"{filename}.pdf"), dpi=dpi, bbox_inches='tight')
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
(x_train_z_eeg, x_train_b_eeg, x_train_l_eeg, x_train_t_eeg, x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, y_train_eeg, y_test_eeg, y_train_data_Ex_eeg, y_test_data_Ex_eeg, y_train_data_Ag_eeg,
y_test_data_Ag_eeg, y_train_data_Co_eeg, y_test_data_Co_eeg, y_train_data_Ne_eeg, y_test_data_Ne_eeg, y_train_data_Op_eeg, y_test_data_Op_eeg) = loading_data()

(x_train_z_emg, x_train_b_emg, x_train_l_emg, x_train_t_emg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg, y_train_emg, y_test_emg, y_train_data_Ex_emg, y_test_data_Ex_emg, y_train_data_Ag_emg,
y_test_data_Ag_emg, y_train_data_Co_emg, y_test_data_Co_emg, y_train_data_Ne_emg, y_test_data_Ne_emg, y_train_data_Op_emg, y_test_data_Op_emg) = loading_data2()

# print("y_test_emg", y_test_emg.shape)
# y_test_emg = torch.tensor(y_test_emg, dtype=torch.float32).to(device)
#
# save_dir = r"D:\pycharm\SKF\实验结果\our\tongdaozhuyilifen\label\yiyu"
#
# # 文件名
# file_name = 'y_test.npy'
#
# # 完整路径
# full_path = os.path.join(save_dir, file_name)
#
# # 将 PyTorch 张量转换为 NumPy 数组
# numpy_array = y_test_emg.cpu().numpy()
#
# # 将 NumPy 数组保存为 .npy 文件
# np.save(full_path, numpy_array)



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

# 加载已保存的模型权重
saved_model = torch.load(
        r"best_model_iteration_10_epoch_2.pt")

# 提取状态字典
state_dict = saved_model.state_dict()

# 获取新模型的状态字典
model_dict = model.state_dict()

# 创建一个新的状态字典，包含匹配的权重
matched_state_dict = {k: v for k, v in state_dict.items() if
                    k in model_dict and v.size() == model_dict[k].size()}

# 更新新模型的状态字典
model_dict.update(matched_state_dict)
# 将更新后的状态字典加载到新模型
model.load_state_dict(model_dict)
for name, param in model.named_parameters():
    if name in matched_state_dict:
        param.requires_grad = False
    else:
        param.requires_grad = False


# 打印参数的 requires_grad 状态以验证
for name, param in model.named_parameters():
    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

model.eval()
#加载整个模型对象
with torch.no_grad():
    y_test_pred_depression, y_test_pred_Ex, y_test_pred_Ag, y_test_pred_Co, y_test_pred_Ne, y_test_pred_Op, x = model(x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg)
    print("output_depression", y_test_pred_depression.shape)
    print("x", x.shape)
    print("label", y_test_eeg.shape)
    visualize(x, y_test_eeg, folder="test", filename="test", dpi=800)
    y_test_pred_classes = torch.argmax(y_test_pred_depression, dim=1)

# 打印分类报告
print("Classification Report:")
report = classification_report(y_test_eeg, y_test_pred_classes.cpu().numpy(), output_dict=True)
print(classification_report(y_test_eeg, y_test_pred_classes.cpu().numpy()))

# 计算准确率
accuracy = (y_test_pred_classes.cpu().numpy() == y_test_eeg).mean()
print("Accuracy:", accuracy)

# 打印F1分数
f1 = f1_score(y_test_eeg, y_test_pred_classes.cpu().numpy(), average='weighted')
print("F1 Score (Weighted):", f1)

# 生成混淆矩阵并可视化
conf_matrix_test = confusion_matrix(y_test_eeg, y_test_pred_classes.cpu().numpy())
conf_matrix_test_normalized = conf_matrix_test.astype('float') / conf_matrix_test.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(max(6, len(conf_matrix_test)*0.7), max(6, len(conf_matrix_test)*0.7)))
sns.heatmap(conf_matrix_test_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()