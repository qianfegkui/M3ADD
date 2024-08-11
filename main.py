import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
from shujuchuli_eeg import loading_data, save_model, load_model
from shujuchuli_emg import loading_data2
from model2 import MULTAV_CLASSFICATIONModel1
#from focalloss import FocalLoss
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, f1_score
import sys
#sys.path.append(r'D:\pycharm\SKF\lstm_attention-p\danmotai_fenlei_one\shiyan_eeg_danshijian_per')

# 检查GPU是否可用，如果不可用，将使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"可用的GPU数量: {num_gpus}")
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"当前GPU: {current_gpu_name}")
else:
    print("GPU不可用。使用CPU。")


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

loss_func = nn.CrossEntropyLoss()
k1 = 0.9
k2 = 0.1
epochs = 200
batch_size = 64

# 创建空列表来存储每次迭代的最佳测试准确度和F1分数
best_test_accuracies = []
best_f1_scores = []
best_model_details = {}  # 用于存储最佳模型的迭代次数和轮次

for iteration in range(1, 21):
    print(f"Starting iteration {iteration}")

    model = MULTAV_CLASSFICATIONModel1().to(device)
#
    saved_model_1 = torch.load(
        r"D:\pycharm\SKF\实验结果\our\eeg-duoshijian\抑郁\per_tongdao\0.005-0.000005-0.8684\best_model_iteration_16_epoch_183.pt.pt")
    state_dict_1 = saved_model_1.state_dict()
    state_dict_eeg = {}

    for key, value in state_dict_1.items():
        new_key = key.split('.', 1)
        new_key = '_eeg.'.join(new_key)
        state_dict_eeg[new_key] = value

    # Example usage:
    saved_model_2 = torch.load(
        r"D:\pycharm\SKF\实验结果\our\emg-duoshijian\抑郁\per-tongdao\0.006-0.000005-0.7368()\best_model_iteration_5_epoch_9.pt.pt")
    state_dict_2 = saved_model_2.state_dict()
    state_dict_emg = {}

    for key, value in state_dict_2.items():
        new_key = key.split('.', 1)
        new_key = '_emg.'.join(new_key)
        state_dict_emg[new_key] = value

    # 获取新模型的状态字典
    model_dict = model.state_dict()

    # 创建一个新的状态字典，包含匹配的权重
    matched_state_dict_1 = {k: v for k, v in state_dict_eeg.items() if
                            k in model_dict and v.size() == model_dict[k].size()}
    matched_state_dict_2 = {k: v for k, v in state_dict_emg.items() if
                            k in model_dict and v.size() == model_dict[k].size()}

    # 更新新模型的状态字典
    model_dict.update(matched_state_dict_1)
    model_dict.update(matched_state_dict_2)

    # 将更新后的状态字典加载到新模型
    model.load_state_dict(model_dict)
    #
    unfrozen_params = [
        "lstm_z_eeg.weight_ih_l0",
        "lstm_z_eeg.weight_hh_l0",
        "lstm_z_eeg.bias_ih_l0",
        "lstm_z_eeg.bias_hh_l0",
        "lstm_z_eeg.weight_ih_l0_reverse",
        "lstm_z_eeg.weight_hh_l0_reverse",
        "lstm_z_eeg.bias_ih_l0_reverse",
        "lstm_z_eeg.bias_hh_l0_reverse",
        "lstm_z_emg.weight_ih_l0",
        "lstm_z_emg.weight_hh_l0",
        "lstm_z_emg.bias_ih_l0",
        "lstm_z_emg.bias_hh_l0",
        "lstm_z_emg.weight_ih_l0_reverse",
        "lstm_z_emg.weight_hh_l0_reverse",
        "lstm_z_emg.bias_ih_l0_reverse",
        "lstm_z_emg.bias_hh_l0_reverse",

        "lstm_b_eeg.weight_ih_l0",
        "lstm_b_eeg.weight_hh_l0",
        "lstm_b_eeg.bias_ih_l0",
        "lstm_b_eeg.bias_hh_l0",
        "lstm_b_eeg.weight_ih_l0_reverse",
        "lstm_b_eeg.weight_hh_l0_reverse",
        "lstm_b_eeg.bias_ih_l0_reverse",
        "lstm_b_eeg.bias_hh_l0_reverse",
        "lstm_b_emg.weight_ih_l0",
        "lstm_b_emg.weight_hh_l0",
        "lstm_b_emg.bias_ih_l0",
        "lstm_b_emg.bias_hh_l0",
        "lstm_b_emg.weight_ih_l0_reverse",
        "lstm_b_emg.weight_hh_l0_reverse",
        "lstm_b_emg.bias_ih_l0_reverse",
        "lstm_b_emg.bias_hh_l0_reverse",

        "lstm_l_eeg.weight_ih_l0",
        "lstm_l_eeg.weight_hh_l0",
        "lstm_l_eeg.bias_ih_l0",
        "lstm_l_eeg.bias_hh_l0",
        "lstm_l_eeg.weight_ih_l0_reverse",
        "lstm_l_eeg.weight_hh_l0_reverse",
        "lstm_l_eeg.bias_ih_l0_reverse",
        "lstm_l_eeg.bias_hh_l0_reverse",
        "lstm_l_emg.weight_ih_l0",
        "lstm_l_emg.weight_hh_l0",
        "lstm_l_emg.bias_ih_l0",
        "lstm_l_emg.bias_hh_l0",
        "lstm_l_emg.weight_ih_l0_reverse",
        "lstm_l_emg.weight_hh_l0_reverse",
        "lstm_l_emg.bias_ih_l0_reverse",
        "lstm_l_emg.bias_hh_l0_reverse",

        "lstm_t_eeg.weight_ih_l0",
        "lstm_t_eeg.weight_hh_l0",
        "lstm_t_eeg.bias_ih_l0",
        "lstm_t_eeg.bias_hh_l0",
        "lstm_t_eeg.weight_ih_l0_reverse",
        "lstm_t_eeg.weight_hh_l0_reverse",
        "lstm_t_eeg.bias_ih_l0_reverse",
        "lstm_t_eeg.bias_hh_l0_reverse",
        "lstm_t_emg.weight_ih_l0",
        "lstm_t_emg.weight_hh_l0",
        "lstm_t_emg.bias_ih_l0",
        "lstm_t_emg.bias_hh_l0",
        "lstm_t_emg.weight_ih_l0_reverse",
        "lstm_t_emg.weight_hh_l0_reverse",
        "lstm_t_emg.bias_ih_l0_reverse",
        "lstm_t_emg.bias_hh_l0_reverse",

        # "MultiheadAttention_eeg.in_proj_weight",
        # "MultiheadAttention_eeg.in_proj_bias",
        # "MultiheadAttention_eeg.out_proj.weight",
        # "MultiheadAttention_eeg.out_proj.bias",
        # "MultiheadAttention_emg.in_proj_weight",
        # "MultiheadAttention_emg.in_proj_bias",
        # "MultiheadAttention_emg.out_proj.weight",
        # "MultiheadAttention_emg.out_proj.bias",

        "cbam_eeg.ChannelGate.mlp.1.weight",
        "cbam_eeg.ChannelGate.mlp.1.bias",
        "cbam_eeg.ChannelGate.mlp.3.weight",
        "cbam_eeg.ChannelGate.mlp.3.bias",
        "cbam_eeg.SpatialGate.spatial.conv.weight",
        "cbam_eeg.SpatialGate.spatial.bn.weight",
        "cbam_eeg.SpatialGate.spatial.bn.bias",

        "cbam_emg.ChannelGate.mlp.1.weight",
        "cbam_emg.ChannelGate.mlp.1.bias",
        "cbam_emg.ChannelGate.mlp.3.weight",
        "cbam_emg.ChannelGate.mlp.3.bias",
        "cbam_emg.SpatialGate.spatial.conv.weight",
        "cbam_emg.SpatialGate.spatial.bn.weight",
        "cbam_emg.SpatialGate.spatial.bn.bias",

    ]
    for name, param in model.named_parameters():
        if name in matched_state_dict_1 or name in matched_state_dict_2:
            if name in unfrozen_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True

    # 打印参数的 requires_grad 状态以验证
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    optimizer = optim.Adam(model.parameters(), lr=0.007, weight_decay=0.000005)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_test = 0
    best_f1 = 0
    best_epoch = 0  # 记录当前迭代中最佳模型的轮次

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0

        for i in tqdm.trange(0, len(x_train_z_eeg), batch_size):
            x_z_eeg = x_train_z_eeg[i:i + batch_size]
            x_b_eeg = x_train_b_eeg[i:i + batch_size]
            x_l_eeg = x_train_l_eeg[i:i + batch_size]
            x_t_eeg = x_train_t_eeg[i:i + batch_size]

            x_z_emg = x_train_z_emg[i:i + batch_size]
            x_b_emg = x_train_b_emg[i:i + batch_size]
            x_l_emg = x_train_l_emg[i:i + batch_size]
            x_t_emg = x_train_t_emg[i:i + batch_size]

            y_batch_fenlei = y_train_eeg[i:i + batch_size]
            y_batch_Ex = y_train_data_Ex_eeg[i:i + batch_size]
            y_batch_Ag = y_train_data_Ag_eeg[i:i + batch_size]
            y_batch_Co = y_train_data_Co_eeg[i:i + batch_size]
            y_batch_Ne = y_train_data_Ne_eeg[i:i + batch_size]
            y_batch_Op = y_train_data_Op_eeg[i:i + batch_size]

            optimizer.zero_grad()
            output_depression, output_Ex, output_Ag, output_Co, output_Ne, output_Op = model(x_z_eeg, x_b_eeg, x_l_eeg, x_t_eeg, x_z_emg, x_b_emg, x_l_emg, x_t_emg)
            loss_depression = loss_func(output_depression, torch.tensor(y_batch_fenlei).to(device))

            loss_EX = loss_func(output_Ex, torch.tensor(y_batch_Ex).to(device))
            loss_Ag = loss_func(output_Ag, torch.tensor(y_batch_Ag).to(device))
            loss_Co = loss_func(output_Co, torch.tensor(y_batch_Co).to(device))
            loss_Ne = loss_func(output_Ne, torch.tensor(y_batch_Ne).to(device))
            loss_Op = loss_func(output_Op, torch.tensor(y_batch_Op).to(device))
            loss_personality = loss_depression + loss_EX + loss_Ag + loss_Co + loss_Ne + loss_Op

            loss = k1 * loss_depression + k2 * loss_personality

            #loss = loss_depression
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            y_batch_classes = torch.argmax(output_depression, dim=1)
            correct_train += (y_batch_classes == torch.tensor(y_batch_fenlei).to(device)).sum().item()

        train_accuracy = correct_train / len(x_train_z_eeg)
        train_accuracies.append(train_accuracy)
        train_loss = total_loss / len(x_train_z_eeg)
        train_losses.append(train_loss)

        with torch.no_grad():
            model.eval()
            y_test_pred_depression, y_test_pred_Ex, y_test_pred_Ag, y_test_pred_Co, y_test_pred_Ne, y_test_pred_Op = model(x_test_z_eeg, x_test_b_eeg, x_test_l_eeg, x_test_t_eeg, x_test_z_emg, x_test_b_emg, x_test_l_emg, x_test_t_emg)
            y_test_pred_classes = torch.argmax(y_test_pred_depression, dim=1)
            correct_test = torch.sum(y_test_pred_classes == torch.tensor(y_test_eeg).to(device))
            total_test = len(y_test_eeg)
            test_accuracy = correct_test.item() / total_test
            test_accuracies.append(test_accuracy)
            test_loss_depression = loss_func(y_test_pred_depression, torch.tensor(y_test_eeg).to(device))
            test_loss_EX = loss_func(y_test_pred_Ex, torch.tensor(y_test_data_Ex_eeg).to(device))
            test_loss_Ag = loss_func(y_test_pred_Ag, torch.tensor(y_test_data_Ag_eeg).to(device))
            test_loss_Co = loss_func(y_test_pred_Co, torch.tensor(y_test_data_Co_eeg).to(device))
            test_loss_Ne = loss_func(y_test_pred_Ne, torch.tensor(y_test_data_Ne_eeg).to(device))
            test_loss_Op = loss_func(y_test_pred_Op, torch.tensor(y_test_data_Op_eeg).to(device))
            test_loss_personality = test_loss_EX + test_loss_Ag + test_loss_Co + test_loss_Ne + test_loss_Op

            test_loss = k1 * test_loss_depression + k2 * test_loss_personality

            test_losses.append(test_loss.item())

            y_test_cpu = torch.tensor(y_test_eeg).cpu().numpy()
            y_test_pred_classes_cpu = y_test_pred_classes.cpu().numpy()
            precision = precision_score(y_test_cpu, y_test_pred_classes_cpu, average=None)
            precision = 0 if any(element <= 0.1 for element in precision) else 1
            f1 = f1_score(y_test_cpu, y_test_pred_classes_cpu, average='weighted')

            print(f'\nEpoch [{epoch + 1}/{epochs}] \n'
                  f'Training Loss:    {train_loss:.4f} - Training Accuracy:    {train_accuracy:.4f}\n'
                  f'Test Loss:        {test_loss:.4f} - Test Accuracy:        {test_accuracy:.4f}\n'
                  f'F1 Score:         {f1:.4f}\n\n')

            if test_accuracy > best_test and precision == 1 :
                model_path = f"best_model_iteration_{iteration}_epoch_{epoch + 1}.pt"
                save_model(model, name=model_path)
                print(f"Saved model at {model_path}!")
                best_test = test_accuracy
                best_f1 = f1
                best_epoch = epoch + 1  # 记录最佳模型的轮次

    best_test_accuracies.append(best_test)
    best_f1_scores.append(best_f1)
    best_model_details[iteration] = (best_epoch, best_test, best_f1)  # 记录当前迭代中最佳模型的迭代次数和轮次

# 计算5次迭代中最佳测试准确度和F1分数的均值和标准差
max_best_accuracy = np.max(best_test_accuracies)
mean_best_accuracy = np.mean(best_test_accuracies)
std_best_accuracy = np.std(best_test_accuracies)
mean_best_f1 = np.mean(best_f1_scores)
std_best_f1 = np.std(best_f1_scores)
print(f"20次迭代中最佳测试准确度的最大值: {max_best_accuracy:.4f}")
print(f"20次迭代中最佳测试准确度的均值: {mean_best_accuracy:.4f}")
print(f"20次迭代中最佳测试准确度的标准差: {std_best_accuracy:.4f}")
print(f"20次迭代中最佳F1分数的均值: {mean_best_f1:.4f}")
print(f"20次迭代中最佳F1分数的标准差: {std_best_f1:.4f}")

# 打印出5次迭代的最好模型的次数和轮次
for iteration, (epoch, accuracy, f1) in best_model_details.items():
    print(f"Iteration {iteration} had the best model at epoch {epoch} with accuracy {accuracy:.4f} and f1 {f1:.4f}")
