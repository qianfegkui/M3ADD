import torch.nn.init
import os
import pandas as pd
import numpy as np
from scipy.special import softmax

def save_model(model, name=''):

    save_dir = 'pre_trained_models'

    # 检测目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, f'{save_dir}/{name}.pt')

def load_model(name=''):
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


def split_and_pad(sequence, num_rows):
    """
    Split the sequence into samples of fixed number of rows. If the sample rows are less than the fixed rows, pad with zeros.
    If the sample rows are greater than the fixed rows, handle the remaining rows appropriately.
    """
    samples = []
    num_total_rows = len(sequence)
    num_columns = sequence.shape[1]

    if num_total_rows > num_rows:
        start = 0
        while start < num_total_rows:
            end = start + num_rows
            if end <= num_total_rows:
                samples.append(sequence[start:end])
                start = end
            else:
                sample = sequence[-num_rows:]
                samples.append(sample)
                break
    else:
        padding_needed = num_rows - num_total_rows
        padding = np.zeros((padding_needed, num_columns))
        sample_padded = np.vstack((sequence, padding))
        samples.append(sample_padded)

    return samples


def balance_lists(samples_z, samples_b, samples_l, samples_t):
    # 找出四个列表中元素个数最多的那个
    max_length = max(len(samples_z), len(samples_b), len(samples_l), len(samples_t))

    # 定义一个函数来扩展列表到指定的长度
    def extend_list(lst, target_length):
        original_length = len(lst)
        # 通过循环复制原始列表中的元素直到达到目标长度
        # 通过循环复制原始列表中的元素直到达到目标长度
        while len(lst) < target_length:
            lst.extend(lst[:target_length - len(lst)])

    # 扩展其他列表到max_length
    if len(samples_z) < max_length:
        extend_list(samples_z, max_length)
    if len(samples_b) < max_length:
        extend_list(samples_b, max_length)
    if len(samples_l) < max_length:
        extend_list(samples_l, max_length)
    if len(samples_t) < max_length:
        extend_list(samples_t, max_length)


    return samples_z, samples_b, samples_l, samples_t



def add_gaussian_noise(X, noise_mean=0, noise_std=0.1):
    """
    给输入的数组 X 添加高斯噪声。

    参数:
    X: numpy 数组，要添加噪声的数组，维度为 (样本数, 维度1, 维度2)。
    noise_mean: float，高斯噪声的均值，默认为 0。
    noise_std: float，高斯噪声的标准差，默认为 0.1。

    返回:
    添加了高斯噪声的数组。
    """
    sample_dim, dim_1, dim_2 = X.shape
    # 复制输入数组以避免修改原始数据
    noisy_X = X.copy()

    for sample_index in range(sample_dim):
        # 随机选择起始位置
        start_index = np.random.randint(0, dim_1)
        # 随机选择连续段长度
        segment_length = np.random.randint(1, dim_1 - start_index + 1)
        # 生成噪声并添加
        noise = np.random.normal(noise_mean, noise_std, size=(segment_length, dim_2))
        noisy_X[sample_index, start_index:start_index+segment_length, :] += noise

    return noisy_X


def interleave_and_merge(data1, data2, data3, labels0,labels0_1,labels0_2, labels0_3,labels0_4,labels0_5,
                         labels1,labels1_1,labels1_2, labels1_3,labels1_4,labels1_5,
                         labels2,labels2_1,labels2_2, labels2_3,labels2_4,labels2_5):
    # 确定样本数
    len1, len2, len3 = len(data1), len(data2), len(data3)
    total_samples = len1 + len2 + len3

    # 初始化最终的数据和标签数组
    merged_data = []
    merged_labels = []
    merged_labels_1 = []
    merged_labels_2 = []
    merged_labels_3 = []
    merged_labels_4 = []
    merged_labels_5 = []

    # 依次从三个数组中取样本
    indices1, indices2, indices3 = 0, 0, 0
    while indices1 < len1 or indices2 < len2 or indices3 < len3:
        if indices1 < len1:
            merged_data.append(data1[indices1])
            merged_labels.append(labels0[indices1])
            merged_labels_1.append(labels0_1[indices1])
            merged_labels_2.append(labels0_2[indices1])
            merged_labels_3.append(labels0_3[indices1])
            merged_labels_4.append(labels0_4[indices1])
            merged_labels_5.append(labels0_5[indices1])
            indices1 += 1
        if indices2 < len2:
            merged_data.append(data2[indices2])
            merged_labels.append(labels1[indices2])
            merged_labels_1.append(labels1_1[indices2])
            merged_labels_2.append(labels1_2[indices2])
            merged_labels_3.append(labels1_3[indices2])
            merged_labels_4.append(labels1_4[indices2])
            merged_labels_5.append(labels1_5[indices2])
            indices2 += 1
        if indices3 < len3:
            merged_data.append(data3[indices3])
            merged_labels.append(labels2[indices3])
            merged_labels_1.append(labels2_1[indices3])
            merged_labels_2.append(labels2_2[indices3])
            merged_labels_3.append(labels2_3[indices3])
            merged_labels_4.append(labels2_4[indices3])
            merged_labels_5.append(labels2_5[indices3])
            indices3 += 1

    # 转换为numpy数组
    merged_data = np.array(merged_data)
    merged_labels = np.array(merged_labels)
    merged_labels_1 = np.array(merged_labels_1)
    merged_labels_2 = np.array(merged_labels_2)
    merged_labels_3 = np.array(merged_labels_3)
    merged_labels_4 = np.array(merged_labels_4)
    merged_labels_5 = np.array(merged_labels_5)

    return merged_data, merged_labels, merged_labels_1, merged_labels_2, merged_labels_3, merged_labels_4, merged_labels_5

def interleave_and_merge2(data1, data2, data3, labels1, labels2, labels3):
    # 确定样本数
    len1, len2, len3 = len(data1), len(data2), len(data3)
    total_samples = len1 + len2 + len3

    # 初始化最终的数据和标签数组
    merged_data = []
    merged_labels = []

    # 依次从三个数组中取样本
    indices1, indices2, indices3 = 0, 0, 0
    while indices1 < len1 or indices2 < len2 or indices3 < len3:
        if indices1 < len1:
            merged_data.append(data1[indices1])
            merged_labels.append(labels1[indices1])
            indices1 += 1
        if indices2 < len2:
            merged_data.append(data2[indices2])
            merged_labels.append(labels2[indices2])
            indices2 += 1
        if indices3 < len3:
            merged_data.append(data3[indices3])
            merged_labels.append(labels3[indices3])
            indices3 += 1

    # 转换为numpy数组
    merged_data = np.array(merged_data)
    merged_labels = np.array(merged_labels)

    return merged_data, merged_labels


def shujuzengqiang(X_train_padded2, y_train_fenlei):
    # 分别初始化三个空列表来存储不同标签的数据和标签
    X_train_class_0 = []
    X_train_class_1 = []
    X_train_class_2 = []

    y_train_class_0 = []
    y_train_class_1 = []
    y_train_class_2 = []
    # 遍历所有数据和标签，根据标签将数据和标签分别存入对应的列表中
    for data, label in zip(X_train_padded2, y_train_fenlei):
        if label == 0:
            X_train_class_0.append(data)
            y_train_class_0.append(label)
        elif label == 1:
            X_train_class_1.append(data)
            y_train_class_1.append(label)
        elif label == 2:
            X_train_class_2.append(data)
            y_train_class_2.append(label)

    # 将列表转换为numpy数组
    X_train_class_0 = np.array(X_train_class_0)
    X_train_class_1 = np.array(X_train_class_1)
    X_train_class_2 = np.array(X_train_class_2)

    y_train_class_0 = np.array(y_train_class_0)
    y_train_class_1 = np.array(y_train_class_1)
    y_train_class_2 = np.array(y_train_class_2)

    # 复制 X_train_class_0 并添加高斯噪声
    # X_train_class_0_noisy = X_train_class_0 + np.random.normal(0, 0.1, X_train_class_0.shape)
    X_train_class_0_noisy = add_gaussian_noise(X_train_class_0)
    # 合并原始数据和添加噪声后的数据
    X_train_class_0_augmented = np.concatenate((X_train_class_0, X_train_class_0_noisy), axis=0)
    y_train_class_0_augmented = np.concatenate((y_train_class_0, y_train_class_0), axis=0)

    # 计算样本数差异
    num_samples_0 = X_train_class_0_augmented.shape[0]
    num_samples_1 = X_train_class_1.shape[0]
    num_samples_2 = X_train_class_2.shape[0]

    diff_1 = num_samples_0 - num_samples_1
    diff_2 = num_samples_0 - num_samples_2

    # 计算需要复制的倍数
    replicate_factor_1 = diff_1 / num_samples_1
    replicate_factor_2 = diff_2 / num_samples_2

    # 复制并添加噪声
    X_train_class_1_replicated = np.tile(X_train_class_1, (int(replicate_factor_1), 1, 1))
    X_train_class_2_replicated = np.tile(X_train_class_2, (int(replicate_factor_2), 1, 1))

    # 处理余数部分
    remainder_1 = X_train_class_1[:int(replicate_factor_1 % 1 * num_samples_1)]
    remainder_2 = X_train_class_2[:int(replicate_factor_2 % 1 * num_samples_2)]

    X_train_class_1_augmented = np.concatenate((X_train_class_1_replicated, remainder_1), axis=0)
    X_train_class_2_augmented = np.concatenate((X_train_class_2_replicated, remainder_2), axis=0)

    X_train_class_1_noisy = add_gaussian_noise(X_train_class_1_augmented)
    X_train_class_2_noisy = add_gaussian_noise(X_train_class_2_augmented)

    # 合并原始数据和添加噪声后的数据
    X_train_class_1_final = np.concatenate((X_train_class_1, X_train_class_1_noisy), axis=0)
    X_train_class_2_final = np.concatenate((X_train_class_2, X_train_class_2_noisy), axis=0)

    # 更新标签
    y_train_class_1_final = np.hstack((y_train_class_1, [1] * X_train_class_1_noisy.shape[0]))
    y_train_class_2_final = np.hstack((y_train_class_2, [2] * X_train_class_2_noisy.shape[0]))

    X_train, Y_train = interleave_and_merge2(X_train_class_0_augmented, X_train_class_1_final, X_train_class_2_final,
                                            y_train_class_0_augmented, y_train_class_1_final, y_train_class_2_final)
    return X_train, Y_train





def loading_data2():
    """
    Load data from specified paths, apply column-wise softmax normalization,
    and return train/test data and labels after splitting and padding.
    """
    # Specified paths and parameters
    data_folder_path = r"D:\实验数据\jidian-97\PHQ_9_2"
    csv_file_path = r"D:\实验数据\实验结果.csv"
    normalization = True
    num_rows_z = 88
    num_rows_b = 36
    num_rows_l = 107
    num_rows_t = 28

    # Load CSV file with ID and label3 columns, skipping the first row
    csv_data = pd.read_csv(csv_file_path, encoding='gbk')
    ids = csv_data['ID'].values
    labels = csv_data['Label-PHQ_9'].values
    labels_Ex = csv_data['Label-Extraversion'].values
    labels_Ag = csv_data['Label-Agreeableness'].values
    labels_Co = csv_data['Label-Conscientiousness'].values
    labels_Ne = csv_data['Label-Neuroticism'].values
    labels_Op = csv_data['Label-Openness'].values

    # Initialize lists to store data and labels
    X_train_data_z = []
    X_test_data_z = []
    X_train_data_b = []
    X_test_data_b = []
    X_train_data_l = []
    X_test_data_l = []
    X_train_data_t = []
    X_test_data_t = []
    y_train_data = []
    y_test_data = []
    y_train_data_Ex = []
    y_test_data_Ex = []
    y_train_data_Ag = []
    y_test_data_Ag = []
    y_train_data_Co = []
    y_test_data_Co = []
    y_train_data_Ne = []
    y_test_data_Ne = []
    y_train_data_Op = []
    y_test_data_Op = []

    # Load data from train and test folders
    for folder_name in ['train', 'test']:
        folder_path = os.path.join(data_folder_path, folder_name)
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Load data from 4.csv
                csv_file_z = os.path.join(subfolder_path, '1.csv')
                df_z = pd.read_csv(csv_file_z, header=None)

                csv_file_b = os.path.join(subfolder_path, '2.csv')
                df_b = pd.read_csv(csv_file_b, header=None)

                csv_file_l = os.path.join(subfolder_path, '3.csv')
                df_l = pd.read_csv(csv_file_l, header=None)

                csv_file_t = os.path.join(subfolder_path, '4.csv')
                df_t = pd.read_csv(csv_file_t, header=None)

                # Apply softmax normalization column-wise if required
                if normalization:
                    df_normalized_z = df_z.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
                else:
                    df_normalized_z = df_z

                if normalization:
                    df_normalized_b = df_b.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
                else:
                    df_normalized_b = df_b

                if normalization:
                    df_normalized_l = df_l.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
                else:
                    df_normalized_l = df_l

                if normalization:
                    df_normalized_t = df_t.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
                else:
                    df_normalized_t = df_t

                # Convert the normalized DataFrame to numpy array
                data_array_z = df_normalized_z.values
                data_array_b = df_normalized_b.values
                data_array_l = df_normalized_l.values
                data_array_t = df_normalized_t.values
                #data_array = df.values

                # Split and pad the data
                samples_z = split_and_pad(data_array_z, num_rows_z)
                samples_b = split_and_pad(data_array_b, num_rows_b)
                samples_l = split_and_pad(data_array_l, num_rows_l)
                samples_t = split_and_pad(data_array_t, num_rows_t)

                samples_z, samples_b, samples_l, samples_t = balance_lists(samples_z, samples_b, samples_l, samples_t)

                try:
                    index = ids.tolist().index(int(subfolder_name))
                    current_labels = (
                    labels[index], labels_Ex[index], labels_Ag[index], labels_Co[index], labels_Ne[index],
                    labels_Op[index])
                except ValueError:
                    print(f"ID {subfolder_name} not found in CSV data.")
                    continue

                # Determine whether to add to train or test
                if folder_name == 'train':
                    X_train_data_z.extend(samples_z)
                    X_train_data_b.extend(samples_b)
                    X_train_data_l.extend(samples_l)
                    X_train_data_t.extend(samples_t)
                    y_train_data.extend([current_labels[0]] * len(samples_z))
                    y_train_data_Ex.extend([current_labels[1]] * len(samples_z))
                    y_train_data_Ag.extend([current_labels[2]] * len(samples_z))
                    y_train_data_Co.extend([current_labels[3]] * len(samples_z))
                    y_train_data_Ne.extend([current_labels[4]] * len(samples_z))
                    y_train_data_Op.extend([current_labels[5]] * len(samples_z))
                else:
                    X_test_data_z.extend(samples_z)
                    X_test_data_b.extend(samples_b)
                    X_test_data_l.extend(samples_l)
                    X_test_data_t.extend(samples_t)
                    y_test_data.extend([current_labels[0]] * len(samples_z))
                    y_test_data_Ex.extend([current_labels[1]] * len(samples_z))
                    y_test_data_Ag.extend([current_labels[2]] * len(samples_z))
                    y_test_data_Co.extend([current_labels[3]] * len(samples_z))
                    y_test_data_Ne.extend([current_labels[4]] * len(samples_z))
                    y_test_data_Op.extend([current_labels[5]] * len(samples_z))

    # Convert lists to numpy arrays
    X_train_z = np.array(X_train_data_z)
    X_test_z = np.array(X_test_data_z)
    X_train_b = np.array(X_train_data_b)
    X_test_b = np.array(X_test_data_b)
    X_train_l = np.array(X_train_data_l)
    X_test_l = np.array(X_test_data_l)
    X_train_t = np.array(X_train_data_t)
    X_test_t = np.array(X_test_data_t)
    y_train_fenlei = np.array(y_train_data)
    y_test_fenlei = np.array(y_test_data)
    y_train_data_Ex = np.array(y_train_data_Ex)
    y_test_data_Ex = np.array(y_test_data_Ex)
    y_train_data_Ag = np.array(y_train_data_Ag)
    y_test_data_Ag = np.array(y_test_data_Ag)
    y_train_data_Co = np.array(y_train_data_Co)
    y_test_data_Co = np.array(y_test_data_Co)
    y_train_data_Ne = np.array(y_train_data_Ne)
    y_test_data_Ne = np.array(y_test_data_Ne)
    y_train_data_Op = np.array(y_train_data_Op)
    y_test_data_Op = np.array(y_test_data_Op)

    # 分别初始化三个空列表来存储不同标签的数据和标签
    X_train_class_0 = []
    X_train_class_1 = []
    X_train_class_2 = []

    y_train_class_0 = []
    y_train_class_0_Ex = []
    y_train_class_0_Ag = []
    y_train_class_0_Co = []
    y_train_class_0_Ne = []
    y_train_class_0_Op = []
    y_train_class_1 = []
    y_train_class_1_Ex = []
    y_train_class_1_Ag = []
    y_train_class_1_Co = []
    y_train_class_1_Ne = []
    y_train_class_1_Op = []
    y_train_class_2 = []
    y_train_class_2_Ex = []
    y_train_class_2_Ag = []
    y_train_class_2_Co = []
    y_train_class_2_Ne = []
    y_train_class_2_Op = []
    # 遍历所有数据和标签，根据标签将数据和标签分别存入对应的列表中
    for data, label, label_Ex, label_Ag, label_Co, label_Ne, label_Op in zip(X_train_z, y_train_fenlei,
                                                                             y_train_data_Ex, y_train_data_Ag,
                                                                             y_train_data_Co, y_train_data_Ne,
                                                                             y_train_data_Op):
        if label == 0:
            X_train_class_0.append(data)
            y_train_class_0.append(label)
            y_train_class_0_Ex.append(label_Ex)
            y_train_class_0_Ag.append(label_Ag)
            y_train_class_0_Co.append(label_Co)
            y_train_class_0_Ne.append(label_Ne)
            y_train_class_0_Op.append(label_Op)

        elif label == 1:
            X_train_class_1.append(data)
            y_train_class_1.append(label)
            y_train_class_1_Ex.append(label_Ex)
            y_train_class_1_Ag.append(label_Ag)
            y_train_class_1_Co.append(label_Co)
            y_train_class_1_Ne.append(label_Ne)
            y_train_class_1_Op.append(label_Op)
        elif label == 2:
            X_train_class_2.append(data)
            y_train_class_2.append(label)
            y_train_class_2_Ex.append(label_Ex)
            y_train_class_2_Ag.append(label_Ag)
            y_train_class_2_Co.append(label_Co)
            y_train_class_2_Ne.append(label_Ne)
            y_train_class_2_Op.append(label_Op)

    # 将列表转换为numpy数组
    X_train_class_0 = np.array(X_train_class_0)
    X_train_class_1 = np.array(X_train_class_1)
    X_train_class_2 = np.array(X_train_class_2)

    y_train_class_0 = np.array(y_train_class_0)
    y_train_class_0_Ex = np.array(y_train_class_0_Ex)
    y_train_class_0_Ag = np.array(y_train_class_0_Ag)
    y_train_class_0_Co = np.array(y_train_class_0_Co)
    y_train_class_0_Ne = np.array(y_train_class_0_Ne)
    y_train_class_0_Op = np.array(y_train_class_0_Op)
    y_train_class_1 = np.array(y_train_class_1)
    y_train_class_1_Ex = np.array(y_train_class_1_Ex)
    y_train_class_1_Ag = np.array(y_train_class_1_Ag)
    y_train_class_1_Co = np.array(y_train_class_1_Co)
    y_train_class_1_Ne = np.array(y_train_class_1_Ne)
    y_train_class_1_Op = np.array(y_train_class_1_Op)
    y_train_class_2 = np.array(y_train_class_2)
    y_train_class_2_Ex = np.array(y_train_class_2_Ex)
    y_train_class_2_Ag = np.array(y_train_class_2_Ag)
    y_train_class_2_Co = np.array(y_train_class_2_Co)
    y_train_class_2_Ne = np.array(y_train_class_2_Ne)
    y_train_class_2_Op = np.array(y_train_class_2_Op)

    # 复制 X_train_class_0 并添加高斯噪声
    # X_train_class_0_noisy = X_train_class_0 + np.random.normal(0, 0.1, X_train_class_0.shape)
    X_train_class_0_noisy = add_gaussian_noise(X_train_class_0)
    # 合并原始数据和添加噪声后的数据
    X_train_class_0_augmented = np.concatenate((X_train_class_0, X_train_class_0_noisy), axis=0)
    y_train_class_0_augmented = np.concatenate((y_train_class_0, y_train_class_0), axis=0)
    y_train_class_0_Ex = np.concatenate((y_train_class_0_Ex, y_train_class_0_Ex), axis=0)
    y_train_class_0_Ag = np.concatenate((y_train_class_0_Ag, y_train_class_0_Ag), axis=0)
    y_train_class_0_Co = np.concatenate((y_train_class_0_Co, y_train_class_0_Co), axis=0)
    y_train_class_0_Ne = np.concatenate((y_train_class_0_Ne, y_train_class_0_Ne), axis=0)
    y_train_class_0_Op = np.concatenate((y_train_class_0_Op, y_train_class_0_Op), axis=0)

    # 计算样本数差异
    num_samples_0 = X_train_class_0_augmented.shape[0]
    num_samples_1 = X_train_class_1.shape[0]
    num_samples_2 = X_train_class_2.shape[0]

    diff_1 = num_samples_0 - num_samples_1
    diff_2 = num_samples_0 - num_samples_2

    # 计算需要复制的倍数
    replicate_factor_1 = diff_1 / num_samples_1
    replicate_factor_2 = diff_2 / num_samples_2

    # 复制并添加噪声
    X_train_class_1_replicated = np.tile(X_train_class_1, (int(replicate_factor_1), 1, 1))
    X_train_class_2_replicated = np.tile(X_train_class_2, (int(replicate_factor_2), 1, 1))
    y_train_class_1_replicated = np.tile(y_train_class_1, int(replicate_factor_1))
    y_train_class_2_replicated = np.tile(y_train_class_2, int(replicate_factor_2))
    y_train_class_1_Ex_replicated = np.tile(y_train_class_1_Ex, int(replicate_factor_1))
    y_train_class_1_Ag_replicated = np.tile(y_train_class_1_Ag, int(replicate_factor_1))
    y_train_class_1_Co_replicated = np.tile(y_train_class_1_Co, int(replicate_factor_1))
    y_train_class_1_Ne_replicated = np.tile(y_train_class_1_Ne, int(replicate_factor_1))
    y_train_class_1_Op_replicated = np.tile(y_train_class_1_Op, int(replicate_factor_1))
    y_train_class_2_Ex_replicated = np.tile(y_train_class_2_Ex, int(replicate_factor_2))
    y_train_class_2_Ag_replicated = np.tile(y_train_class_2_Ag, int(replicate_factor_2))
    y_train_class_2_Co_replicated = np.tile(y_train_class_2_Co, int(replicate_factor_2))
    y_train_class_2_Ne_replicated = np.tile(y_train_class_2_Ne, int(replicate_factor_2))
    y_train_class_2_Op_replicated = np.tile(y_train_class_2_Op, int(replicate_factor_2))

    # 处理余数部分
    remainder_1 = X_train_class_1[:int(replicate_factor_1 % 1 * num_samples_1)]
    remainder_2 = X_train_class_2[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1 = y_train_class_1[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2 = y_train_class_2[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1_Ex = y_train_class_1_Ex[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2_Ex = y_train_class_2_Ex[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1_Ag = y_train_class_1_Ag[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2_Ag = y_train_class_2_Ag[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1_Co = y_train_class_1_Co[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2_Co = y_train_class_2_Co[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1_Ne = y_train_class_1_Ne[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2_Ne = y_train_class_2_Ne[:int(replicate_factor_2 % 1 * num_samples_2)]
    y_remainder_1_Op = y_train_class_1_Op[:int(replicate_factor_1 % 1 * num_samples_1)]
    y_remainder_2_Op = y_train_class_2_Op[:int(replicate_factor_2 % 1 * num_samples_2)]

    X_train_class_1_augmented = np.concatenate((X_train_class_1_replicated, remainder_1), axis=0)
    X_train_class_2_augmented = np.concatenate((X_train_class_2_replicated, remainder_2), axis=0)
    y_train_class_1_augmented = np.concatenate((y_train_class_1_replicated, y_remainder_1), axis=0)
    y_train_class_2_augmented = np.concatenate((y_train_class_2_replicated, y_remainder_2), axis=0)
    y_train_class_1_Ex_augmented = np.concatenate((y_train_class_1_Ex_replicated, y_remainder_1_Ex), axis=0)
    y_train_class_2_Ex_augmented = np.concatenate((y_train_class_2_Ex_replicated, y_remainder_2_Ex), axis=0)
    y_train_class_1_Ag_augmented = np.concatenate((y_train_class_1_Ag_replicated, y_remainder_1_Ag), axis=0)
    y_train_class_2_Ag_augmented = np.concatenate((y_train_class_2_Ag_replicated, y_remainder_2_Ag), axis=0)
    y_train_class_1_Co_augmented = np.concatenate((y_train_class_1_Co_replicated, y_remainder_1_Co), axis=0)
    y_train_class_2_Co_augmented = np.concatenate((y_train_class_2_Co_replicated, y_remainder_2_Co), axis=0)
    y_train_class_1_Ne_augmented = np.concatenate((y_train_class_1_Ne_replicated, y_remainder_1_Ne), axis=0)
    y_train_class_2_Ne_augmented = np.concatenate((y_train_class_2_Ne_replicated, y_remainder_2_Ne), axis=0)
    y_train_class_1_Op_augmented = np.concatenate((y_train_class_1_Op_replicated, y_remainder_1_Op), axis=0)
    y_train_class_2_Op_augmented = np.concatenate((y_train_class_2_Op_replicated, y_remainder_2_Op), axis=0)

    X_train_class_1_noisy = add_gaussian_noise(X_train_class_1_augmented)
    X_train_class_2_noisy = add_gaussian_noise(X_train_class_2_augmented)

    # 合并原始数据和添加噪声后的数据
    X_train_class_1_final = np.concatenate((X_train_class_1, X_train_class_1_noisy), axis=0)
    X_train_class_2_final = np.concatenate((X_train_class_2, X_train_class_2_noisy), axis=0)
    y_train_class_1_final = np.hstack((y_train_class_1, y_train_class_1_augmented))
    y_train_class_2_final = np.hstack((y_train_class_2, y_train_class_2_augmented))
    y_train_class_1_Ex_final = np.hstack((y_train_class_1_Ex, y_train_class_1_Ex_augmented))
    y_train_class_2_Ex_final = np.hstack((y_train_class_2_Ex, y_train_class_2_Ex_augmented))
    y_train_class_1_Ag_final = np.hstack((y_train_class_1_Ag, y_train_class_1_Ag_augmented))
    y_train_class_2_Ag_final = np.hstack((y_train_class_2_Ag, y_train_class_2_Ag_augmented))
    y_train_class_1_Co_final = np.hstack((y_train_class_1_Co, y_train_class_1_Co_augmented))
    y_train_class_2_Co_final = np.hstack((y_train_class_2_Co, y_train_class_2_Co_augmented))
    y_train_class_1_Ne_final = np.hstack((y_train_class_1_Ne, y_train_class_1_Ne_augmented))
    y_train_class_2_Ne_final = np.hstack((y_train_class_2_Ne, y_train_class_2_Ne_augmented))
    y_train_class_1_Op_final = np.hstack((y_train_class_1_Op, y_train_class_1_Op_augmented))
    y_train_class_2_Op_final = np.hstack((y_train_class_2_Op, y_train_class_2_Op_augmented))

    # 更新标签
    # y_train_class_1_final = np.hstack((y_train_class_1, [1] * X_train_class_1_noisy.shape[0]))
    # y_train_class_2_final = np.hstack((y_train_class_2, [2] * X_train_class_2_noisy.shape[0]))

    x_train_z, y_train, y_train_data_Ex, y_train_data_Ag, y_train_data_Co, y_train_data_Ne, y_train_data_Op = interleave_and_merge(
        X_train_class_0_augmented, X_train_class_1_final, X_train_class_2_final, y_train_class_0_augmented,
        y_train_class_0_Ex, y_train_class_0_Ag, y_train_class_0_Co, y_train_class_0_Ne, y_train_class_0_Op,
        y_train_class_1_final, y_train_class_1_Ex_final, y_train_class_1_Ag_final, y_train_class_1_Co_final,
        y_train_class_1_Ne_final, y_train_class_1_Op_final,
        y_train_class_2_final, y_train_class_2_Ex_final, y_train_class_2_Ag_final, y_train_class_2_Co_final,
        y_train_class_2_Ne_final, y_train_class_2_Op_final)

    x_train_b, y_train_b = shujuzengqiang(X_train_b, y_train_fenlei)
    x_train_l, y_train_l = shujuzengqiang(X_train_l, y_train_fenlei)
    x_train_t, y_train_t = shujuzengqiang(X_train_t, y_train_fenlei)



    return  (x_train_z , x_train_b , x_train_l , x_train_t ,
             X_test_z, X_test_b, X_test_l, X_test_t, y_train, y_test_fenlei, y_train_data_Ex, y_test_data_Ex , y_train_data_Ag,
            y_test_data_Ag, y_train_data_Co, y_test_data_Co ,y_train_data_Ne , y_test_data_Ne ,y_train_data_Op ,y_test_data_Op)




'''if __name__ == '__main__':
    (x_train_z, x_train_b, x_train_l, x_train_t,
     X_test_z, X_test_b, X_test_l, X_test_t, y_train, y_test_fenlei, y_train_data_Ex, y_test_data_Ex, y_train_data_Ag,
     y_test_data_Ag, y_train_data_Co, y_test_data_Co, y_train_data_Ne, y_test_data_Ne, y_train_data_Op, y_test_data_Op) = loading_data()




    num_rows = 10
    data_folder_path = r"C:/Users\su\Desktop\naodian-97\PHQ-9\test\46\4.csv"
    csv_file = os.path.join(data_folder_path)
    df = pd.read_csv(csv_file, header=None)
    data_array = df.values
    samples = split_and_pad(data_array, num_rows)
    print(samples)'''




