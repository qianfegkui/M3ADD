import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM
import numpy as np
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MULTAV_CLASSFICATIONModel1(nn.Module):
    def __init__(self):
        super(MULTAV_CLASSFICATIONModel1, self).__init__()
        self.input_dim1 = 211  # 输入特征的数量
        self.input_dim2 = 220
        self.hidden_dim = 30  # 隐藏单元的数量
        self.hidden_dim2 = 240
        self.hidden_dim3 = 960
        #self.hidden_dim3 = 480
        self.num_layers = 1  # LSTM 层的数量
        self.num_heads = 10  # 注意力头的数量
        self.output_dim = 3  # 输出单元的数量
        self.out_dropout1 = 0.1
        self.out_dropout = 0.2

        self.cnn_z_eeg = nn.Conv1d(in_channels=self.input_dim1, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_b_eeg = nn.Conv1d(in_channels=self.input_dim1, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_l_eeg = nn.Conv1d(in_channels=self.input_dim1, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_t_eeg = nn.Conv1d(in_channels=self.input_dim1, out_channels=self.hidden_dim, kernel_size=3, padding=1)

        self.cnn_z_emg = nn.Conv1d(in_channels=self.input_dim2, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_b_emg = nn.Conv1d(in_channels=self.input_dim2, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_l_emg = nn.Conv1d(in_channels=self.input_dim2, out_channels=self.hidden_dim, kernel_size=3, padding=1)
        self.cnn_t_emg = nn.Conv1d(in_channels=self.input_dim2, out_channels=self.hidden_dim, kernel_size=3, padding=1)

        self.encoder_layer_z_eeg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_z_eeg = nn.TransformerEncoder(self.encoder_layer_z_eeg, self.num_layers)
        self.encoder_layer_b_eeg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_b_eeg = nn.TransformerEncoder(self.encoder_layer_b_eeg, self.num_layers)
        self.encoder_layer_l_eeg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_l_eeg = nn.TransformerEncoder(self.encoder_layer_l_eeg, self.num_layers)
        self.encoder_layer_t_eeg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_t_eeg = nn.TransformerEncoder(self.encoder_layer_t_eeg, self.num_layers)

        self.encoder_layer_z_emg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_z_emg = nn.TransformerEncoder(self.encoder_layer_z_emg, self.num_layers)
        self.encoder_layer_b_emg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_b_emg = nn.TransformerEncoder(self.encoder_layer_b_emg, self.num_layers)
        self.encoder_layer_l_emg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_l_emg = nn.TransformerEncoder(self.encoder_layer_l_emg, self.num_layers)
        self.encoder_layer_t_emg = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads,
                                                              batch_first=True, dropout=self.out_dropout1)
        self.transformer_encoder_t_emg = nn.TransformerEncoder(self.encoder_layer_t_emg, self.num_layers)

        self.lstm_z_eeg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_b_eeg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_l_eeg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_t_eeg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)

        self.lstm_z_emg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_b_emg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_l_emg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.lstm_t_emg = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                  batch_first=True, bidirectional=True)
        self.cbam_eeg = CBAM(gate_channels=4, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam_emg = CBAM(gate_channels=4, reduction_ratio=16, pool_types=['avg', 'max'])

        self.MultiheadAttention_eeg = nn.MultiheadAttention(embed_dim=self.hidden_dim2, num_heads=self.num_heads,
                                                            dropout=self.out_dropout1, batch_first=True)
        self.MultiheadAttention_emg = nn.MultiheadAttention(embed_dim=self.hidden_dim2, num_heads=self.num_heads,
                                                            dropout=self.out_dropout1, batch_first=True)

        self.proj1 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj2 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj11 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj21 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj12 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj22 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj13 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj23 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj14 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj24 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj15 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        self.proj25 = nn.Linear(self.hidden_dim3, self.hidden_dim3)
        # self.out_layer = nn.Linear(combined_dim, output_dim)
        self.out_layer_depression = nn.Linear(self.hidden_dim3, 3)

        self.out_layer_Ex = nn.Linear(self.hidden_dim3, 2)
        self.out_layer_Ag = nn.Linear(self.hidden_dim3, 2)
        self.out_layer_Co = nn.Linear(self.hidden_dim3, 2)
        self.out_layer_Ne = nn.Linear(self.hidden_dim3, 2)
        self.out_layer_Op = nn.Linear(self.hidden_dim3, 2)

    def forward(self, x_z_eeg, x_b_eeg, x_l_eeg, x_t_eeg, x_z_emg, x_b_emg, x_l_emg, x_t_emg):
        mask_z_eeg = torch.all(x_z_eeg == 0, dim=-1)
        mask_b_eeg = torch.all(x_b_eeg == 0, dim=-1)
        mask_l_eeg = torch.all(x_l_eeg == 0, dim=-1)
        mask_t_eeg = torch.all(x_t_eeg == 0, dim=-1)

        mask_z_emg = torch.all(x_z_emg == 0, dim=-1)
        mask_b_emg = torch.all(x_b_emg == 0, dim=-1)
        mask_l_emg = torch.all(x_l_emg == 0, dim=-1)
        mask_t_emg = torch.all(x_t_emg == 0, dim=-1)

        # CNN 层
        x_z_eeg = x_z_eeg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_z_eeg = self.cnn_z_eeg(x_z_eeg)
        x_z_eeg = x_z_eeg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_b_eeg = x_b_eeg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_b_eeg = self.cnn_b_eeg(x_b_eeg)
        x_b_eeg = x_b_eeg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_l_eeg = x_l_eeg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_l_eeg = self.cnn_l_eeg(x_l_eeg)
        x_l_eeg = x_l_eeg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_t_eeg = x_t_eeg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_t_eeg = self.cnn_t_eeg(x_t_eeg)
        x_t_eeg = x_t_eeg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        # CNN 层
        x_z_emg = x_z_emg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_z_emg = self.cnn_z_emg(x_z_emg)
        x_z_emg = x_z_emg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_b_emg = x_b_emg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_b_emg = self.cnn_b_emg(x_b_emg)
        x_b_emg = x_b_emg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_l_emg = x_l_emg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_l_emg = self.cnn_l_emg(x_l_emg)
        x_l_emg = x_l_emg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        x_t_emg = x_t_emg.permute(0, 2, 1)  # 形状从 (batch, time, features) 转换为 (batch, features, time)
        x_t_emg = self.cnn_t_emg(x_t_emg)
        x_t_emg = x_t_emg.permute(0, 2, 1)  # 形状转回 (batch, time, features)

        # 多头自注意力层
        attn_output_z_eeg = self.transformer_encoder_z_eeg(x_z_eeg, src_key_padding_mask=mask_z_eeg)
        attn_output_b_eeg = self.transformer_encoder_b_eeg(x_b_eeg, src_key_padding_mask=mask_b_eeg)
        attn_output_l_eeg = self.transformer_encoder_l_eeg(x_l_eeg, src_key_padding_mask=mask_l_eeg)
        attn_output_t_eeg = self.transformer_encoder_t_eeg(x_t_eeg, src_key_padding_mask=mask_t_eeg)

        # 多头自注意力层
        attn_output_z_emg = self.transformer_encoder_z_emg(x_z_emg, src_key_padding_mask=mask_z_emg)
        attn_output_b_emg = self.transformer_encoder_b_emg(x_b_emg, src_key_padding_mask=mask_b_emg)
        attn_output_l_emg = self.transformer_encoder_l_emg(x_l_emg, src_key_padding_mask=mask_l_emg)
        attn_output_t_emg = self.transformer_encoder_t_emg(x_t_emg, src_key_padding_mask=mask_t_emg)

        # LSTM 层
        lstm_out_z_eeg, _ = self.lstm_z_eeg(attn_output_z_eeg)
        # 取最后一个时间步的输出
        x_z_eeg = lstm_out_z_eeg[:, -1, :]

        lstm_out_b_eeg, _ = self.lstm_b_eeg(attn_output_b_eeg)
        # 取最后一个时间步的输出
        x_b_eeg = lstm_out_b_eeg[:, -1, :]

        lstm_out_l_eeg, _ = self.lstm_l_eeg(attn_output_l_eeg)
        # 取最后一个时间步的输出
        x_l_eeg = lstm_out_l_eeg[:, -1, :]

        lstm_out_t_eeg, _ = self.lstm_t_eeg(attn_output_t_eeg)
        # 取最后一个时间步的输出
        x_t_eeg = lstm_out_t_eeg[:, -1, :]

        # LSTM 层
        lstm_out_z_emg, _ = self.lstm_z_emg(attn_output_z_emg)
        # 取最后一个时间步的输出
        x_z_emg = lstm_out_z_emg[:, -1, :]

        lstm_out_b_emg, _ = self.lstm_b_emg(attn_output_b_emg)
        # 取最后一个时间步的输出
        x_b_emg = lstm_out_b_emg[:, -1, :]

        lstm_out_l_emg, _ = self.lstm_l_emg(attn_output_l_emg)
        # 取最后一个时间步的输出
        x_l_emg = lstm_out_l_emg[:, -1, :]

        lstm_out_t_emg, _ = self.lstm_t_emg(attn_output_t_emg)
        # 取最后一个时间步的输出
        x_t_emg = lstm_out_t_emg[:, -1, :]

        x_eeg = torch.stack([x_z_eeg, x_b_eeg, x_l_eeg, x_t_eeg], dim=1)
        x_eeg, scale_eeg = self.cbam_eeg(x_eeg)
        x_eeg = x_eeg.view(x_eeg.shape[0], -1)

        print("scale_emg", scale_eeg)

        #指定你想要保存 .npy 文件的目录路径
        save_dir = r"D:\pycharm\SKF\实验结果\our\tongdaozhuyilifen\xingfugan\eeg"

        # 文件名
        file_name = 'scale_eeg.npy'

        # 完整路径
        full_path = os.path.join(save_dir, file_name)

        # 将 PyTorch 张量转换为 NumPy 数组
        numpy_array = scale_eeg.cpu().numpy()

        # 将 NumPy 数组保存为 .npy 文件
        np.save(full_path, numpy_array)

        # x_eeg = torch.cat([x_z_eeg, x_b_eeg, x_l_eeg, x_t_eeg], dim=1)
        # x_emg = torch.cat([x_z_emg, x_b_emg, x_l_emg, x_t_emg], dim=1)

        x_emg = torch.stack([x_z_emg, x_b_emg, x_l_emg, x_t_emg], dim=1)
        x_emg, scale_emg = self.cbam_emg(x_emg)
        x_emg = x_emg.view(x_emg.shape[0], -1)

        print("scale_emg", scale_emg)

        # 指定你想要保存 .npy 文件的目录路径
        save_dir = r"D:\pycharm\SKF\实验结果\our\tongdaozhuyilifen\xingfugan\emg"

        # 文件名
        file_name = 'scale_emg.npy'

        # 完整路径
        full_path = os.path.join(save_dir, file_name)

        # 将 PyTorch 张量转换为 NumPy 数组
        numpy_array = scale_emg.cpu().numpy()

        # 将 NumPy 数组保存为 .npy 文件
        np.save(full_path, numpy_array)

        # x_eeg = torch.cat((z_eeg, b_eeg, l_eeg, t_eeg), dim=1)
        # x_emg = torch.cat((z_emg, b_emg, l_emg, t_emg), dim=1)

        # 交叉多头自注意力层
        x_m_eeg, _ = self.MultiheadAttention_eeg(x_emg, x_eeg, x_eeg)
        x_m_emg, _ = self.MultiheadAttention_emg(x_eeg, x_emg, x_emg)
        #
        b = torch.cat([x_eeg, x_m_eeg], dim=1)
        c = torch.cat([x_emg, x_m_emg], dim=1)

        x = torch.cat([b, c], dim=1)

        #x = torch.cat([x_eeg, x_emg], dim=1)

        print("x.shape", x.shape)
        # 全连接层
        last_hs_proj_depression = self.proj2(
            F.dropout(F.relu(self.proj1(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_depression += x
        output_depression = self.out_layer_depression(last_hs_proj_depression)

        last_hs_proj_Ex = self.proj21(
            F.dropout(F.relu(self.proj11(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_Ex += x
        output_Ex = self.out_layer_Ex(last_hs_proj_Ex)

        last_hs_proj_Ag = self.proj22(
            F.dropout(F.relu(self.proj12(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_Ag += x
        output_Ag = self.out_layer_Ag(last_hs_proj_Ag)

        last_hs_proj_Co = self.proj23(
            F.dropout(F.relu(self.proj13(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_Co += x
        output_Co = self.out_layer_Co(last_hs_proj_Co)

        last_hs_proj_Ne = self.proj24(
            F.dropout(F.relu(self.proj14(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_Ne += x
        output_Ne = self.out_layer_Ne(last_hs_proj_Ne)

        last_hs_proj_Op = self.proj25(
            F.dropout(F.relu(self.proj15(x)), p=self.out_dropout, training=self.training))
        last_hs_proj_Op += x
        output_Op = self.out_layer_Op(last_hs_proj_Op)

        return output_depression, output_Ex, output_Ag, output_Co, output_Ne, output_Op, x


# Example usage:

'''if __name__ == '__main__':

    z = torch.randn(32, 30, 211)
    b = torch.randn(50, 60, 211)
    l = torch.randn(90, 50, 211)
    t = torch.randn(80, 80, 211)

    #input1 = F.softmax(input1, dim=-1)
    model = DuoShiJian()
    output = model(z,b,l,t)
    print(output)'''


