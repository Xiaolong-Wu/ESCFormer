import os
import mne
import copy
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.linalg import inv
from scipy.interpolate import griddata


# ================= 1. 加载预计算的 Leadfield 和 源空间信息 =================
# 加载 Leadfield（n_channels * n_sources）

# L = np.load('source/leadfield_22ch.npy')
# L = np.load('source/leadfield_59ch.npy')
L = np.load('source/leadfield_59ch.npy')   # (n_channels, n_sources)
n_chan, n_src = L.shape
print(f"Leadfield 形状: {L.shape}")

# 重新创建源空间对象，只是为了获取哪些顶点被选为源点（不保存任何文件）
subjects_dir = 'D:/test_pycharm/project_LORETA/source/root'
src = mne.setup_source_space('fsaverage', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
# 从全部顶点坐标中取出实际源点的坐标（用于点云显示）
coords_all = np.load('source/source_coords.npy')   # (n_sourcesAll, 3)
verts_lh_idx = src[0]['vertno']
verts_rh_idx = src[1]['vertno'] + src[0]['np']
all_source_idx = np.concatenate([verts_lh_idx, verts_rh_idx])
coords = coords_all[all_source_idx, :]  # (n_sources, 3)

nCoords = coords.shape[0]
minX, minY, minZ = coords.min(axis=0)
maxX, maxY, maxZ = coords.max(axis=0)

coords_norm = np.empty_like(coords)
coords_norm[:, 0] = 0.0 + (1 - (-1)) * (coords[:, 0] - 0) / (maxY - minY)
coords_norm[:, 1] = -1.0 + (1 - (-1)) * (coords[:, 1] - minY) / (maxY - minY)
coords_norm[:, 2] = 0.0 + (0 - (-1)) * (coords[:, 2] - maxZ) / (maxZ - minZ)

nRangeRate = 1.5
nCloud=10
xi = np.linspace(-nRangeRate,nRangeRate,2*nCloud)
yi = np.linspace(-nRangeRate,nRangeRate,2*nCloud)
zi = np.linspace(-nRangeRate,0,nCloud)
X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')

# 通道名称（与 leadfield 行对应）
montage = mne.channels.make_standard_montage('standard_1020')

# 15ch: ch_names = ['F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'T7', 'T8', 'P3', 'P4', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
# 22ch: ch_names = ['FC1', 'FC2', 'FC3', 'FC4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'P1', 'P2', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz']
# 59ch: ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'O2', 'Fpz', 'Fz', 'FCz', 'Cz', 'Pz', 'POz', 'Oz']
# 94ch: ch_names = montage.ch_names
ch_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'O1', 'O2', 'Fpz', 'Fz', 'FCz', 'Cz', 'Pz', 'POz', 'Oz']
print(f"通道数: {len(ch_names)}")

# ================= 2. 纯 NumPy 实现 sLORETA / eLORETA 逆算子 =================
def compute_inverse_operator(L, noise_cov=None, lambda2=1/9, method='sLORETA'):
    """
    计算线性逆算子 W
    L: (n_channels, n_sources)
    noise_cov: (n_channels, n_channels) 噪声协方差，None 则使用单位阵
    lambda2: Tikhonov 正则化参数
    method: 'sLORETA' 或 'eLORETA'
    """
    n_chan, n_src = L.shape
    if noise_cov is None:
        noise_cov = np.eye(n_chan)
    C_inv = inv(noise_cov)

    # 普通加权最小范数解（MNE 逆算子）
    G = L.T @ C_inv @ L + lambda2 * np.eye(n_src)
    G_inv = inv(G)
    W_mne = G_inv @ L.T @ C_inv   # (n_src, n_chan)

    if method == 'sLORETA':
        # 计算分辨率矩阵对角元，对每个源点进行噪声归一化
        R = W_mne @ L              # (n_src, n_src)
        S_diag = np.diag(R)
        norm_factors = 1.0 / np.sqrt(np.maximum(S_diag, 1e-15))
        W = np.diag(norm_factors) @ W_mne
        return W

    elif method == 'eLORETA':
        # 迭代计算加权矩阵
        max_iter = 50
        tol = 1e-6
        W_weight = np.eye(n_src)
        A = W_mne @ L               # (n_src, n_src)
        for it in range(max_iter):
            diag_prev = np.diag(W_weight).copy()
            for k in range(n_src):
                ak = A[k, :] @ W_weight   # 向量乘法
                W_weight[k, k] = 1.0 / np.sqrt(max(ak[k], 1e-15))
            if np.max(np.abs(np.diag(W_weight) - diag_prev)) < tol:
                break
        W = W_weight @ W_mne
        return W
    else:
        raise ValueError("method 必须是 'sLORETA' 或 'eLORETA'")

# 计算 sLORETA 逆算子（也可以改成 'eLORETA'）
print("计算 sLORETA 逆算子...")
W = compute_inverse_operator(L, method='sLORETA')

# ================= 3. 加载 EEG 数据 =================

# path_data_save = 'D:/test_matlab/project_EEG/data_2024_Competition/data_003_ICA(-100,700)/0.8s(0)/LORETAcloud'
# path_data_save = 'D:/test_matlab/project_EEG/data_2024_DoC/data_003_ICA(-100,700)/0.8s(0)/LORETAcloud'
path_data_save = 'D:/test_matlab/project_EEG/data_2024_DoC/data_003_ICA(-100,700)/0.8s(0)/LORETAcloud'

# path_data = 'D:/test_matlab/project_EEG/data_2024_Competition/data_003_ICA(-100,700)/0.8s(0)/ICAspace_Norm1(1)'
# path_data = 'D:/test_matlab/project_EEG/data_2024_DoC/data_003_ICA(-100,700)/0.8s(0)/ICAspace_Norm1(1)'
path_data = 'D:/test_matlab/project_EEG/data_2024_DoC/data_003_ICA(-100,700)/0.8s(0)/ICAspace_Norm1(1)'

# path_group = [
#     'group_001',
#     'group_002',
#     'group_003',
#     'group_004'
# ]
# path_group = [
#     'group_004',
#     'group_007',
#     'group_009',
#     'group_013',
#     'group_016',
#     'group_018',
#     'group_022',
#     'group_025',
#     'group_027'
# ]
path_group = [
    'group_004',
    'group_007',
    'group_009',
    'group_013',
    'group_016',
    'group_018',
    'group_022',
    'group_025',
    'group_027'
]

if not os.path.exists(path_data_save):
    os.makedirs(path_data_save)

for rPath_group in path_group:
    path_data_save1 = os.path.join(path_data_save, rPath_group)
    if not os.path.exists(path_data_save1):
        os.makedirs(path_data_save1)

    path_data1 = os.path.join(path_data, rPath_group)
    fileAll = os.listdir(path_data1)
    fileMat = []
    for rFileAll in fileAll:
        if rFileAll.endswith('.mat'):
            fileMat.append(rFileAll)

    for rFileMat in fileMat:
        path_data_save2 = os.path.join(path_data_save1, rFileMat)
        if not os.path.exists(path_data_save2):
            path_data2 = os.path.join(path_data1, rFileMat)
            data_ICAspace = io.loadmat(path_data2)

            all_EEG = data_ICAspace['all_EEG']
            all_activation = copy.deepcopy(all_EEG)
            all_cloud = copy.deepcopy(all_EEG)

            nBlock = all_EEG.shape[0]
            for rBlock in range(nBlock):

                nSegment = all_EEG[rBlock][0].shape[1]
                for rSegment in range(nSegment):

                    nEmbedding = all_EEG[rBlock][0][0][rSegment].shape[0]
                    for rEmbedding in range(nEmbedding):
                        temp_EEG_embedding = all_EEG[rBlock][0][0][rSegment][rEmbedding][0]
                        eeg_data = np.real(temp_EEG_embedding)

                        # ================= 4. 应用到整段数据，得到源空间时间序列 =================
                        source_ts = W @ eeg_data   # (n_src, n_times)

                        # ================= 5. 提取空间特征：整段 RMS =================
                        rms_activation = np.sqrt(np.mean(source_ts ** 2, axis=1))   # (n_src,)
                        all_activation[rBlock][0][0][rSegment][rEmbedding][0] = rms_activation
                        all_cloud[rBlock][0][0][rSegment][rEmbedding][0] = griddata(points=coords_norm, values=rms_activation, xi=(X, Y, Z), method='linear', fill_value=0)

            data_dict = {'all_EEG': all_EEG, 'all_activation': all_activation, 'all_cloud': all_cloud, 'coords': coords, 'coords_norm': coords_norm}
            io.savemat(path_data_save2,data_dict)
            print(' ')
            print('have been done:')
            print(path_data_save2)
        else:
            print(' ')
            print('have existed:')
            print(path_data_save2)

            # ================= 6. 显示结果（点云） =================
            act_show = 0

            if act_show == 1:

                data_LORETAcloud = io.loadmat(path_data_save2)

                all_EEG = data_LORETAcloud['all_EEG']
                all_activation = data_LORETAcloud['all_activation']
                coords = data_LORETAcloud['coords']

                nBlock = all_EEG.shape[0]
                for rBlock in range(nBlock):

                    nSegment = all_EEG[rBlock][0].shape[1]
                    for rSegment in range(nSegment):

                        nEmbedding = all_EEG[rBlock][0][0][rSegment].shape[0]
                        for rEmbedding in range(nEmbedding):
                            rms_activation = all_activation[rBlock][0][0][rSegment][rEmbedding][0]

                            # 归一化用于显示
                            act_norm = rms_activation / np.max(rms_activation)

                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=act_norm, cmap='jet', s=2, alpha=0.5)

                            ax.set_title('sLORETA source RMS activation')

                            # 隐藏坐标轴线和刻度
                            ax.axis('off')

                            # 固定视角 (elev: 仰角, azim: 方位角)
                            # 俯视 ax.view_init(elev=90, azim=-90)
                            # 仰视 ax.view_init(elev=-90, azim=-90)
                            # 左视 ax.view_init(elev=0, azim=180)
                            # 右视 ax.view_init(elev=0, azim=0)
                            # 后视 ax.view_init(elev=0, azim=90)
                            # 前视 ax.view_init(elev=0, azim=-90)
                            ax.view_init(elev=90, azim=-90)

                            # 去掉背景网格和面板
                            ax.grid(False)
                            ax.xaxis.pane.set_visible(False)
                            ax.yaxis.pane.set_visible(False)
                            ax.zaxis.pane.set_visible(False)

                            plt.colorbar(sc)
                            plt.show(block=False)  # 非阻塞显示，窗口保持打开

                            print("完成。源空间点云已显示。")

                        # input("按回车继续……")
                        plt.close('all')
