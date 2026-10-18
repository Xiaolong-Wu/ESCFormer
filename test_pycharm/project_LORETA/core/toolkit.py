import torch
import numpy as np
import scipy.io as io


def strCode(num, nCode):
    code = str(num)
    nCode_temp = code.__len__()

    if nCode_temp < nCode:
        n_temp = nCode - nCode_temp
        for r in range(n_temp):
            if r == 0:
                str_temp = '0'
            else:
                str_temp = str_temp + '0'
        code = str_temp + code

    return code


def loadData(fileName_LORETAcloud):
    data_LORETAcloud = io.loadmat(fileName_LORETAcloud)

    all_EEG = data_LORETAcloud['all_EEG']
    all_cloud = data_LORETAcloud['all_cloud']

    nBlock = all_EEG.shape[0]
    for rBlock in range(nBlock):

        nSegment = all_EEG[rBlock][0].shape[1]
        for rSegment in range(nSegment):

            nEmbedding = all_EEG[rBlock][0][0][rSegment].shape[0]
            for rEmbedding in range(nEmbedding):

                temp_EEG_embedding = all_EEG[rBlock][0][0][rSegment][rEmbedding][0]
                temp_cloud_embedding = all_cloud[rBlock][0][0][rSegment][rEmbedding][0]
                if rEmbedding == 0:
                    temp_EEG = np.expand_dims(temp_EEG_embedding, axis=0)
                    temp_cloud = np.expand_dims(temp_cloud_embedding, axis=0)
                else:
                    temp_EEG_append = np.expand_dims(temp_EEG_embedding, axis=0)
                    temp_cloud_append = np.expand_dims(temp_cloud_embedding, axis=0)

                    temp_EEG = np.append(temp_EEG, temp_EEG_append, axis=0)
                    temp_cloud = np.append(temp_cloud, temp_cloud_append, axis=0)

            if rBlock == 0 and rSegment == 0:
                EEG = np.expand_dims(temp_EEG, axis=0)
                cloud = np.expand_dims(temp_cloud, axis=0)
            else:
                EEG_append = np.expand_dims(temp_EEG, axis=0)
                cloud_append = np.expand_dims(temp_cloud, axis=0)

                EEG = np.append(EEG, EEG_append, axis=0)
                cloud = np.append(cloud, cloud_append, axis=0)

    EEG = torch.tensor(np.real(EEG), dtype=torch.float)
    cloud = torch.tensor(cloud, dtype=torch.float)

    return EEG, cloud
