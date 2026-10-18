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


def loadData(fileName_ICAspace, fileName_ICAcloud):
    data_ICAspace = io.loadmat(fileName_ICAspace)
    data_ICAcloud = io.loadmat(fileName_ICAcloud)

    all_EEG = data_ICAspace['all_EEG']

    all_cloudPower = data_ICAcloud['all_power_cloud']
    all_cloudBregma = data_ICAcloud['all_bregma_cloud']
    all_cloudLoss = data_ICAcloud['all_loss_cloud']

    nBlock = all_EEG.shape[0]
    for rBlock in range(nBlock):

        nSegment = all_EEG[rBlock][0].shape[1]
        for rSegment in range(nSegment):

            nEmbedding = all_EEG[rBlock][0][0][rSegment].shape[0]
            for rEmbedding in range(nEmbedding):

                temp_EEG_embedding = all_EEG[rBlock][0][0][rSegment][rEmbedding][0]

                temp_cloudPower_embedding = all_cloudPower[rBlock][0][0][rSegment][rEmbedding][0]
                temp_cloudBregma_embedding = all_cloudBregma[rBlock][0][0][rSegment][rEmbedding][0]
                temp_cloudLoss_embedding = all_cloudLoss[rBlock][0][0][rSegment][rEmbedding][0]
                if rEmbedding == 0:
                    temp_EEG = np.expand_dims(temp_EEG_embedding, axis=0)

                    temp_cloudPower = np.expand_dims(temp_cloudPower_embedding, axis=0)
                    temp_cloudBregma = np.expand_dims(temp_cloudBregma_embedding, axis=0)
                    temp_cloudLoss = np.expand_dims(temp_cloudLoss_embedding, axis=0)
                else:
                    temp_EEG_append = np.expand_dims(temp_EEG_embedding, axis=0)
                    temp_cloudPower_append = np.expand_dims(temp_cloudPower_embedding, axis=0)
                    temp_cloudBregma_append = np.expand_dims(temp_cloudBregma_embedding, axis=0)
                    temp_cloudLoss_append = np.expand_dims(temp_cloudLoss_embedding, axis=0)

                    temp_EEG = np.append(temp_EEG, temp_EEG_append, axis=0)
                    temp_cloudPower = np.append(temp_cloudPower, temp_cloudPower_append, axis=0)
                    temp_cloudBregma = np.append(temp_cloudBregma, temp_cloudBregma_append, axis=0)
                    temp_cloudLoss = np.append(temp_cloudLoss, temp_cloudLoss_append, axis=0)

            if rBlock == 0 and rSegment == 0:
                EEG = np.expand_dims(temp_EEG, axis=0)

                cloudPower = np.expand_dims(temp_cloudPower, axis=0)
                cloudBregma = np.expand_dims(temp_cloudBregma, axis=0)
                cloudLoss = np.expand_dims(temp_cloudLoss, axis=0)
            else:
                EEG_append = np.expand_dims(temp_EEG, axis=0)
                cloudPower_append = np.expand_dims(temp_cloudPower, axis=0)
                cloudBregma_append = np.expand_dims(temp_cloudBregma, axis=0)
                cloudLoss_append = np.expand_dims(temp_cloudLoss, axis=0)

                EEG = np.append(EEG, EEG_append, axis=0)
                cloudPower = np.append(cloudPower, cloudPower_append, axis=0)
                cloudBregma = np.append(cloudBregma, cloudBregma_append, axis=0)
                cloudLoss = np.append(cloudLoss, cloudLoss_append, axis=0)

    EEG = torch.tensor(np.real(EEG), dtype=torch.float)
    cloudPower = torch.tensor(cloudPower, dtype=torch.float)
    cloudBregma = torch.tensor(cloudBregma, dtype=torch.float)
    cloudLoss = torch.tensor(cloudLoss, dtype=torch.float)

    return EEG, cloudPower, cloudBregma, cloudLoss
