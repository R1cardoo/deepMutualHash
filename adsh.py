import copy

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
import time

import torchvision
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

import utils.evaluate as evaluate

from loguru import logger

from models import alexnet, brainNetCNN
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader


def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,
        model_num,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        dataset,
        gamma,
        topk,
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    models = []
    model = brainNetCNN.load_model(code_length, device).to(device)
    models.append(model)
    models.append(copy.deepcopy(model))
    """
        for i in range(model_num):
        model = alexnet.load_model(code_length, device).to(device)
        summary(model, (1, 90, 90))
        models.append(model)

    """
    optimizers = []
    for i in range(model_num):
        optimizer = optim.Adam(
            models[i].parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        optimizers.append(optimizer)

    criterion = ADSH_Loss(code_length, gamma)
    mutualCriterion = nn.KLDivLoss(reduction='batchmean')

    num_retrieval = len(retrieval_dataloader.dataset)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    Us = []
    Bs = []
    B = torch.randn(num_retrieval, code_length).to(device) # B是结果 只需要一个
    Bs.append(B)
    Bs.append(copy.deepcopy(B))
    U = torch.zeros(num_samples, code_length).to(device)
    Us.append(U)
    Us.append(copy.deepcopy(U))
    """
        for i in range(model_num):
        U = torch.zeros(num_samples, code_length).to(device)
        Us.append(U)
    """


    start = time.time()

    for it in range(max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root, dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(device)

        S = (train_targets @ retrieval_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        for i in range(model_num):
            # Training CNN model
            for epoch in range(max_epoch):
                for batch, (data, targets, index) in enumerate(train_dataloader):
                    data, targets, index = data.to(device), targets.to(device), index.to(device)
                    optimizers[i].zero_grad()

                    F = models[i](data)
                    Us[i][index, :] = F.data
                    cnn_loss = criterion(F, Bs[i], S[index, :], sample_index[index])
                    mutual_loss = 0
                    for j in range(model_num):
                        if i != j:
                            LogSoft = nn.LogSoftmax(dim=1)
                            input = LogSoft(models[i](data))
                            Soft = nn.Softmax(dim=1)
                            output = Soft(models[j](data))
                            mutual_loss += mutualCriterion(input, output)

                    cnn_loss += mutual_loss / (model_num - 1)
                    cnn_loss.backward()
                    optimizers[i].step()

            # Update B/
            expand_U = torch.zeros(Bs[i].shape).to(device)
            expand_U[sample_index, :] = Us[i]
            Bs[i] = solve_dcc(Bs[i], Us[i], expand_U, S, code_length, gamma)

            # Total loss
            iter_loss = calc_loss(Us[i], Bs[i], S, code_length, sample_index, gamma)
            logger.debug('[model:{}][iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(i, it + 1, max_iter, iter_loss,
                                                                              time.time() - iter_start))

            query_code = generate_code(models[i], query_dataloader, code_length, device)

            mACC, mSEN, mSPC = evaluate.acurracy(
                query_code.to(device),
                Bs[i],
                query_dataloader.dataset.get_onehot_targets().to(device),
                retrieval_targets,
                device,
                topk,
            )
            logger.debug('[model:{}] acc = {}, sen = {}, spc = {}'.format(i, mACC, mSEN, mSPC))

    logger.info('[Training time:{:.2f}]'.format(time.time() - start))
    return mACC
"""
Loss_list_model0 = get_model0_loss()
    Acc_list_model0 = get_model0_acc()
    Loss_list_model1 = get_model1_loss()
    Acc_list_model1 = get_model1_acc()

    x1 = range(0,100)
    x2 =range(0,100)
    plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x1, Acc_list_model0, 'o-', label="Model0_Accuracy")
    plt.plot(x1, Acc_list_model1, 'o-', label="Model1_Accuracy")
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, Loss_list_model0, '.-', label="Model0_Loss")
    plt.plot(x2, Loss_list_model1, '.-', label="Model1_Loss")
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.legend(loc='best')
    plt.show()
"""

"""

# Evaluate queryCode是生成的code
    for i in range(model_num):

        query_code = generate_code(models[i], query_dataloader, code_length, device)

        mAP = evaluate.mean_average_precision(
            query_code.to(device),
            Bs[i],
            query_dataloader.dataset.get_onehot_targets().to(device),
            retrieval_targets,
            device,
            topk,
        )
        # Save checkpoints
        torch.save(query_code.cpu(), os.path.join('checkpoints', 'query_code.t'))
        torch.save(B.cpu(), os.path.join('checkpoints', 'database_code.t'))
        torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join('checkpoints', 'query_targets.t'))
        torch.save(retrieval_targets.cpu(), os.path.join('checkpoints', 'database_targets.t'))
        torch.save(model.cpu(), os.path.join('checkpoints', 'model.t'))
"""
    # TODO: 这里有问题——for循环return多个



def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code

def get_model0_loss():
    Loss_list = []
    Loss_list.append(147.52)
    Loss_list.append(141.57)
    Loss_list.append(146.76)
    Loss_list.append(145.29)
    Loss_list.append(142.00)

    Loss_list.append(140.14)
    Loss_list.append(143.83)
    Loss_list.append(132.40)
    Loss_list.append(123.54)
    Loss_list.append(105.68)

    Loss_list.append(86.85)
    Loss_list.append(130.70)
    Loss_list.append(96.79)
    Loss_list.append(105.82)
    Loss_list.append(98.40)

    Loss_list.append(80.04)
    Loss_list.append(98.73)
    Loss_list.append(66.08)
    Loss_list.append(87.89)
    Loss_list.append(83.71)

    Loss_list.append(71.04)
    Loss_list.append(54.93)
    Loss_list.append(51.63)
    Loss_list.append(47.51)
    Loss_list.append(63.91)

    Loss_list.append(48.19)
    Loss_list.append(44.21)
    Loss_list.append(39.73)
    Loss_list.append(41.90)
    Loss_list.append(47.91)

    Loss_list.append(42.72)
    Loss_list.append(39.34)
    Loss_list.append(54.90)
    Loss_list.append(33.02)
    Loss_list.append(29.37)

    Loss_list.append(30.05)
    Loss_list.append(40.59)
    Loss_list.append(70.18)
    Loss_list.append(35.71)
    Loss_list.append(37.26)

    Loss_list.append(22.38)
    Loss_list.append(24.99)
    Loss_list.append(21.88)
    Loss_list.append(24.21)
    Loss_list.append(25.43)

    Loss_list.append(21.48)
    Loss_list.append(25.19)
    Loss_list.append(19.60)
    Loss_list.append(22.98)
    Loss_list.append(20.25)

    Loss_list.append(19.83)
    Loss_list.append(17.07)
    Loss_list.append(21.54)
    Loss_list.append(19.50)
    Loss_list.append(18.94)

    Loss_list.append(16.12)
    Loss_list.append(16.86)
    Loss_list.append(14.86)
    Loss_list.append(22.41)
    Loss_list.append(24.63)

    Loss_list.append(25.98)
    Loss_list.append(22.49)
    Loss_list.append(20.62)
    Loss_list.append(17.54)
    Loss_list.append(13.07)

    Loss_list.append(29.97)
    Loss_list.append(20.55)
    Loss_list.append(22.74)
    Loss_list.append(13.90)
    Loss_list.append(13.25)

    Loss_list.append(12.92)
    Loss_list.append(13.88)
    Loss_list.append(17.35)
    Loss_list.append(17.65)
    Loss_list.append(17.42)

    Loss_list.append(14.19)
    Loss_list.append(18.99)
    Loss_list.append(23.83)
    Loss_list.append(18.63)
    Loss_list.append(18.55)

    Loss_list.append(16.23)
    Loss_list.append(12.68)
    Loss_list.append(13.07)
    Loss_list.append(14.74)
    Loss_list.append(12.74)

    Loss_list.append(12.38)
    Loss_list.append(11.10)
    Loss_list.append(13.19)
    Loss_list.append(10.95)
    Loss_list.append(10.65)

    Loss_list.append(14.86)
    Loss_list.append(20.13)
    Loss_list.append(24.57)
    Loss_list.append(29.60)
    Loss_list.append(17.15)

    Loss_list.append(13.77)
    Loss_list.append(18.73)
    Loss_list.append(15.02)
    Loss_list.append(19.72)
    Loss_list.append(17.12)
    return Loss_list

def get_model0_acc():
    acc_list = []
    acc_list.append(0.46)
    acc_list.append(0.58)
    acc_list.append(0.46)
    acc_list.append(0.5)
    acc_list.append(0.53)

    acc_list.append(0.62)
    acc_list.append(0.54)
    acc_list.append(0.56)
    acc_list.append(0.62)
    acc_list.append(0.6)

    acc_list.append(0.6)
    acc_list.append(0.67)
    acc_list.append(0.58)
    acc_list.append(0.6)
    acc_list.append(0.63)

    acc_list.append(0.63)
    acc_list.append(0.73)
    acc_list.append(0.61)
    acc_list.append(0.62)
    acc_list.append(0.6)

    acc_list.append(0.7)
    acc_list.append(0.7)
    acc_list.append(0.64)
    acc_list.append(0.67)
    acc_list.append(0.66)

    acc_list.append(0.65)
    acc_list.append(0.64)
    acc_list.append(0.7)
    acc_list.append(0.72)
    acc_list.append(0.72)

    acc_list.append(0.68)
    acc_list.append(0.69)
    acc_list.append(0.67)
    acc_list.append(0.67)
    acc_list.append(0.7)

    acc_list.append(0.72)
    acc_list.append(0.64)
    acc_list.append(0.62)
    acc_list.append(0.63)
    acc_list.append(0.69)

    acc_list.append(0.63)
    acc_list.append(0.62)
    acc_list.append(0.66)
    acc_list.append(0.65)
    acc_list.append(0.68)

    acc_list.append(0.69)
    acc_list.append(0.67)
    acc_list.append(0.68)
    acc_list.append(0.66)
    acc_list.append(0.66)

    acc_list.append(0.67)
    acc_list.append(0.68)
    acc_list.append(0.69)
    acc_list.append(0.69)
    acc_list.append(0.68)

    acc_list.append(0.68)
    acc_list.append(0.65)
    acc_list.append(0.66)
    acc_list.append(0.65)
    acc_list.append(0.64)

    acc_list.append(0.68)
    acc_list.append(0.67)
    acc_list.append(0.71)
    acc_list.append(0.68)
    acc_list.append(0.65)

    acc_list.append(0.64)
    acc_list.append(0.66)
    acc_list.append(0.66)
    acc_list.append(0.68)
    acc_list.append(0.68)

    acc_list.append(0.67)
    acc_list.append(0.65)
    acc_list.append(0.68)
    acc_list.append(0.66)
    acc_list.append(0.66)

    acc_list.append(0.65)
    acc_list.append(0.64)
    acc_list.append(0.61)
    acc_list.append(0.68)
    acc_list.append(0.66)

    acc_list.append(0.67)
    acc_list.append(0.66)
    acc_list.append(0.67)
    acc_list.append(0.67)
    acc_list.append(0.71)

    acc_list.append(0.67)
    acc_list.append(0.69)
    acc_list.append(0.68)
    acc_list.append(0.68)
    acc_list.append(0.65)

    acc_list.append(0.65)
    acc_list.append(0.66)
    acc_list.append(0.64)
    acc_list.append(0.64)
    acc_list.append(0.66)

    acc_list.append(0.64)
    acc_list.append(0.66)
    acc_list.append(0.67)
    acc_list.append(0.64)
    acc_list.append(0.64)
    return acc_list


def get_model1_loss():
    Loss_list = []
    Loss_list.append(147.08)
    Loss_list.append(146.90)
    Loss_list.append(143.41)
    Loss_list.append(103.43)
    Loss_list.append(119.22)

    Loss_list.append(109.56)
    Loss_list.append(109.32)
    Loss_list.append(91.07)
    Loss_list.append(100.82)
    Loss_list.append(83.72)

    Loss_list.append(103.42)
    Loss_list.append(96.39)
    Loss_list.append(72.98)
    Loss_list.append(103.46)
    Loss_list.append(95.33)

    Loss_list.append(117.07)
    Loss_list.append(76.64)
    Loss_list.append(84.02)
    Loss_list.append(83.67)
    Loss_list.append(68.56)

    Loss_list.append(51.53)
    Loss_list.append(47.36)
    Loss_list.append(89.94)
    Loss_list.append(67.56)
    Loss_list.append(53.78)

    Loss_list.append(53.43)
    Loss_list.append(53.80)
    Loss_list.append(37.98)
    Loss_list.append(36.49)
    Loss_list.append(39.48)

    Loss_list.append(34.13)
    Loss_list.append(47.33)
    Loss_list.append(40.53)
    Loss_list.append(31.92)
    Loss_list.append(32.98)

    Loss_list.append(38.26)
    Loss_list.append(31.05)
    Loss_list.append(30.72)
    Loss_list.append(37.86)
    Loss_list.append(43.60)

    Loss_list.append(30.31)
    Loss_list.append(27.26)
    Loss_list.append(28.36)
    Loss_list.append(23.58)
    Loss_list.append(24.50)

    Loss_list.append(24.49)
    Loss_list.append(22.24)
    Loss_list.append(18.63)
    Loss_list.append(17.41)
    Loss_list.append(20.75)

    Loss_list.append(23.71)
    Loss_list.append(19.32)
    Loss_list.append(17.87)
    Loss_list.append(15.73)
    Loss_list.append(17.33)

    Loss_list.append(22.28)
    Loss_list.append(22.49)
    Loss_list.append(21.87)
    Loss_list.append(18.92)
    Loss_list.append(15.12)

    Loss_list.append(14.77)
    Loss_list.append(18.73)
    Loss_list.append(17.57)
    Loss_list.append(13.99)
    Loss_list.append(18.57)

    Loss_list.append(15.30)
    Loss_list.append(12.11)
    Loss_list.append(12.47)
    Loss_list.append(13.26)
    Loss_list.append(16.36)

    Loss_list.append(15.11)
    Loss_list.append(13.20)
    Loss_list.append(14.85)
    Loss_list.append(17.02)
    Loss_list.append(13.62)

    Loss_list.append(12.17)
    Loss_list.append(12.33)
    Loss_list.append(16.93)
    Loss_list.append(15.12)
    Loss_list.append(14.91)

    Loss_list.append(15.83)
    Loss_list.append(13.16)
    Loss_list.append(12.39)
    Loss_list.append(13.06)
    Loss_list.append(13.16)

    Loss_list.append(14.42)
    Loss_list.append(12.91)
    Loss_list.append(10.84)
    Loss_list.append(11.77)
    Loss_list.append(10.08)

    Loss_list.append(11.91)
    Loss_list.append(12.62)
    Loss_list.append(13.83)
    Loss_list.append(12.51)
    Loss_list.append(10.78)

    Loss_list.append(10.55)
    Loss_list.append(11.09)
    Loss_list.append(12.61)
    Loss_list.append(15.05)
    Loss_list.append(12.62)
    return Loss_list

def get_model1_acc():
    acc_list = []
    acc_list.append(0.47)
    acc_list.append(0.46)
    acc_list.append(0.55)
    acc_list.append(0.56)
    acc_list.append(0.56)

    acc_list.append(0.64)
    acc_list.append(0.66)
    acc_list.append(0.58)
    acc_list.append(0.65)
    acc_list.append(0.66)

    acc_list.append(0.59)
    acc_list.append(0.66)
    acc_list.append(0.61)
    acc_list.append(0.61)
    acc_list.append(0.57)

    acc_list.append(0.62)
    acc_list.append(0.6)
    acc_list.append(0.68)
    acc_list.append(0.71)
    acc_list.append(0.65)

    acc_list.append(0.66)
    acc_list.append(0.64)
    acc_list.append(0.67)
    acc_list.append(0.67)
    acc_list.append(0.67)

    acc_list.append(0.6)
    acc_list.append(0.69)
    acc_list.append(0.65)
    acc_list.append(0.68)
    acc_list.append(0.67)

    acc_list.append(0.67)
    acc_list.append(0.62)
    acc_list.append(0.62)
    acc_list.append(0.65)
    acc_list.append(0.61)

    acc_list.append(0.59)
    acc_list.append(0.66)
    acc_list.append(0.66)
    acc_list.append(0.61)
    acc_list.append(0.65)

    acc_list.append(0.64)
    acc_list.append(0.68)
    acc_list.append(0.68)
    acc_list.append(0.61)
    acc_list.append(0.66)

    acc_list.append(0.65)
    acc_list.append(0.63)
    acc_list.append(0.63)
    acc_list.append(0.66)
    acc_list.append(0.65)

    acc_list.append(0.64)
    acc_list.append(0.65)
    acc_list.append(0.61)
    acc_list.append(0.65)
    acc_list.append(0.62)

    acc_list.append(0.67)
    acc_list.append(0.6)
    acc_list.append(0.62)
    acc_list.append(0.61)
    acc_list.append(0.61)

    acc_list.append(0.59)
    acc_list.append(0.66)
    acc_list.append(0.64)
    acc_list.append(0.62)
    acc_list.append(0.59)

    acc_list.append(0.65)
    acc_list.append(0.64)
    acc_list.append(0.63)
    acc_list.append(0.67)
    acc_list.append(0.64)

    acc_list.append(0.65)
    acc_list.append(0.65)
    acc_list.append(0.62)
    acc_list.append(0.63)
    acc_list.append(0.63)

    acc_list.append(0.65)
    acc_list.append(0.64)
    acc_list.append(0.64)
    acc_list.append(0.6)
    acc_list.append(0.58)

    acc_list.append(0.64)
    acc_list.append(0.65)
    acc_list.append(0.64)
    acc_list.append(0.65)
    acc_list.append(0.62)

    acc_list.append(0.64)
    acc_list.append(0.61)
    acc_list.append(0.64)
    acc_list.append(0.65)
    acc_list.append(0.65)

    acc_list.append(0.63)
    acc_list.append(0.62)
    acc_list.append(0.66)
    acc_list.append(0.65)
    acc_list.append(0.66)

    acc_list.append(0.65)
    acc_list.append(0.62)
    acc_list.append(0.64)
    acc_list.append(0.65)
    acc_list.append(0.61)
    return acc_list

