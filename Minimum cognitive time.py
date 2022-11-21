import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from braincog.base.strategy.surrogate import *
from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d
from braincog.base.node import *
from braincog.base.learningrule import *
from braincog.base.connection import *
from braincog.base.brainarea import *  # 直接可以使用导入库中的各个类
import seaborn as sns

def show_left(data, output_dir=''):
    assert len(data.shape) == 2, 'Shape should be (t, c).'

    data = rearrange(data, 'i j -> j i')
    if isinstance(data, torch.Tensor):
        data = data.to('cpu').numpy()

    plt.figure(figsize=(8, 8))
    sns.heatmap(data, annot=None, cmap='YlGnBu')
    # plt.ylim(0, _max + 1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('left.png')


def show_right(data, output_dir=''):
    assert len(data.shape) == 2, 'Shape should be (t, c).'

    data = rearrange(data, 'i j -> j i')
    if isinstance(data, torch.Tensor):
        data = data.to('cpu').numpy()

    plt.figure(figsize=(8, 8))
    sns.heatmap(data, annot=None, cmap='YlGnBu')
    # plt.ylim(0, _max + 1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('right.png')
#################################################
input = torch.rand(50) * 50
T = 100
left_output = []
right_output = []
left = NoiseLIFNode(threshold=16, tau=2.2)
right = NoiseLIFNode(threshold=16, tau=2.2)
com = LIFNode(threshold=16, tau=2.2)
left.n_reset()
right.n_reset()
com.n_reset()
for i in range(T):
    left_input = torch.zeros(50)
    right_input = torch.zeros(50)
    if i > 30 and i < 40:
        left_input = input
    if i > 30+1  and i < 40 +1:
        right_input = input
    left_temp = left(left_input)
    right_temp = right(right_input)
    left_output.append(left_temp)
    right_output.append(right_temp)

left_output = torch.stack(left_output).detach().cpu().numpy()
right_output = torch.stack(right_output).detach().cpu().numpy()

show_left(left_output)  # 可视化一维输入
show_right(right_output)  # 可视化一维输入
#通过比较输出的两幅png图片可以显著看出不同
#################################################
#################################################
# lif = NoiseLIFNode(threshold=16, tau=2.2)
# signa_input = []
# lif.n_reset()
# for t in range(1000):
#     x = torch.zeros(200)
#     if t > 200 and t < 205:
#         x = torch.rand(200)
#     signa_input.append(lif(x*50))
#
# signa_input = torch.stack(signa_input).detach().cpu().numpy()
# print(signa_input,type(signa_input))
# spike_rate_vis_1d(signa_input)#可视化一维输入
#################################################
# signa_input = torch.tensor([[1., 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # 定义10*10的张量作为输入
#
# left_ear = IFNode(threshold=1.)  # 定义两个节点模拟两个耳朵
# right_ear = IFNode(threshold=1.)
#
# w1 = torch.tensor([[1., 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#
# w2 = torch.tensor([[1., 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#
# connection = [CustomLinear(w1), CustomLinear(w2)]
# # print(connection)
#
# stdp = []
# stdp.append(STDP(left_ear, connection[0]))  # 定义学习规则
# stdp.append(STDP(right_ear, connection[1]))
# # print(stdp)
#
# out1, dw1 = stdp[0](signa_input)
# out2, dw2 = stdp[1](signa_input)
#
# T = 20  # 时间步长
# for i in range(T):
#     out1, dw1 = stdp[0](signa_input)
#     if T > 4:
#         out2, dw2 = stdp[1](signa_input)
#     print("T:",i)
#     print('out1:',out1)
#     print('out2:',out2)
# # spike_rate_vis_1d(out1)#一维可视化
# # spike_rate_vis_1d(out2)#一维可视化
