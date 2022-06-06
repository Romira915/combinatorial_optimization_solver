import os
from dotenv import load_dotenv
from dwave.system import DWaveSampler, EmbeddingComposite
import random

import matplotlib.pyplot as plt
import numpy as np
import openjij as oj
from openjij import SQASampler
from sklearn.datasets import load_digits
from tqdm import tqdm

load_dotenv()
TOKEN = os.getenv("TOKEN")

# 手書き数字を取得
digits = load_digits()


def show_img(row, col, img_list1, img_list2, title_list1, title_list2, subtitle, subtitlesize, figsize):
    fig, ax = plt.subplots(row, col, figsize=figsize)
    fig.suptitle(subtitle, fontsize=subtitlesize, color='black')

    for i in range(col):
        if row == 1:
            img1 = np.reshape(img_list1[i], (8, 8))
            ax[i].imshow(img1, cmap='Greys')
            ax[i].set_title(title_list1[i])
        else:
            img1 = np.reshape(img_list1[i], (8, 8))
            ax[0, i].imshow(img1, cmap='Greys')
            ax[0, i].set_title(title_list1[i])

            img2 = np.reshape(img_list2[i], (8, 8))
            ax[1, i].imshow(img2, cmap='Greys')
            ax[1, i].set_title(title_list2[i])


# 「0」のみのデータセットを取得
zero_index_list = [i for i, x in enumerate(digits.target) if x == 0]
raw_data_list = [digits.data[i] for i in zero_index_list]

# 画像の表示
show_img(row=1, col=5, img_list1=raw_data_list, img_list2=None,
         title_list1=np.arange(1, 6), title_list2=None,
         subtitle="MNIST zero-dataset", subtitlesize=24, figsize=(14, 3))

num_data = 50  # 使用するデータの数
num_spin = len(raw_data_list[0])  # 画像1枚のスピンの数

# データの加工
edit_data_list = []
for n in range(num_data):
    edit_data = [1 if raw_data_list[n][spin] >=
                 4 else -1 for spin in range(num_spin)]
    edit_data_list.append(edit_data)

raw_title = ['raw-' + str(num) for num in range(1, 6)]
edit_title = ['edit-' + str(num) for num in range(1, 6)]

show_img(row=2, col=5, img_list1=raw_data_list, img_list2=edit_data_list,
         title_list1=raw_title, title_list2=edit_title,
         subtitle="zero-dataset", subtitlesize=20, figsize=(10, 5))


def minus_J(num_spin, J):
    m_J = {(i, j): J[(i, j)] * -1 for i in range(num_spin)
           for j in range(i+1, num_spin)}
    return m_J


def minus_h(num_spin, h):
    m_h = {i: h[i] * -1 for i in range(num_spin)}
    return m_h


def train_model(Tall, eta, dataset, sampler, sample_params, times):
    """ボルツマン機械学習

    Tall: 学習回数
    eta: 学習率
    dataset: データセット
    sampler: サンプラー
    sample_params: サンプリングの設定
    times: times回ごとにパラメータを保存する
    """
    num_data = len(dataset)  # データの数
    num_spin = len(dataset[0])  # スピンの数

    J_dict = {}  # 学習済みの相互作用を格納する
    h_dict = {}  # 学習済みの局所磁場を格納する
    h = {i: 0 for i in range(num_spin)}  # 局所磁場の初期状態
    J = {(i, j): 0 for i in range(num_spin)
         for j in range(i+1, num_spin)}  # 相互作用の初期状態

    for t in tqdm(np.arange(Tall)):
        # OpenJijやD-Waveマシンには-1倍したh,Jを与える必要がある
        m_h = minus_h(num_spin, h)
        m_J = minus_J(num_spin, J)

        # サンプリング
        sampleset = sampler.sample_ising(h=m_h, J=m_J, **sample_params)
        x = sampleset.record.sample

        # 勾配法
        for i in range(num_spin):
            data_h_ave = np.average([dataset[k][i]
                                    for k in range(num_data)])  # データの経験平均
            sample_h_ave = np.average([x[k][i]
                                      for k in range(len(x))])  # サンプルの経験平均
            h[i] += eta * (data_h_ave - sample_h_ave)  # 局所磁場の更新
            for j in range(i+1, num_spin):
                data_J_ave = np.average(
                    [dataset[k][i] * dataset[k][j] for k in range(num_data)])  # データの経験平均
                sample_J_ave = np.average(
                    [x[k][i] * x[k][j] for k in range(len(x))])  # サンプルの経験平均
                J[(i, j)] += eta * (data_J_ave - sample_J_ave)  # 相互作用の更新

        # 一定の学習回数ごとにパラメータを保存
        if t == 0 or (t+1) % times == 0:
            J_dict[t+1] = J.copy()
            h_dict[t+1] = h.copy()

    return J_dict, h_dict


# Openjijの場合
# sampler = oj.SQASampler()

# D-Waveマシンの場合
sampler_config = {'solver': 'DW_2000Q_6', 'token': TOKEN}
sampler = EmbeddingComposite(DWaveSampler(**sampler_config))

J_dict, h_dict = train_model(Tall=30, eta=0.001, dataset=edit_data_list,
                             sampler=sampler, sample_params={'beta': 1, 'num_reads': 100}, times=5)

ans_list = []
for k in J_dict.keys():
    m_h = minus_h(num_spin, h_dict[k])
    m_J = minus_J(num_spin, J_dict[k])
    sampleset = sampler.sample_ising(h=m_h, J=m_J)
    ans_list.append(sampleset.record.sample)

times_title = [str(k) + '-times' for k in J_dict.keys()]
show_img(img_list1=ans_list, img_list2=None,
         title_list1=times_title, title_list2=None,
         subtitle="", subtitlesize=24, figsize=(14, 3))

# 0-4のデータセットを抽出
zero_four_index_list = [i for i, x in enumerate(digits.target) if x <= 4]
raw_data04_list = [digits.data[i] for i in zero_four_index_lst]

# データの加工
num_data04 = len(raw_data04_list)
edit_data04_list = []
for n in range(num_data04):
    edit_data04 = [1 if raw_data04_list[n][spin]
                   >= 4 else -1 for spin in range(num_spin)]
    edit_data04_list.append(edit_data04)

# 学習に用いる画像の表示
raw_title = ['raw-' + str(num) for num in range(5)]
edit_title = ['edit-' + str(num) for num in range(5)]

show_img(row=2, col=5, img_list1=raw_data04_list, img_list2=edit_data04_list,
         title_list1=raw_title, title_list2=edit_title,
         subtitle="", subtitlesize=20, figsize=(10, 5))

target_num = 3
loss_data = edit_data04_list[target_num].copy()


loss_spin_list = random.sample(list(np.arange(64)), 32)
for spin in loss_spin_list:
    loss_data[spin] = 0

vs_list = [edit_data04_list[target_num], loss_data]
show_img(row=1, col=2, img_list1=vs_list, img_list2=None,
         title_list1=['raw', 'loss'], title_list2=None,
         subtitle="raw vs loss", subtitlesize=20, figsize=(8, 4))

loss_h = {s: loss_data[s] for s in range(num_spin)}

J_loss_dict, h = train_model(Tall=30, eta=0.001, dataset=edit_data04_list,
                             sampler=sampler, sample_params={'beta': 1, 'num_reads': 100}, times=10)

result_list = vs_list.copy()
for k in J_loss_dict.keys():
    m_h = minus_h(num_spin, loss_h)
    m_J = minus_J(num_spin, J_loss_dict[k])
    sampleset = sampler.sample_ising(h=m_h, J=m_J, num_reads=100)
    result_list.append(list(sampleset.first.sample.values()))

times_title = ['raw', 'loss'] + [str(k) + '-times' for k in J_loss_dict.keys()]
show_img(row=1, col=2 + len(J_loss_dict), img_list1=result_list, img_list2='',
         title_list1=times_title, title_list2='',
         subtitle="", subtitlesize=24, figsize=(14, 3))
