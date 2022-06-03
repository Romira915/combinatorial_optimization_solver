import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

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
plt.show()

num_data = 50  # 使用するデータの数
num_spin = len(raw_data_list[0])  # 画像1枚のスピンの数

# データの加工
edit_data_list = []
for n in range(num_data):
    edit_data = [1 if raw_data_list[n][spin] >=
                 4 else -1 for spin in range(num_spin)]
    edit_data_list.append(edit_data)
