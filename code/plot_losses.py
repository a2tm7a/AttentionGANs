import numpy as np
import matplotlib.pyplot as plt
import os

def loss_plot(file, path = '', model_name = ''):
    hist = file.tolist()
    x = range(len(hist['d_loss']))

    y1 = hist['d_loss']
    new_y1 = []
    for i in range(0, len(y1), 100):
        new_y1.append(y1[i])

    y2 = hist['g_loss']
    new_y2 = []
    for i in range(0, len(y2), 100):
        new_y2.append(y2[i])

    x = range(len(new_y1))
    plt.plot(x, new_y1, label='d_loss')
    plt.plot(x, new_y2, label='g_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path2 = os.path.join(path, model_name + '_loss3.png')

    plt.savefig(path2)

    plt.close()

    plt.plot(x, new_y1, label='d_loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path3 = os.path.join(path, model_name + '_dloss3.png')

    plt.savefig(path3)

    plt.close()


def loss_plot_DAMSM(file, path='', model_name=''):
    hist = file.tolist()
    x = range(len(hist['S_loss0']))

    y1 = hist['S_loss0']
    y2 = hist['S_loss1']
    y3 = hist['W_loss0']
    y4 = hist['W_loss1']

    plt.plot(x, y1, label='S_loss0')
    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join('', model_name + '_DAMSM_lossS0.png')
    plt.savefig(path)
    plt.close()

    plt.plot(x, y2, label='S_loss1')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join('', model_name + '_DAMSM_lossS1.png')
    plt.savefig(path)
    plt.close()

    plt.plot(x, y3, label='W_loss0')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join('', model_name + '_DAMSM_lossW0.png')
    plt.savefig(path)
    plt.close()

    plt.plot(x, y4, label='W_loss1')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join('', model_name + '_DAMSM_lossW1.png')
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # file = np.load('training_history.npy')
    # loss_plot(file)
    file = np.load('pretrain_DAMSM_train_hist.npy')
    loss_plot_DAMSM(file)