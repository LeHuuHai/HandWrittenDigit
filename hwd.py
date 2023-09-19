import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


mntrain = MNIST('data/MNIST/')
# chon file dua tren ten file trong thu muc MNIST_ORG
# train-images.idx3-ubyte
# train-labels.idx1-ubyte
mntrain.load_training()
Xtrain_all = np.asarray(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels.tolist())

mntest = MNIST('data/MNIST/')
# chon file dua tren ten file trong thu muc MNIST_ORG
# t10k-images.idx3-ubyte
# t10k-labels.idx1-ubyte
mntest.load_testing()
Xtest_all = np.asarray(mntest.test_images)
ytest_all = np.array(mntest.test_labels.tolist())

cls = [[0],[1]]
def extract_data(X, y, cls):
    y_res_id = np.array([])
    for i in cls[0]:
        y_res_id = np.hstack((y_res_id, np.where(y==i)[0]))
        # np.where tra ve tuple va dtype
    n0 = len(y_res_id)
    for i in cls[1]:
        y_res_id = np.hstack((y_res_id, np.where(y==i)[0]))
    n1 = len(y_res_id)-n0
    y_res_id = y_res_id.astype(int)
    x_res = X[y_res_id, :]/255
    # /255 :chuan hoa ve [0,1]
    y_res = np.asarray([[0]*n0 + [1]*n1])
    return x_res, y_res

X_train, y_train = extract_data(Xtrain_all, ytrain_all, cls)
y_train = y_train.T
X_test, y_test = extract_data(Xtest_all, ytest_all, cls)
y_test = y_test.T

def sigmoid(s):
    return 1/(1+np.exp(-s))

def predict(weight):
    '''
    :param weight: 784*1
    :param X_test: 2115*728
    :return: y_predict: 2115*1
    '''
    sig = sigmoid(X_test.dot(weight))
    y_predict = np.round(sig)
    return y_predict

def update_weight(w_old, eta, id):
    '''
    :param w_old: 784*1
    :param eta: learning rate(float)
    :param id: id of data point
    :return: new weight: 784*1
    '''
    xi = X_train[id,:]
    yi = y_train[id]
    zi = sigmoid(xi.dot(w_old))
    grad = (zi - yi) * xi.T
    grad = np.reshape(grad, (-1,1))
    w_new = w_old - eta*grad
    return w_new

def train(w_init, eta, iters):
    '''
    :param w_init: initial value: 784*1
    :param eta: learning rate(float)
    :param iters: loops
    :return: list of weight
    '''
    n = X_train.shape[0]
    W = [w_init]
    for it in range(iters):
        mix_data = np.random.permutation(n)
        for i in range(n):
            w_new = update_weight(W[-1], eta, mix_data[i])
            W.append(w_new)
    return W

def accuracy_score(y_predict, y_test):
    n = y_test.shape[0]
    tmp = y_predict - y_test
    true_counts = len(np.where(tmp==0)[0])
    return true_counts/n

def main():
    w_init = np.array([[0]*784]).reshape(-1,1)
    W = train(w_init, 0.0001, 100)
    y_pred = predict(W[-1])
    print('accuracy score: %.2f %%' % (100*accuracy_score(y_pred, y_test)))

    id = np.random.randint(0, y_test.shape[0]+1)
    plt.imshow(X_test[id,:].reshape(28,28))
    plt.gray()
    plt.title(str(y_pred[id]))
    plt.show()


main()



