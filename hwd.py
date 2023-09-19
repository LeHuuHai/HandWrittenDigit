import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
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
X_test, y_test = extract_data(Xtest_all, ytest_all, cls)

logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train[0])

y_pred = logreg.predict(X_test)
print('accuracy_score: %.2f%%' % (100*accuracy_score(y_test[0], y_pred.tolist())))
weights = logreg.coef_[0]

f = open('model_train','wt')
for w in weights:
    f.write(str(w)+' ')
f.close()