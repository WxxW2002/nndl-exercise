# python: 3.5.2
# encoding: utf-8

import numpy as np


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self, learning_rate=0.0001, max_iter=1000, C=0.5, tol=1e-3):
        # 请补全此处代码
        self.lr = learning_rate   # 学习率
        self.C = C                # 惩罚系数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol            # 迭代终止条件
        self.W = None             # 参数



    def train(self, data_train):
        """
        训练模型。
        """
        # 请补全此处代码
        x = data_train[:, :2]  # feature [x1, x2]
        y = data_train[:, 2]  # 真实标签

        # 初始化参数
        self.W = np.zeros(x.shape[1])
        b = 0
        n_samples, n_features = x.shape

        # 迭代更新参数
        for epoch in range(self.max_iter):
            for i in range(n_samples):
                condition = y[i] * (np.dot(x[i], self.W) - b ) >= 1
                if condition:
                    self.W -= self.lr * (2 * self.C * np.dot(self.W, x[i]))
                else:
                    self.W -= self.lr * (2 * self.C * np.dot(self.W, x[i]) - np.dot(y[i], x[i]))
                    b -= self.lr * y[i]
                
            if epoch % 100 == 0:
                loss = self.C * np.dot(self.W, self.W) + np.sum(1 - y * (np.dot(x, self.W) - b))
                print(f"Epoch {epoch} Loss: {loss}")
            
            if np.linalg.norm(self.lr * (2 * self.C * np.dot(self.W, x.T) - y * (np.dot(self.W, x.T) + b))) < self.tol:
                break

        self.W = np.concatenate((self.W, np.array([b])))

    def predict(self, x):
        """
        预测标签。
        """
        # 请补全此处代码
        pred = np.dot(x, self.W[:-1]) - self.W[-1]
        return np.sign(pred)


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'D:\Github Space\\nndl-exercise\chap3_SVM\data\\train_linear.txt'
    test_file = 'D:\Github Space\\nndl-exercise\chap3_SVM\data\\test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)

    # 使用训练集训练SVM模型
    svm = SVM()  # 初始化模型
    svm.train(data_train)  # 训练模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]  # 真实标签
    t_train_pred = svm.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))
