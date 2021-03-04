import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model
from collections import Counter
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix


class Actions(Enum):
    # framewise_recognition.h5

    stand = 0
    wave = 1
    call = 2
    pick = 3


# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def createDataset(dataX_raw):
    outData = []
    for i in range(0,dataX_raw.shape[0]):
        outData.append(dataX_raw[i,0:28])
    return np.array(outData)




# load data
#raw_data = pd.read_csv('data_with_scene.csv', header=0)
raw_data = pd.read_csv('data2.csv', header=0)
dataset = raw_data.values
X = dataset[:, 0:28].astype(float)   #所有行，0-36列都读进来
# X = createDataset(X)
Y = dataset[:, 28]                 #所有行，第36列读进来，36列为动作标签
result = Counter(Y)
print(result)      #计算某个值出现的次数
# print(X.shape[0])
# print(Y.shape[0])
# X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
# Y = dataset[0:3289, 36]

# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
encoder_Y = [0]*result['stand'] + [1]*result['wave']+[2]*result['call']+[3]*result['pick']
print(len(encoder_Y))
print(encoder_Y[0])
test = np.array(encoder_Y)
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)
print(dummy_Y.shape[0])

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)
print(X_train.shape[0])
X_train = X_train.reshape (X_train.shape[0],28,1)
X_test = X_test.reshape (X_test.shape[0],28,1)

#test_size=0.1表示所有数据中有10%的数据作为验证集

# # build keras model
# model = Sequential()
# model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=16, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=4, activation='softmax'))  # units = nums of classes 有几种输出


# build keras model
model = Sequential()
model.add(LSTM(units=30, input_shape=(28, 1)))
# model.add(BatchNormalization())
# model.add(Dense(units=64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=4, activation='softmax'))  # units = nums of classes 有几种输出

# training
# his = LossHistory()
# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])    #模型编译，已经训练设置
# model.fit(X_train, Y_train, batch_size=32, epochs=0, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
# #model.fit  训练数据
# model.summary()
# his.loss_plot('epoch')
#
for i in range(0,10):
    #model.load_weights('framewise_recognition.h5')  #把原来的权重加进来

    his = LossHistory()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
    #model.fit  训练数据
    model.summary()
    his.loss_plot('epoch')

    model.save('framewise_recognition.h5')   #模型保存

    # evaluate and draw confusion matrix
    print('Test:')
    score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
    print('Test Score:{:.3}'.format(score))
    print('Test accuracy:{:.3}'.format(accuracy))
    # confusion matrix
    Y_pred = model.predict(X_test)
    cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
    np.set_printoptions(precision=2)

    plt.figure()
    class_names = ['stand',  'wave', 'call', 'pick']
    plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
    plt.show()

# test
#model.fit(X_train, Y_train, batch_size=32, epochs=0, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
#model1 = Sequential(layers=7)
model.load_weights('framewise_recognition.h5')
#model.load_weights('framewise_recognition.h5', by_name=True)

test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
              0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
              0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
              0.91]
test_np = np.array(test_input)
test_np = test_np.reshape(1, 28,1)


if test_np.size > 0:
    pred = np.argmax(model.predict(test_np))
    init_label = Actions(pred).name
    print(init_label)
