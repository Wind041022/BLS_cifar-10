import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
import torchvision
import keras
import tensorflow as tf

from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


#预处理步骤
# 定义随机水平翻转的函数
def random_horizontal_flip(images):
    # 创建一个 TensorFlow 数据集对象
    dataset = tf.data.Dataset.from_tensor_slices(images)
    # 应用随机水平翻转变换
    dataset = dataset.map(lambda x: tf.image.random_flip_left_right(x))
    # 将数据集转换回 NumPy 数组
    flipped_images = np.array(list(dataset.as_numpy_iterator()), dtype=np.uint8)
    return flipped_images

 
''' For Keras dataset_load()'''
(traindata, trainlabel), (testdata, testlabel) = keras.datasets.cifar10.load_data()#cifar10数据集内的图像是32*32*3(高*宽*通道)，（*3是因为它不是灰度图像）



#PCA特征
train_features = np.load('train_features_pca.npy')
test_features = np.load('test_features_pca.npy')

traindata = train_features.reshape(train_features.shape[0], -1).astype('float64')/255
trainlabel = keras.utils.to_categorical(trainlabel, 10)
testdata = test_features.reshape(test_features.shape[0], -1).astype('float64')/255
# testdata = testdata.reshape(testdata.shape[0], 32*32*3).astype('float64')/255
testlabel = keras.utils.to_categorical(testlabel, 10)

# #resnet50特征——翻转
# train_features = np.load('train_features_resnet50.npy')
# test_features = np.load('test_features_resnet50.npy')

# traindata = train_features.reshape(train_features.shape[0], -1).astype('float64')/255
# trainlabel = keras.utils.to_categorical(trainlabel, 10)
# testdata = test_features.reshape(test_features.shape[0], -1).astype('float64')/255
# testlabel = keras.utils.to_categorical(testlabel, 10)



# # 以 50% 的概率对训练数据进行随机水平翻转 原始数据
# traindata = random_horizontal_flip(traindata)

# traindata = traindata.reshape(traindata.shape[0], 32*32*3).astype('float64')/255
# trainlabel = trainlabel.flatten()
# trainlabel = keras.utils.to_categorical(trainlabel, 10)
# testdata = testdata.reshape(testdata.shape[0], 32*32*3).astype('float64')/255
# testlabel = testlabel.flatten()
# testlabel = keras.utils.to_categorical(testlabel, 10)



N1 = 24  #  # of nodes belong to each window
N2 = 16  #  # of windows -------Feature mapping layer
N3 = 680 #  # of enhancement nodes -----Enhance layer N3=680/690/700
L = 5    #  # of incremental steps
M1 = 50 #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------BLS_BASE---------------------------')
BLS(traindata, trainlabel,testdata , testlabel, s, C, N1, N2, N3)
# print('-------------------BLS_ENHANCE------------------------')
# BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
# print('-------------------BLS_FEATURE&ENHANCE----------------')
# M2 = 50  #  # of adding feature mapping nodes
# M3 = 50  #  # of adding enhance nodes
# BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)


teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Training Time
t0 = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
# BLS parameters
s = 0.8  #reduce coefficient
C = 2**(-30) #Regularization coefficient
N1 = 22  #Nodes for each feature mapping layer window 
N2 = 20  #Windows for feature mapping layer
N3 = 540 #Enhancement layer nodes
#  bls-网格搜索
for N1 in range(8,25,2):
    r1 = len(range(8,25,2))
    for N2 in range(10,21,2):
        r2 = len(range(10,21,2))
        for N3 in range(600,701,10):
            r3 = len(range(600,701,10))
            a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
            t0 += 1
            if a>t1:
                tt1 = N1
                tt2 = N2
                tt3 = N3
                t1 = a
            teA.append(a)
            tet.append(b)
            trA.append(c)
            trt.append(d)
            print('percent:' ,round(t0/(r1*r2*r3)*100,4),'%','The best result:', t1,'N1:',tt1,'N2:',tt2,'N3:',tt3)
meanTeACC = np.mean(teA)
meanTrTime = np.mean(trt)
maxTeACC = np.max(teA)   
np.save('meanTeACC_PCA',meanTeACC)
np.save('meanTrTime_PCA',meanTrTime)
np.save('maxTeAcc_PCA',maxTeACC)
# 网格搜索结束后，将最佳参数保存为NumPy数组，并写入.npy文件
best_params = np.array([tt1, tt2, tt3])  # [N1, N2, N3]
np.save('best_params.npy_PCA', best_params)

