import tensorflow as tf 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
#==================EXTRACTING AND ANALYSING DATA FROM .mat FILES===========================

train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')
shape_train = train_data.shape
shape_test = test_data.shape

temp_labels = train_labels.reshape(73257).tolist()
temp_labels = dict(Counter(temp_labels))
plt.bar(range(len(temp_labels)), temp_labels.values(), align='center')
plt.xticks(range(len(temp_labels)), temp_labels.keys())

temp_labels = test_labels.reshape(26032).tolist()
temp_labels = dict(Counter(temp_labels))
plt.bar(range(len(temp_labels)), temp_labels.values(), align='center', color='red')

plt.show()
#============================================================================

print shape_train[3], "Images with", shape_train[0], "x", shape_train[0], "RGB grid"

#==================NORMALISATION=============================================

train_data = train_data.astype('float32') / 128.0 - 1
test_data = test_data.astype('float32') / 128.0 - 1

#============================================================================

train_labels = (np.arrange(10) == train_labels[:, None]).astype(float32)
test_labels = (np.arrange(10) == train_labels[:, None]).astype(float32)

#============================================================================

width = 32
height = 32
depth = 3
