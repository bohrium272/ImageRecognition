import tensorflow as tf 
import scipy.io
from sklearn.cross_validation import train_test_split
import numpy
#==================EXTRACTING DATA FROM .mat FILES===========================

train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')
extra_data = scipy.io.loadmat('extra_32x32.mat', variable_name='X').get('X')
extra_labels = scipy.io.loadmat('extra_32x32.mat', variabl_name='y').get('y')
shape_train = train_data.shape
shape_test = test_data.shape

#============================================================================

print shape_train[3], "Images with", shape_train[0], "x", shape_train[0], "RGB grid"

#==================RESHAPING DATA AND LABELS=================================

train_data.reshape((-1, 32 * 32)).astype(numpy.float32)
train_labels = (numpy.arange(10) == train_labels[:,None]).astype(numpy.float32)

#============================================================================


#==================CONSTRUCTING VALIDATION AND TRAINING SETS=================
import random
random.seed()
n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,0] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,0] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,0] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,0] == (i))[0][200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_data = np.concatenate((extra_data[:,:,:,valid_index2], train_data[:,:,:,valid_index]), axis=3).transpose((3,0,1,2))
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)[:,0]
train_data_t = np.concatenate((extra_data[:,:,:,train_index2], train_data[:,:,:,train_index]), axis=3).transpose((3,0,1,2))
train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)[:,0]
test_data = test_data.transpose((3,0,1,2))
test_labels = test_labels[:,0]

print(train_data_t.shape, train_labels_t.shape)
print(test_data.shape, test_labels.shape)
print(valid_data.shape, valid_labels.shape)
#============================================================================
