import tensorflow 
import scipy.io
from sklearn.cross_validation import train_test_split
import numpy
train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')

shape_train = train_data.shape
shape_test = test_data.shape

print shape_train[3], "Images with", shape_train[0], "x", shape_train[0], "RGB grid"

train_data.reshape((-1, 32 * 32)).astype(numpy.float32)
train_labels = (numpy.arange(10) == train_labels[:,None]).astype(numpy.float32)