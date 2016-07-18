import tensorflow as tf 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import SGDClassifier
#==================EXTRACTING AND ANALYSING DATA FROM .mat FILES===========================

train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')
shape_train = train_data.shape
shape_test = test_data.shape

# temp_labels = train_labels.reshape(73257).tolist()
# temp_labels = dict(Counter(temp_labels))
# plt.bar(range(len(temp_labels)), temp_labels.values(), align='center')
# plt.xticks(range(len(temp_labels)), temp_labels.keys())

# temp_labels = test_labels.reshape(26032).tolist()
# temp_labels = dict(Counter(temp_labels))
# plt.bar(range(len(temp_labels)), temp_labels.values(), align='center', color='red')

# plt.show()
#============================================================================

print shape_train[3], "Images with", shape_train[0], "x", shape_train[0], "RGB grid"
#==================NORMALISATION=============================================

# train_data = train_data.astype('float32') / 128.0 - 1
# test_data = test_data.astype('float32') / 128.0 - 1

#============================================================================
image_size = 32
width = 32
height = 32
channels = 3

n_labels = 10
kernel_dimen = 5
batch = 16
depth = 16
hidden = 64

learning_rate = 0.001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
print train_labels.shape
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, channels)).astype(np.float32)
    labels = (np.arange(n_labels) == labels[:,:]).astype(np.float32)
    return dataset, labels
train_data, train_labels = reformat(train_data, train_labels)
test_data, test_labels = reformat(test_data, test_labels)
print train_labels.shape, test_labels.shape
# graph = tf.Graph()
# with graph.as_default():
tf_train_dataset = tf.placeholder(tf.float32, shape=(batch, width, height, channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch, n_labels))
tf_test_dataset = tf.constant(test_data)

layer1_weights = tf.Variable(tf.truncated_normal([kernel_dimen, kernel_dimen, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([depth]))

layer2_weights = tf.Variable(tf.truncated_normal([kernel_dimen, kernel_dimen, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden]))

layer4_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

dropout = tf.placeholder(tf.float32)

def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)

    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(hidden2, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + layer2_biases)

    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shape = hidden4.get_shape().as_list()

    reshape = tf.reshape(hidden4, [-1, shape[1] * shape[2] * shape[3]])
    hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
    dropout_layer = tf.nn.dropout(hidden5, 0.75)
    
    return tf.matmul(dropout_layer, layer4_weights) + layer4_biases

logits = model(tf_train_dataset)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 3000

with tf.Session() as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average = 0
    for step in range(num_steps):
        offset = (step * batch) % (train_labels.shape[0] - batch)
        batch_data = train_data[offset:(offset + batch), :, :, :]
        batch_labels = train_labels[offset:(offset + batch), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.75}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            accu = accuracy(predictions, batch_labels)
            print('Minibatch accuracy: %.1f%%' % accu)
            average += accu
    print "Average Accuracy : ", (average / num_steps) * 100
  
  # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))