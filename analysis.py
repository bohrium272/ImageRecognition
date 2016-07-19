import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats
file_names = ['output_1.txt', 'output_2.txt','output_3.txt', 'output_4.txt', 'output_5.txt']
# for fname in file_names:
#     plt.clf()
#     f = file(fname)
#     lines = f.read().split('\n')
#     i = 0
#     training_accuracies = []
#     while 'Average Accuracy' not in lines[i] and i < len(lines):
#         if 'Minibatch accuracy' in lines[i]:
#             training_accuracies.append(float(lines[i][20:lines[i].index('%', 20)]))
#         i += 1

#     train_accuracies = []

#     for j in xrange(len(training_accuracies)):
#         if (j * 50) % 100 == 0:
#             train_accuracies.append(training_accuracies[j]) 
#     plt.plot(range(100), train_accuracies, color='blue', label='Training')
#     train_mode = (stats.mode(train_accuracies).mode[0])
#     i += 1

#     testing_accuracies = []
#     while 'Average Accuracy' not in lines[i] and i < len(lines):
#         if 'Minibatch accuracy' in lines[i]:
#             testing_accuracies.append(float(lines[i][20:lines[i].index('%', 20)]))
#         i += 1
#     test_accuracies = []
#     for j in xrange(len(testing_accuracies)):
#         if (j * 50) % 100 == 0:
#             test_accuracies.append(testing_accuracies[j]) 
#     test_mode = (stats.mode(test_accuracies).mode[0])
#     plt.plot(range(100), test_accuracies, color='red', label='Testing')
    
    
#     plt.title(fname + '\nTraining Mode: ' + str(train_mode) + '\nTesting Mode: ' + str(test_mode))
#     plt.ylabel('Percentage Accuracy')
#     plt.xlabel('Number of iterations(x100)')
#     plt.legend()
#     figname = fname[0:fname.index('.')] + '.png'
#     plt.savefig(figname, pad_inches=0.1, bbox_inches='tight')
fname = 'output_2.txt'
f = file('output_2.txt')
lines = f.read().split('\n')

i = 0
info_loss_training = []
while 'Average Accuracy' not in lines[i] and i < len(lines):
    if 'Minibatch loss' in lines[i]:
        info_loss_training.append(float(lines[i][lines[i].index(':') + 1:].strip()))
    i += 1
print info_loss_training
info_loss_train = []

for j in xrange(len(info_loss_training)):
    if (j * 50) % 100 == 0:
        info_loss_train.append(info_loss_training[j]) 
plt.plot(range(100), info_loss_train, color='blue', label='Training')
i += 1
print info_loss_train
info_loss_testing = []
while 'Average Accuracy' not in lines[i] and i < len(lines):
    if 'Minibatch loss' in lines[i]:
        info_loss_testing.append(float(lines[i][lines[i].index(':') + 1:].strip()))
    i += 1
info_loss_test = []
for j in xrange(len(info_loss_testing)):
    if (j * 50) % 100 == 0:
        info_loss_test.append(info_loss_testing[j]) 
plt.plot(range(100), info_loss_test, color='red', label='Testing')

plt.title(fname + '\nInformation Loss')
plt.ylabel('Information loss')
plt.xlabel('Number of iterations(x100)')
plt.legend()
figname = 'info.png'
plt.savefig(figname, pad_inches=0.1, bbox_inches='tight')