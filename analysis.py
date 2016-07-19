import matplotlib.pyplot as plt
from collections import OrderedDict

file_names = ['output_1.txt', 'output_2.txt','output_3.txt', 'output_4.txt', 'output_5.txt']
for fname in file_names:
    plt.clf()
    f = file(fname)
    lines = f.read().split('\n')
    i = 0
    training_accuracies = []
    while 'Average Accuracy' not in lines[i] and i < len(lines):
        if 'Minibatch accuracy' in lines[i]:
            training_accuracies.append(float(lines[i][20:lines[i].index('%', 20)]))
        i += 1

    train_accuracies = []

    for j in xrange(len(training_accuracies)):
        if (j * 50) % 100 == 0:
            train_accuracies.append(training_accuracies[j]) 
    plt.plot(range(100), train_accuracies, color='blue', label='Training')
    
    i += 1

    testing_accuracies = []
    while 'Average Accuracy' not in lines[i] and i < len(lines):
        if 'Minibatch accuracy' in lines[i]:
            testing_accuracies.append(float(lines[i][20:lines[i].index('%', 20)]))
        i += 1

    test_accuracies = []
    for j in xrange(len(testing_accuracies)):
        if (j * 50) % 100 == 0:
            test_accuracies.append(testing_accuracies[j]) 
    
    plt.plot(range(100), test_accuracies, color='red', label='Testing')
    
    
    plt.title(fname)
    plt.ylabel('Percentage Accuracy')
    plt.xlabel('Number of iterations(x100)')
    plt.legend()
    figname = fname[0:fname.index('.')] + '.png'
    plt.savefig(figname)