# K-Nearest Neighbor Classification

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn import datasets
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

# load the MNIST digits dataset
#face = datasets.fetch_olivetti_faces()
#face = datasets.fetch_lfw_people(color = True)
face = datasets.fetch_covtype()

# mtrc = [
#     'manhattan',
#     'euclidean',
#     'chebyshev',
#     #'hamming',
#     #'canberra',
#     #'braycurtis'
# ]

mtrc = [
    64,
    128,
    256,
    260,
    265,
    270,
    280
]


vote = [
    'distance',  
    #'uniform', 
]
lgdl = []
y = []
plt.figure()
for m in mtrc: 
    for v in vote:

        # Training and testing split,
        # 75% for training and 25% for testing
        (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(face.data), face.target, test_size=0.9, random_state=1)

        # take 10% of the training data and use that for validation
        (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1  , random_state=2)

        # Checking sizes of each data split
        print("training data points: {}".format(len(trainLabels)))
        print("validation data points: {}".format(len(valLabels)))
        print("testing data points: {}".format(len(testLabels)))


        # initialize the values of k for our k-Nearest Neighbor classifier along with the
        # list of accuracies for each value of k
        kVals = range(1, 233, 2)
        accuracies = []
        x = []

        # loop over kVals
        for k in range(1, 233, 2):
            # train the classifier with the current value of `k`
            model = KNeighborsClassifier(
                n_neighbors=k,
                algorithm = 'auto',
                weights = v,
                metric = "minkowski",
                p = m
            )
            model.fit(trainData, trainLabels)

            # evaluate the model and print the accuracies list
            score = model.score(valData, valLabels)
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            score = round(score * 100, 2)
            accuracies.append(score)
            k = float(k)
            k = round(k, 2)
            x.append(k)


        y.append(accuracies)

        plt.plot(x, accuracies)
        
        if v == 'distance':
            n = 'weighted'
        else:
            n = 'unweighted'
        #lgd = ''.join([m, ' + ', n])
        lgd = ''.join(['p=', str(m)])
        lgdl.append(lgd)

        # largest accuracy
        # np.argmax returns the indices of the maximum values along an axis
        i = np.argmax(accuracies)
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
            accuracies[i]))


        #Now that I know the best value of k, re-train the ＃classifier
        # model = KNeighborsClassifier(
        #         n_neighbors=kVals[i],
        #         algorithm = 'auto',
        #         weights = v,
        #         metric = m
        #     )
        # model.fit(trainData, trainLabels)


        # # Predict labels for the test set
        # predictions = model.predict(testData)
        # # Evaluate performance of model for each of the digits
        # print("EVALUATION ON TESTING DATA")
        # print(sklearn.metrics.classification_report(testLabels, predictions))
        # # sklearn.metrics.plot_roc_curve(model, testData, testLabels, average = 'binary')
        # # plt.legend(lgd)
        # # plt.plot()

        


# plt.plot(x, accuracies)
# plt.xlabel("k")
# plt.ylabel("precision/100%")
#plt.title("validation of hyperparameter k")
# plt.show()

# plt.plot(
#     x, y[0], 
#     x, y[1], 
#     x, y[2], 
#     x, y[3], 
#     x, y[4], 
#     x, y[5], 
#     x, y[6], 
#     x, y[7]
#     )
plt.xlabel("k")
plt.ylabel("precision/100%")
plt.title("ablation study")
# plt.legend([
#     "L2+weighted", 
#     #"L2+unweighted", 
#     "L1+weighted", 
#     #"L1+unweighted",
#     "L3+weighted", 
#     #"L3+unweighted", 
#     "L4+weighted", 
#     #"L4+unweighted"
#     ])
plt.legend(lgdl)
plt.show()

# AUC visualization




