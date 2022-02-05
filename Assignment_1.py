# This file has been developed to answer the questions within Assignment 1 of COMP527
# It contains classes for:
# -- The binary Perceptron algoirthm (one class vs one class)               -- Perceptron()
# -- The OneVsRest algorithm (utilsing the binary perceptron for training)  -- MultiClassifer()
# -- A file importer and data manipulator                                   -- GetData()


# Last Modified -- 2021/03/19
# Rob Lamprell



# only numpy import (assignment restriction)
import numpy as np


# single layer perceptron class used for binary classification
# -- train() runs the binary perceptron algorithm to calculate weights
# -- predict() can be used to generate a best guess on a given feature set
class Perceptron():
    # -- Inputs = features
    # -- MaxIter= number of iterations to be performed 
    # -- learning rate has been assumed to be =1, therefore it has been omitted
    def __init__(self, featureCount, MaxIter=20):
        self.MaxIter    = MaxIter
        self.features   = featureCount

        # setup the weights, errors, results and accuracies
        self.reset()


    # zero all the weights and results
    # -- seperated to make model retraining possible without creating another
    def reset(self):

        # zero all weights to begin with, +1 denotes the bias (+b)
        self.weights        = np.zeros(self.features + 1)
        self.errors_        = []
        self.results_       = []
        self.trainingAccs_  = []
        

    # Calc the activation score 
    def activationScore(self, X):
        
        # W(^T)*X + b
        a = np.dot(X, self.weights[1:]) + self.weights[0]

        return a


    # return the class obtained from the prediction
    # -- binary (+1 or -1)
    def __getClass(self, a):
        
        if a > 0:
            classOutput = +1
        else:
            classOutput = -1            
        
        # return sign(a) - where 0 is part of the negative class
        return classOutput


    # predict using label using the provided features
    def predict(self, X):

        a = self.activationScore(X)        

        return self.__getClass(a)


    # train the perceptron on the data
    def train(self, training_inputs, labels, L2=False, lambda_=0):

        # perform the number of epochs specified (20 by default)
        for _ in range(self.MaxIter):
            # arrays to hold the number of errors and results of each iteration
            errors  = 0
            results = np.empty((0, 2), int)

            # Shuffle the training indicies randomly 
            # -- produces better results than having the labels in-order
            training_inputs, labels = self.__shuffle(training_inputs, labels)

            # iterate through the inputs and labels
            for X, y in zip(training_inputs, labels):
               
                # make a prediction - activation score
                prediction = self.predict(X)

                # record the result of the test
                result  = np.array([y, prediction])
                results = np.vstack([results, result])

                # incase of a classification mismatch
                # -- record an error 
                # -- update the weights (standard or with ridge regression)
                if self._check(y, prediction):
                    errors += 1
                    
                    # update weights for wi
                    # ridge regression
                    if L2:
                        self.weights[1:] = (1-2*lambda_)*self.weights[1:] + y*X

                        # clip the weights of larger values 
                        # -- prevents the weights exploding to infinity
                        # -- arbitrary value selected which could potentially impact accuracy
                        self.weights[1:] = [np.clip(w, -500, 500) for w in self.weights[1:]]
                    # standard perceptron update
                    else:
                        self.weights[1:] += y*X

                    # update bias
                    self.weights[0] += y

            # record all the results and errors
            self.errors_.append(errors)
            self.results_.append(results)
            self.trainingAccs_.append(self.getAccuracy(training_inputs, errors))
        
        # return all weights (including bias)
        return self.weights


    # test a set of features and labels
    def test(self, X, y):

        accuracies  = np.ones(len(y))
        predictions = []
        errors      = 0
        for i in range(len(y)):
            
            # make a prediction on the row - based on the features
            prediction = self.predict(X[i])
            predictions.append(prediction)

            # check if the prediction was correct
            #if y[i]*prediction <= 0:
            if self._check(y[i], prediction):
                accuracies[i] = -1
                errors += 1

        acc_overall = self.getAccuracy(X, errors)

        return acc_overall, predictions, accuracies, 


    # are the actual and predicted labels the same
    # -- pushed in here so MultiClassifier() can 
    #    inherit test from Perceptron()
    def _check(self, actual, prediction):

        return actual*prediction <= 0


    # Shuffle the training indicies randomly 
    def __shuffle(self, training_inputs, labels):

        indicies        = np.arange(training_inputs.shape[0])
        np.random.shuffle(indicies)
        training_inputs = training_inputs[indicies]
        labels          = labels[indicies]

        return training_inputs, labels
    

    # return the perceptron results (which predictions is made)
    def getResults(self):
        return self.results_


    # return the perceptron errors
    def getErrors(self):
        return self.errors_
    

    # return the perceptron weights
    def getWeights(self):
        return self.weights


    # Calculate the accuracy
    def getAccuracy(self, X, errors):
        # Accuracy formula taken from https://www.omnicalculator.com/statistics/accuracy (method #1)
        # (TP + TN) / (TP + TN + FP + FN)
        # -- TP + TN           = population size - errors
        # -- TP + TN + FP + FN = population size
        accuracy = (len(X) - errors) / (len(X))

        return accuracy


    # return the training accuracies 
    def getTrainingAccuracies(self, rounding=2):

        return np.round(self.trainingAccs_, 2)




# Muticlassifier - OneVsRest
# -- generate and train a series of binary classification models 
#    using the Perceptron class
# -- predict by taking the highest activation score from the
#    trained binary models
class MultiClassifier(Perceptron):
    # -- Inputs = features
    # -- nter   = number of iterations to be performed 
    # -- lr     = learning rate
    def __init__(self, featureCount, MaxIter=20):
        self.MaxIter    = MaxIter

        # zero all weights to begin with, +1 denotes the bias (+b)
        self.weights    = np.zeros(featureCount + 1)
        self.errors_    = []
        self.results_   = []

        # unique classes in the set
        self.perceptrons= []
        self.classes    = []


    # train all OneVsRest combinations
    # -- class1 vs (class2 & class3)
    # -- class2 vs (class1 & class3)
    # -- class3 vs (class2 & class3)
    def train(self, X, y, L2=False, lambda_=0):

        # all the unique classes
        self.classes = np.unique(y)

        # train each binary perceptron for every class
        # -- matches to the class label are given +1
        # -- anything other class label are given -1
        for c in self.classes:

            # format the data labels for this class 
            y_c = np.where(y==c, 1, -1)

            # create a binary perceptron
            percep = Perceptron(4)

            # train this value
            percep.train(X, y_c, L2, lambda_)

            # append this to our list of perceptrons
            self.perceptrons.append(percep)


    # are the actual and predicted labels the same
    # -- this is only used when comparing the argmax to the actual label
    def _check(self, actual, prediction):
        return actual!=prediction


    # take the max activation score of the (3) seperate binary classifiers
    def predict(self, X):

        # container for all our predictions
        predictions = []

        # for each class, work out the activation score
        for percep in self.perceptrons:

            a = percep.activationScore(X)
            predictions.append(a)

        # return the class with the highest associated activation score
        # -- these scores represent our confidence in the result
        return self.classes[np.argmax(predictions)]


    # return all the binary perceptron models we have in this model
    def getPerceptrons(self):
        return self.perceptrons

    
    # return the list of classes
    def getClasses(self):
        return self.classes


    # return the training accuracies
    def getTrainingAccuracies(self, rounding=2):

        percepCount       = len(self.perceptrons)
        trainingAccuaries = np.empty((0, 20), float)

        for i in range(percepCount):

            accuracy          = np.round(self.perceptrons[i].getTrainingAccuracies(), 2)
            trainingAccuaries = np.vstack([trainingAccuaries, accuracy])    

        return trainingAccuaries




# Get the training or testing data from the files
class GetData():    
    # default is 4 - same as the assignment
    # -- Note: this assumes the class is in the last index of every row
    def __init__(self, fileName, featureCount=4):

        self.featureCount   = featureCount
        self.X, self.y      = self.__load(fileName)


    # load all data from the files
    def __load(self, fileName):

        X = np.loadtxt(fileName, delimiter=",", usecols=np.arange(0, self.featureCount))
        y = np.loadtxt(fileName, delimiter=",", usecols=[self.featureCount], dtype='str')

        return X, y


    # return features and labels from the file
    def load(self):
        return self.X, self.y
    

    # split on the class name - This should not be used for One Vs Rest.
    #   -- X = Features
    #   -- y = labels
    def __getClass(self, className, X, y, bool_val, featureCount=4):

        # create arrays to push our data into
        out_y = []
        out_X = np.empty((0, featureCount), float)

        # iterate through each row and append the labels and features respectively
        for i, elem in enumerate(y):
            # take everything that is labeled with this class (binary)
            if className in elem:
                out_y.append(bool_val)
                out_X = np.append(out_X, np.array([X[i]]), axis=0)

        return out_X, out_y


    # make the data ready for training / testing
    # -- extract only the necessary data / classes (Example: only class-1 and class-2)
    def classExtractor(self, firstClass, secondClass):
        # load the data
        #features, labels = self.loader.load(fileName)

        # filter the data for the two classes we want to train on 
        # -- (this way should make it more flexible if our dataset changes)
        # -- bool_val is chosen to distinguish between the the two classes (binary Perceptron format)
        x1, y1 = self.__getClass(firstClass,  self.X, self.y, bool_val=+1)
        x2, y2 = self.__getClass(secondClass, self.X, self.y, bool_val=-1)

        # combine the features and labels into a single set for training
        X = np.append(x1, x2, axis=0)
        y = np.append(y1, y2, axis=0)

        # X - features, y - labels
        return X, y




# answer the questions within the assignment
class Assignment_Questions():
    def __init__(self, train_data, test_data):

        self.train_data     = train_data
        self.test_data      = test_data

        self.trainData      = GetData(self.train_data, 4)
        self.testData       = GetData(self.test_data, 4)

    # question 3 - binary perceptron classification 
    # -- the datasets provided include three classes, therefore one class 
    #    is omitted in both training and testing
    def q3(self):

        self.__runq3(1, "class-1", "class-2")
        self.__runq3(2, "class-2", "class-3")
        self.__runq3(3, "class-1", "class-3")


    def __runq3(self, questionNumber, firstClass, secondClass):
        
        # load the data - only collect data with labels of the 2 classes provided
        X, y = self.trainData.classExtractor(firstClass, secondClass)

        # create our perceptron to train and test on
        # create a binary perceptron classifier
        # -- (4) denotes the number of features
        q3 = Perceptron(4)

        # inform the user the of the question
        print("==============================================")
        print("Beginning Question 3_", questionNumber, " -- ", firstClass, " vs ", secondClass, sep='')
        print("----------------------------------------------")

        # train on first_class vs second_class
        q3.train(X, y)

        # print the accuracy of each epoch during the training phase
        print("Training accuracies:")
        print(q3.getTrainingAccuracies())
        print("----------------------------------------------")

        # test and print the results, using the weights established from the training phase
        # load test data
        X_test, y_test = self.testData.classExtractor(firstClass, secondClass)

        # return overall accuracy and predictions of each test line in the data
        accuracy, predictions, accuracies = q3.test(X_test, y_test)
        
        print("Testing accuracy =", accuracy)
        print("==============================================")
        print()
        print()


    # question 4 - multiclassification using One vs Rest
    def q4(self):

        # load in the data
        X, y = self.trainData.load()

        # create a multiclassifer
        # -- (4) denotes the number of features
        q4 = MultiClassifier(4)

        print("==============================================")
        print("Beginning Question 4 One vs Rest", sep='')
        print("----------------------------------------------")

        q4.train(X, y)

        # get the list of perceptron objects we've trained
        perceptrons = q4.getPerceptrons()

        # get the list of classes in the model
        classes = q4.getClasses()

        # get the accuracies from each training set
        accuracies = q4.getTrainingAccuracies()
        
        # print the accuracies
        for i in range(len(perceptrons)):
            print("Training accuracy (", classes[i], ") = ", accuracies[i], sep='')
            print("----------------------------------------------")

        # load test data
        X_test, y_test = self.testData.load()

        # return overall accuracy and predictions of each test line in the data
        accuracy, predictions, accuracies = q4.test(X_test, y_test)

        print("Testing accuracy = ", round(accuracy, 2), "(2dp)")
        print("==============================================")
        print()
        print()

        # return value was used to generate min, max and avg is report
        # -- not implemented in submission
        return round(accuracy, 2)


    # question 5 - l2 (ridge regression)
    # -- Default Lambda values are matched to question 5
    def q5(self, lambdas=[0.01, 0.1, 1.0, 10.0, 100.0]):
        
        # load training an testing data
        # -- outside the loop to avoid loading muiltiple times
        X, y            = self.trainData.load()
        X_test, y_test  = self.testData.load()

        print("==============================================")
        print("Beginning Question 5 One vs Rest -- l2 reg", sep='')
        print("----------------------------------------------")
        print()
        
        # for each lambda value (reg coefficient)
        for l in lambdas:
            # create a multiclassifer
            # -- (4) denotes the number of features
            q5 = MultiClassifier(4)

            print("Running for lambda =", l)
            print("==============================================")
            q5.train(X, y, L2=True, lambda_=l)

            # get the list of perceptron objects we've trained
            perceptrons = q5.getPerceptrons()

            # get the list of classes in the model
            classes = q5.getClasses()       

            # get the accuracies from each training set
            accuracies = q5.getTrainingAccuracies()
            
            # print the accuracies
            for i in range(len(perceptrons)):
                print("Training accuracy (", classes[i], ") = ", accuracies[i], sep='')
                print("----------------------------------------------")

            # return overall accuracy and predictions of each test line in the data
            accuracy, predictions, accuracies = q5.test(X_test, y_test)

            print("Testing accuracy = ", round(accuracy, 2), "(2dp)")
            print("==============================================")
            print()
            print()

        # return value was used to generate min, max and avg is report
        # -- not implemented in submission
        return round(accuracy, 2)




# Provide answers for the assignment
if __name__ == "__main__":

    answers = Assignment_Questions("train.data", "test.data")
    
    answers.q3()
    answers.q4()
    answers.q5()