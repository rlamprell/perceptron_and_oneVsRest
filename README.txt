############
Assignment 1 
############

The file Assignment_1.py can be run to generate outputs for the questions in the assignment.

Doing so will activate the if __name__ == '__main__' function at the base of the file, which will
in turn utilise the class Assignment_Questions().

Assignment_Questions() contains methods to run q3, q4 and q5 which answer questions 3, 4 and 5 repespectively
from assignment 1.  These do not need any parameters and are preconfigured to output training and testing
accuracies - see Sample output below for an example of this.

Additionally, it's worth mentioning that these classes have been developed to be as 'linearly' readable as possible 
- hence there's a bit of repetition between the methods.

NOTE: Ensure train.data and test.data are in the same location as Assignment_1.py when running




###############################################
Breakdown of each class - creating your own run
###############################################

Each question requires data to be imported from the files train.data and test.data.  This can be achieved by through
initialising the class GetData(fileName) and calling the method load() to return the a list of features and a list of 
labels, seperately.

If answering one of the comparisons from question 3 (a single class vs a single class) the data will need extra processing
to eliminate the third class. the method GetData().classExtractor(firstClass, secondClass) can be used for this.  It needs 
to be fed the string presentation of the class. It will again return a seperate list of features and labels from the dataset,
however, this time it will excluding anything other than these two classes.  And additionally, the labels will be formated 
as +1 or -1, for the first class and second class respectively.

This can then be fed into the Perceptron() class.  Utilising train() to train the data and then test() to test the data.

Questions 4 and 5 are very similar.  But do not require the classExtractor() and can be fed straight into Multiclassifier().
Again, ultilising train() to train and test() to test - There is an assumption that everything the test and train files is
relevant.  For question 5 specifically you will also need to include the state L2=True and a lambda value for train(). 
Example: train(L2=True, lambda_= 0.1)

Note: there are no print statements within Perceptron, Multiclassifier or GetData.  If you wish to display the results, you will 
either need to use the class Assignment_Questions() or create your own.




###########################################
Sample Output from running Assignment_1.py:
###########################################

==============================================
Beginning Question 3_1 -- class-1 vs class-2
----------------------------------------------
Training accuracies:
[0.89 1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
 1.   1.   1.   1.   1.   1.  ]
----------------------------------------------
Testing accuracy = 1.0
==============================================


==============================================
Beginning Question 3_2 -- class-2 vs class-3
----------------------------------------------
Training accuracies:
[0.61 0.48 0.54 0.79 0.86 0.71 0.65 0.65 0.78 0.75 0.76 0.82 0.9  0.8
 0.9  0.84 0.92 0.76 0.82 0.76]
----------------------------------------------
Testing accuracy = 0.5
==============================================


==============================================
Beginning Question 3_3 -- class-1 vs class-3
----------------------------------------------
Training accuracies:
[0.94 1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
 1.   1.   1.   1.   1.   1.  ]
----------------------------------------------
Testing accuracy = 1.0
==============================================


==============================================
Beginning Question 4 One vs Rest
----------------------------------------------
Training accuracy (class-1) = [0.94 1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
 1.   1.   1.   1.   1.   1.  ]
----------------------------------------------
Training accuracy (class-2) = [0.62 0.62 0.59 0.62 0.52 0.6  0.65 0.57 0.68 0.65 0.52 0.65 0.56 0.68
 0.59 0.57 0.54 0.64 0.67 0.63]
----------------------------------------------
Training accuracy (class-3) = [0.58 0.86 0.81 0.72 0.88 0.82 0.82 0.85 0.84 0.87 0.93 0.89 0.82 0.88
 0.9  0.92 0.88 0.81 0.88 0.88]
----------------------------------------------
Testing accuracy =  0.67 (2dp)
==============================================


==============================================
Beginning Question 5 One vs Rest -- l2 reg
----------------------------------------------

Running for lambda = 0.01
==============================================
Training accuracy (class-1) = [0.96 1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
 1.   1.   1.   1.   1.   1.  ]
----------------------------------------------
Training accuracy (class-2) = [0.59 0.49 0.52 0.59 0.55 0.55 0.68 0.48 0.56 0.59 0.52 0.61 0.58 0.53
 0.57 0.62 0.63 0.54 0.56 0.57]
----------------------------------------------
Training accuracy (class-3) = [0.67 0.7  0.64 0.72 0.82 0.65 0.85 0.78 0.74 0.77 0.82 0.62 0.79 0.78
 0.82 0.78 0.73 0.75 0.81 0.82]
----------------------------------------------
Testing accuracy =  0.67 (2dp)
==============================================


Running for lambda = 0.1
==============================================
Training accuracy (class-1) = [0.76 1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
 1.   1.   1.   1.   1.   1.  ]
----------------------------------------------
Training accuracy (class-2) = [0.53 0.5  0.55 0.53 0.57 0.57 0.53 0.59 0.57 0.52 0.5  0.5  0.49 0.5
 0.58 0.54 0.49 0.59 0.54 0.53]
----------------------------------------------
Training accuracy (class-3) = [0.62 0.57 0.66 0.63 0.58 0.54 0.51 0.8  0.57 0.74 0.59 0.62 0.56 0.6
 0.69 0.6  0.58 0.6  0.67 0.56]
----------------------------------------------
Testing accuracy =  0.33 (2dp)
==============================================


Running for lambda = 1.0
==============================================
Training accuracy (class-1) = [0.55 0.46 0.47 0.52 0.51 0.62 0.55 0.5  0.52 0.55 0.62 0.57 0.49 0.48
 0.69 0.57 0.48 0.56 0.6  0.51]
----------------------------------------------
Training accuracy (class-2) = [0.54 0.54 0.52 0.57 0.57 0.58 0.54 0.52 0.53 0.52 0.54 0.54 0.57 0.54
 0.56 0.57 0.52 0.55 0.53 0.52]
----------------------------------------------
Training accuracy (class-3) = [0.48 0.53 0.57 0.57 0.62 0.57 0.51 0.55 0.58 0.52 0.52 0.59 0.5  0.63
 0.52 0.49 0.57 0.6  0.58 0.49]
----------------------------------------------
Testing accuracy =  0.33 (2dp)
==============================================


Running for lambda = 10.0
==============================================
Training accuracy (class-1) = [0.55 0.56 0.52 0.57 0.5  0.61 0.6  0.53 0.52 0.48 0.53 0.49 0.56 0.57
 0.61 0.63 0.57 0.54 0.53 0.6 ]
----------------------------------------------
Training accuracy (class-2) = [0.51 0.62 0.53 0.63 0.59 0.52 0.52 0.48 0.54 0.57 0.57 0.48 0.58 0.5
 0.59 0.52 0.54 0.56 0.52 0.54]
----------------------------------------------
Training accuracy (class-3) = [0.57 0.5  0.52 0.52 0.58 0.55 0.53 0.55 0.62 0.58 0.45 0.53 0.57 0.56
 0.53 0.55 0.48 0.63 0.57 0.48]
----------------------------------------------
Testing accuracy =  0.33 (2dp)
==============================================


Running for lambda = 100.0
==============================================
Training accuracy (class-1) = [0.6  0.52 0.55 0.52 0.62 0.52 0.5  0.57 0.46 0.58 0.56 0.49 0.52 0.55
 0.57 0.59 0.49 0.57 0.53 0.54]
----------------------------------------------
Training accuracy (class-2) = [0.52 0.57 0.53 0.57 0.58 0.58 0.57 0.54 0.55 0.52 0.56 0.62 0.6  0.53
 0.53 0.57 0.53 0.48 0.49 0.6 ]
----------------------------------------------
Training accuracy (class-3) = [0.55 0.5  0.53 0.54 0.57 0.58 0.61 0.65 0.57 0.52 0.57 0.57 0.56 0.49
 0.52 0.52 0.54 0.5  0.58 0.6 ]
----------------------------------------------
Testing accuracy =  0.33 (2dp)
==============================================

