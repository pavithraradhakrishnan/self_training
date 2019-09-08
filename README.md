# Self training
Self Training
a) Implemented a self training system using Logistic Regression Classifier for the following data set 
Program file - selftraining_recursion.py
Input File – semisupervisedtest.csv, train.csv

1. Ds = { ((170, 57, 32), W),
((190, 95, 28), M),
((150, 45, 35), W),
((168, 65, 29), M),
((175, 78, 26), M),
((185, 90, 32), M),
((171, 65, 28), W),
((155, 48, 31), W),
((165, 60, 27), W) }
Du = { (182, 80, 30), (175, 69, 28), (178, 80, 27),
(160, 50, 31), (170, 72, 30), (152, 45, 29),
(177, 79, 28), (171, 62, 27), (185, 90, 30),
(181, 83, 28), (168, 59, 24), (158, 45, 28),
(178, 82, 28), (165, 55, 30), (162, 58, 28),
(180, 80, 29), (173, 75, 28), (172, 65, 27),
(160, 51, 29), (178, 77, 28), (182, 84, 27),
(175, 67, 28), (163, 50, 27), (177, 80, 30),
(170, 65, 28) }



b) Learnt a classifier using the semi-supervised learning algorithm and
compared it against a classifier learned only from the labeled data Ds
using the following test set:
Program file - comparison_selftraining_and_supervised.py
Input File /Test file – semisupervisedtest.csv,train.csv,2b_test_data.csv

Dt = { ((169, 58, 30), W),
((185, 90, 29), M),
((148, 40, 31), W),
((177, 80, 29), M),
((170, 62, 27), W),
((172, 72, 30), M),
((175, 68, 27), W),
((178, 80, 29), M) }



