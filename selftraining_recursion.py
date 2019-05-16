
import numpy as np
import csv

x_train =[]
y_train = []
test_data =[]
print("*************************SELF TRAINING****************************************")
def data_clean(x_train,y_train,test_data):
    with open("train.csv",encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile,delimiter=',')
        data = [data for data in rows]
    for x in range(len(data)):
        ins = []
        for y in range(3):
            ins.append(float(data[x][y]))
        if(data[x][y+1] == "M"):
            y_train.append(float(1))
        else:
            np.array(y_train.append(float(0)))
        x_train.append(ins)
    x = np.array(x_train)
    y = np.array(y_train)
    return x,y

def normalise_data(x):
    " we can normalise by subtracting the maximum value by the minimum value and  then divide by the standard deviation of x or divide by the range"
    max_x = np.max(x,axis = 0)
    min_x = np.min(x,axis = 0)
    normalised_x = 1 - (max_x - x)/(max_x - min_x)
    return normalised_x

def calculate_sigmoid(x,weight):
    h = 1.0 / (1 + np.exp(-np.dot(x, weight.T)))
    return h


def calculate_gradient(weight,x,y):
    h = calculate_sigmoid(x,weight)
    a = h - y.reshape(x.shape[0], -1)
    b = np.dot(a.T, x)
    return b

def calculate_cost(x,weight,y):
    h = calculate_sigmoid(x,weight)
    y = np.squeeze(y)
    h = np.mean((-(y * np.log(h)) - ((1 - y)) * np.log(1 -h)))
    return h


def grad_descent(x, y, weight, learning, gradient):
    cost = calculate_cost(x,weight, y)
    iteration, change = 1, 1
    while (change > gradient):
        prev_cost = cost
        # finding the weights and updating it after every itertation. The loop goes on unti the condition is met
        weight = weight -  (learning * calculate_gradient(weight,x,y))
        cost  =  calculate_cost(x,weight,y)
        change = prev_cost - cost
        iteration = iteration + 1
    return weight, iteration
def minimise_loss(weight, learning,gradient):
    return weight - learning * gradient

def predict_val(weight,x):
    prob = calculate_sigmoid(x,weight)
    prediction = np.where(prob >= .5, 1, 0)
    return np.squeeze(prediction),prob

x,y, = data_clean(x_train,y_train,test_data)
semisupervised_test = []
with open("semisupervisedtest.csv", encoding='utf-8-sig') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    test_data = [test_data for test_data in rows]

for a in range(len(test_data)):
    ins = []
    for b in range(3):
        ins.append(float(test_data[a][b]))

    semisupervised_test.append(ins)
while(semisupervised_test):
    X = x

    X = normalise_data(X)
    one_column = np.ones((len(X), 1))
    X = np.concatenate((one_column, X), axis=1)
    weight = np.matrix(np.zeros(4))

    weight, n = grad_descent(X, y, weight, 0.01, 0.0005)
    Semisupervised_test = np.array(semisupervised_test)

    Semisupervised_test = normalise_data(Semisupervised_test)
    one_column = np.ones((len(Semisupervised_test), 1))
    Semisupervised_test = np.concatenate((one_column, Semisupervised_test), axis=1)

    predict_test, prob_test = predict_val(weight, Semisupervised_test)
    prob_test = prob_test.tolist()
    new_x = x.tolist()

    new_y = y.tolist()
    new_semisupervised = []
    for i in range(len(prob_test)):

        if (prob_test[i][0]) >= 0.9 or prob_test[i][0] <= 0.1:

            new_x.extend([semisupervised_test[i]])
            if (prob_test[i][0] >= 0.9):
                new_y.extend([1])
            else:
                new_y.extend([0])

        else:
            new_semisupervised.append(semisupervised_test[i])
    print("The predicted probability for the current unlabelled data set is",prob_test)
    new_y_2 = []
    for i in y:
        if (i == 0.0 or i == 0):
            new_y_2.append('W')
        else:
            new_y_2.append('M')
    print("The Labelled data set after moving the high confidence samples is ",list(zip(new_x,new_y_2)))
    print("The unlabelled data set at this step is ",new_semisupervised)
    if(len(new_semisupervised) == 1):
        new_x.extend([new_semisupervised[0]])
        for i in range(len(prob_test)):
            if(prob_test[i][0] <0.9 and prob_test[i][0] > 0.1):
                if(prob_test[i][0] <0.5):
                    new_y.extend([0])
                else:
                    new_y.extend([1])
        new_semisupervised = []
    x = np.array(new_x)
    y = np.array(new_y)



    semisupervised_test = new_semisupervised

x = x.tolist()
y = y.tolist()
result_y = []
for i in y:
    if(i ==  0.0 or i == 0):
        result_y.append('W')
    else:
        result_y.append('M')

Final_Labelled_Data_set = list(zip(x,result_y))
print("*******************************************************")
print("*************RESULT IS  *******************************")
print("Final labelled set formed as result of SELF TRAINING is ",Final_Labelled_Data_set)
print("Length of the labelled data set formed as s result of self training is",len(Final_Labelled_Data_set))
print("The regression co-efficients are", weight)
print("*******************************************************")