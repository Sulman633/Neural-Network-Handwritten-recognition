import glob
import math
import random
import csv
import numpy as np



file_list = glob.glob("*.txt")

def sigmoids(x,deriv = False):
    if deriv == True:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def tanh(x, deriv = False):
    if deriv == True:
        return 1 - (np.tanh(x)) ** 2

    return np.tanh(x)


#Sets up the structure for the neural network with 1 input layer, 1 hidden layer and 1 output layer.
class NeuralNetwork:

    def __init__(self, hidden_layer, output_layer, epochs, learning_rate, momentum):
        self.input_layer = 64

        self.hidden_layer = hidden_layer

        self.output_layer = output_layer

        self.epochs = epochs

        self.learning_rate = learning_rate

        self.momentum = momentum

        self.correct_answer = 0

    #Initialize weight connections for input_layer -> hidden_layer -> output_layer

        self.inputweights = np.random.uniform(-0.5, 0.5, size=(self.input_layer, self.hidden_layer))

        self.outputweights = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer, self.output_layer))

    #Intialize bias' to be added to each weight after summation
        self.input_bias = np.random.uniform(-0.5, 0.5, size=(1, self.hidden_layer))

        self.output_bias = np.random.uniform(-0.5, 0.5, size=(1, self.output_layer))

    #These lists are used to store the changes that are supposed t 
        self.input_gradients = np.zeros((self.input_layer, self.hidden_layer))

        self.output_gradients = np.zeros((self.hidden_layer, self.output_layer))
    #Used for rProp, when previous gradient is needed, other then that ignore if not dealing with rProp
        self.prev_input_gradients = np.zeros((self.input_layer, self.hidden_layer))

        self.prev_output_gradients = np.zeros((self.hidden_layer, self.output_layer))
    #Used also for rProp, this is what subtarcts the weights for learning 
        self.delta_weight_inputs = np.ones((self.input_layer, self.hidden_layer)) * 0.1

        self.delta_weight_outputs = np.ones((self.hidden_layer, self.output_layer)) * 0.1
    #Used for Delta bar Delta, these array update the weights and are intialized to 0.1
        self.learning_rate_inputs = np.ones((self.input_layer, self.hidden_layer)) * 0.0005

        self.learning_rate_outputs = np.ones((self.input_layer, self.hidden_layer)) * 0.0005

    #Used for storing values after they have been summed up together and went through the activation function
        self.activationinput = np.zeros((self.input_layer))

        self.activationhidden = np.zeros((self.hidden_layer))

        self.activationoutput = np.zeros((self.output_layer))

        self.targets = np.zeros((self.output_layer))

        #Used to print results to csv file
        self.store_results_training = [[0] * self.epochs for i in range(2)]

        self.store_results_test = [[0] for i in range(2)]

    # A simple feed forward using dot product to multiply each layer and either use the tanh
    def feed_forward(self, inputs):
        #fill the inputs into the input_layer.
        self.activationinput = inputs
        #Since dot product doesn't give 2 dimensions, I had to set two dimensions of the inputlayer.
        self.activationinput = np.resize(self.activationinput, (1, self.input_layer))

        #uses dot product to find hiddenlayer values and then adds git , finally then sending it through activation sigmoid function
        self.activationhidden = np.dot(self.activationinput, self.inputweights)
        self.activationhidden += self.input_bias
        self.activationhidden = sigmoids(self.activationhidden)
        #self.activationhidden = tanh(self.activationhidden)

        self.activationoutput = np.dot(self.activationhidden, self.outputweights)
        self.activationoutput += self.output_bias
        self.activationoutput = sigmoids(self.activationoutput)

   
    def back_prop(self,target):
        #set the targets to the appropraite values, this is what supervises the learning.
        self.targets = np.zeros((self.output_layer))
        number = self.calc_index(target)
        self.targets[number] = 1

         #Calculates the output_layer error by first finding error at outputlayer and using derivative sigmoid 
        output_errors = np.zeros((self.output_layer))
        error = -(self.targets - self.activationoutput)
        output_errors = sigmoids(self.activationoutput, True) * error
        output_errors_T = np.matrix.transpose(output_errors)
        

        hidden_errors = np.zeros((1, self.hidden_layer))
         #Calculates the error rates for the hidden_layer like the output_layer but using error from the output_layer
        for i in range(self.hidden_layer):
            h_error = 0
            for j in range(self.output_layer):
                h_error += output_errors[0][j] * self.outputweights[i][j]
            hidden_errors[0][i] = sigmoids(self.activationhidden[0][i], True) * h_error
        hidden_errors_T = np.matrix.transpose(hidden_errors)

        #updates the weights connected from the input_layer to the hidden_layer by using dot product and then transposing the matrix so that input_gradients rows and columns become the same as inputweights. 
        gradient = np.dot(hidden_errors_T, self.activationinput)
        self.input_gradients = np.matrix.transpose(gradient)
        self.input_gradients *= self.learning_rate 
        self.input_gradients += self.input_gradients * self.momentum 
        self.inputweights -= self.input_gradients

         #updates the weights connected from the hidden_layer to the output_layer .
        gradient = np.dot(output_errors_T, self.activationhidden)
        self.output_gradients = np.matrix.transpose(gradient)
        self.output_gradients *= self.learning_rate 
        self.output_gradients += self.output_gradients * self.momentum 
        self.outputweights -= self.output_gradients

        # Update the bias by multiplying errors at hiddenlayer and outputlayer then subtracting from each respective layers bias matrix.
        self.input_bias -= hidden_errors * self.learning_rate
        self.output_bias -= output_errors * self.learning_rate
    
    # Used to conduct backprop training and it randomly sends a line from training files until all lines of data have been sent and then 1 epoch is complete.
    def holdout_training(self, training_data):
        for j in range(15):
            self.inputweights = np.random.uniform(-0.5, 0.5, size=(self.input_layer, self.hidden_layer))
            self.outputweights = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer, self.output_layer))
            self.input_bias = np.random.uniform(-0.5, 0.5, size=(1, self.hidden_layer))
            self.output_bias = np.random.uniform(-0.5, 0.5, size=(1, self.output_layer))
            print("======== Run ", j, " ========")
            for i in range(self.epochs):
                self.correct_answer = 0
                #The random line to be sent into the feedforward.
                random_line = random.sample(range(len(training_data)), len(training_data))
                for k in range(len(training_data)):
                    self.feed_forward(training_data[random_line[k]])
                    self.accuracy(random_line[k])
                    self.back_prop(random_line[k])
                Accuracy = self.correct_answer / len(training_data)
                print("Epoch: ", i, "Accuracy: ", Accuracy)
                #print("Accuracy: ", Accuracy, " ", self.correct_answer)
                #Store the epoch number and accuracy for that epoch into a list to be printed out in csv later.
                # self.store_results_training[0][i] = i
                # self.store_results_training[1][i] = Accuracy
                # self.create_file_csv(self.store_results_training, "Backprop_training.csv")
                if Accuracy >= 0.95:
                    self.test()
                    break
    # Runs backprop through k-fold training method
    def k_fold_training(self, training_data, k):
        for q in range(15):
            average_accuracy_k_runs = np.zeros((k))
            self.inputweights = np.random.uniform(-0.5, 0.5, size=(self.input_layer, self.hidden_layer))
            self.outputweights = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer, self.output_layer))
            self.input_bias = np.random.uniform(-0.5, 0.5, size=(1, self.hidden_layer))
            self.output_bias = np.random.uniform(-0.5, 0.5, size=(1, self.output_layer))
            for epoch in range(self.epochs):
                #run for k times
                for i in range(k):
                    average_accuracy = np.zeros((k))
                    #intialize indicies in random order from 0 - training_data length in this case 7000.
                    random_line = random.sample(range(len(training_data)), len(training_data))
                    #split training data into k chunks
                    split_training_data = np.array_split(random_line,k)
                    #run k-1 times usually 9 times if k is 10.
                    for j in range(k - 1):
                        self.correct_answer = 0
                        #Go through data k-1 times, with training length/k.
                        for t in range(len(split_training_data[0])):
                            self.feed_forward(training_data[split_training_data[j][t]])
                            self.back_prop(split_training_data[j][t])
                            self.accuracy(split_training_data[j][t])
                        average_accuracy[j] = self.correct_answer / len(split_training_data[0]) 
                    # run 1 time to use last k chunk as validation data
                    for test in range(1):
                        self.correct_answer = 0
                        for w in range(len(split_training_data[0])):
                            self.feed_forward(training_data[split_training_data[k-1][w]])
                            self.accuracy(split_training_data[k-1][w])
                        # All the accuracies from each k run are stored in list
                        average_accuracy[k-1] = self.correct_answer / len(split_training_data[0]) 
                        # Then find the average of k runs by summing all k accuracies and dividing by k, that average_accuracy_k_runs.
                    average_accuracy_k_runs[i] = self.kfold_accuracy(average_accuracy,k)
                print("Epoch:", epoch, "Highest Average Accuracy of K runs: ", np.max(average_accuracy_k_runs))
            # self.store_results_training[0][epoch] = epoch
            # self.store_results_training[1][epoch] = np.max(average_accuracy_k_runs)
            # self.create_file_csv(self.store_results_training, "k_fold_training_backprop.csv")
                if np.amax(average_accuracy_k_runs) >= 0.95:
                    self.test()
                    break
    
    def kfold_accuracy(self, each_runs_accuracy, k):
        # sum all the values for each run and then divide by k
        sum = np.sum(each_runs_accuracy) / k
        return sum
    # calculates the summed gradients for each weight, this is for batch training.
    def compute_gradients(self, target):
        #set the targets to the appropraite values
        self.targets = np.zeros((self.output_layer))
        number = self.calc_index(target)
        self.targets[number] = 1

        #Calculates the output_layer error by first finding error at outputlayer and using derivative sigmoid, then the gradient
        output_errors = np.zeros((self.output_layer))
        error = -(self.targets - self.activationoutput)
        output_errors = sigmoids(self.activationoutput, True) * error
        output_errors_T = np.matrix.transpose(output_errors)
        gradient = np.dot(output_errors_T, self.activationhidden)
        self.output_gradients += np.matrix.transpose(gradient)

        hidden_errors = np.zeros((1, self.hidden_layer))
         #Calculates the error rates for the hidden_layer like the output_layer but using error from the output_layer
        for i in range(self.hidden_layer):
            h_error = 0
            for j in range(self.output_layer):
                h_error += output_errors[0][j] * self.outputweights[i][j]
            hidden_errors[0][i] = sigmoids(self.activationhidden[0][i], True) * h_error
        hidden_errors_T = np.matrix.transpose(hidden_errors)
        gradient = np.dot(hidden_errors_T, self.activationinput)
        self.input_gradients += np.matrix.transpose(gradient)

    def rProp(self):
        n_plus = 1.2
        n_minus = 0.5

        for i in range(self.input_layer):
            for j in range(self.hidden_layer):
                #Scenario 1 Gradient has not changed signs when compared to the previous gradient.
                if (self.input_gradients[i][j] > 0 and self.prev_input_gradients[i][j] > 0) or (self.input_gradients[i][j] < 0 and self.prev_input_gradients[i][j] < 0):
                    self.delta_weight_inputs[i][j] = self.delta_weight_inputs[i][j] * n_plus
                    if self.input_gradients[i][j] > 0:
                        self.inputweights[i][j] = self.inputweights[i][j] - self.delta_weight_inputs[i][j]
                    elif self.input_gradients[i][j] < 0:
                        self.inputweights[i][j] = self.inputweights[i][j] + self.delta_weight_inputs[i][j]
                    else:
                        self.inputweights[i][j] = self.inputweights[i][j]
                # Second scenerio when the sign of the current gradient has changed when compared to the previous gradient.
                if (self.input_gradients[i][j] < 0 and self.prev_input_gradients[i][j] > 0) or (self.input_gradients[i][j] > 0 and self.prev_input_gradients[i][j] < 0):
                    self.delta_weight_inputs[i][j] = self.delta_weight_inputs[i][j] * n_minus
                    self.input_gradients[i][j] = 0
                #Third scenerio when one of the gradients is 0, then just subtract or add delta weight depending on the gradient. 
                if self.input_gradients[i][j] == 0 or self.prev_input_gradients[i][j] == 0:
                    if self.input_gradients[i][j] > 0:
                        self.inputweights[i][j] = self.inputweights[i][j] - self.delta_weight_inputs[i][j]
                    elif self.input_gradients[i][j] < 0:
                        self.inputweights[i][j] = self.inputweights[i][j] + self.delta_weight_inputs[i][j]

        for i in range(self.hidden_layer):
            for j in range(self.output_layer):
                if (self.output_gradients[i][j] > 0 and self.prev_output_gradients[i][j] > 0) or (self.output_gradients[i][j] < 0 and self.prev_output_gradients[i][j] < 0):
                    self.delta_weight_outputs[i][j] = self.delta_weight_outputs[i][j] * n_plus
                    if self.output_gradients[i][j] > 0:
                        self.outputweights[i][j] = self.outputweights[i][j] - self.delta_weight_outputs[i][j]
                    elif self.output_gradients[i][j] < 0:
                        self.outputweights[i][j] = self.outputweights[i][j] + self.delta_weight_outputs[i][j]

                if (self.output_gradients[i][j] < 0 and self.prev_output_gradients[i][j] > 0) or (self.output_gradients[i][j] > 0 and self.prev_output_gradients[i][j] < 0):
                    self.delta_weight_outputs[i][j] = self.delta_weight_outputs[i][j] * n_minus
                    self.output_gradients[i][j] = 0

                if self.output_gradients[i][j] == 0 or self.prev_output_gradients[i][j] == 0:
                    if self.output_gradients[i][j] > 0:
                        self.outputweights[i][j] = self.outputweights[i][j] - self.delta_weight_outputs[i][j]
                    elif self.output_gradients[i][j] < 0:
                        self.outputweights[i][j] = self.outputweights[i][j] + self.delta_weight_outputs[i][j]
                    else:
                        self.inputweights[i][j] = self.inputweights[i][j]
        # Don't let the delta weight exceed 50 or go below 0.000001
        np.clip(self.delta_weight_inputs, 0.000001, 50)
        np.clip(self.delta_weight_outputs, 0.000001, 50)
        #Reset the current gradients and set the previous gradients to current gradients
        self.prev_output_gradients = np.copy(self.output_gradients)
        self.prev_input_gradients = np.copy(self.input_gradients)
        self.input_gradients = np.zeros((self.input_layer, self.hidden_layer))
        self.output_gradients = np.zeros((self.hidden_layer, self.output_layer))
    
    def delta_bar_delta(self):
        k_growth = 0.0001
        d_decay = 0.2

        for i in range(self.input_layer):
            for j in range(self.hidden_layer):
                #Scenerio 1 if the sign of the current gradient has changed form the previous gradient then multiply current learning rate of the weight with (1-D)
                if (self.input_gradients[i][j] < 0 and self.prev_input_gradients[i][j] > 0) or (self.input_gradients[i][j] > 0 and self.prev_input_gradients[i][j] < 0):

                    self.learning_rate_inputs[i][j] = self.learning_rate_inputs[i][j] * (1 - d_decay)
                    self.input_gradients[i][j] = self.input_gradients[i][j] * self.learning_rate_inputs[i][j]
                    self.inputweights[i][j] = self.inputweights[i][j] - self.input_gradients[i][j]
                #Scenario 2 if current gradient has not changed signs from the previous gradient then a k_growth to learning rate
                elif (self.input_gradients[i][j] > 0 and self.prev_input_gradients[i][j] > 0) or (self.input_gradients[i][j] < 0 and self.prev_input_gradients[i][j] < 0):

                    self.learning_rate_inputs[i][j] = self.learning_rate_inputs[i][j] + k_growth
                    self.input_gradients[i][j] = self.input_gradients[i][j] * self.learning_rate_inputs[i][j]
                    self.inputweights[i][j] = self.inputweights[i][j] - self.input_gradients[i][j]
                
        for i in range(self.hidden_layer):
            for j in range(self.output_layer):
                if (self.output_gradients[i][j] < 0 and self.prev_output_gradients[i][j] > 0) or (self.output_gradients[i][j] > 0 and self.prev_output_gradients[i][j] < 0):
                    
                    self.learning_rate_outputs[i][j] = self.learning_rate_outputs[i][j] * (1 - d_decay)
                    self.output_gradients[i][j] = self.output_gradients[i][j] * self.learning_rate_outputs[i][j]
                    self.outputweights[i][j] = self.outputweights[i][j] - self.output_gradients[i][j]

                elif (self.output_gradients[i][j] > 0 and self.prev_output_gradients[i][j] > 0) or (self.output_gradients[i][j] < 0 and self.prev_output_gradients[i][j] < 0):
                    self.learning_rate_outputs[i][j] = self.learning_rate_outputs[i][j] + k_growth
                    self.output_gradients[i][j] = self.output_gradients[i][j] * self.learning_rate_outputs[i][j]
                    self.outputweights[i][j] = self.outputweights[i][j] - self.output_gradients[i][j]

        # Don't let the delta weight exceed 50 or go below 0.000001
        np.clip(self.learning_rate_inputs, 0.0001, 0.001)
        np.clip(self.learning_rate_outputs, 0.0001, 0.001)
        #Reset the current gradients and set the previous gradients to current gradients
        self.prev_output_gradients = np.copy(self.output_gradients)
        self.prev_input_gradients = np.copy(self.input_gradients)
        self.input_gradients = np.zeros((self.input_layer, self.hidden_layer))
        self.output_gradients = np.zeros((self.hidden_layer, self.output_layer))

    # Used for rProp, delta bar delta and batch learning.
    def holdout_batch_training(self, training_data):
        for j in range(15):
            self.inputweights = np.random.uniform(-0.5, 0.5, size=(self.input_layer, self.hidden_layer))
            self.outputweights = np.random.uniform(-0.5, 0.5, size=(self.hidden_layer, self.output_layer))
            self.input_bias = np.random.uniform(-0.5, 0.5, size=(1, self.hidden_layer))
            self.output_bias = np.random.uniform(-0.5, 0.5, size=(1, self.output_layer))
            print("======== Run ", j, " ========")
            for i in range(self.epochs):
                self.correct_answer = 0
                random_line = random.sample(range(len(training_data)), len(training_data))
                for k in range(len(training_data)):
                    self.feed_forward(training_data[random_line[k]])
                    self.accuracy(random_line[k])
                    self.compute_gradients(random_line[k])
                #************ To use rProp or delta bar delta uncomment only one and then uncomment holdout_batch_training only************
                #self.rProp()
                self.delta_bar_delta()
                print("Epoch: ", i, "Accuracy: ", self.correct_answer / len(training_data))
                threshold = self.correct_answer / len(training_data)
                # self.store_results_training[0][i] = i
                # self.store_results_training[1][i] = threshold
                # self.create_file_csv(self.store_results_training, "batch_training.csv")

                if threshold >= 0.95:
                    self.test()
                    break
    #Used to test every training method after.
    def test(self):
        test_data = test_files()
        random_line = random.sample(range(len(test_data)), len(test_data))
        self.correct_answer = 0
        for i in range(len(test_data)):
            self.feed_forward(test_data[random_line[i]])
            self.accuracy(random_line[i], True)
        print("Total test data Accuracy: ", self.correct_answer / len(test_data))
        Accuracy = self.correct_answer / len(test_data)
        #self.store_results_test[0][0] = 1
        #self.store_results_test[1][0] = Accuracy
        #self.create_file_csv(self.store_results_test, "test_data.csv")

    
    #Calculates how accurate each epoch run is
    def accuracy(self, index, test = False):
        target = self.calc_index(index, test)
        if np.amax(self.activationoutput) > 0.50:
            if self.activationoutput[0][target] == np.amax(self.activationoutput):
                self.correct_answer += 1
    
    #Finds out what number is being fed to the system.
    def calc_index(self, index, test = False):
        if test == True:
            number = index / 400
            number = math.floor(number)
            return number

        number = index / 700
        number = math.floor(number)
        return number

    def create_file_csv(self,info, filename):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            writer.writerows(info)

#Used to load in training files
def training_files():
    data = []
    training_files = [i for i in file_list if 'train_' in i]

    for file_path in training_files:
        data.extend(np.loadtxt(file_path, delimiter=','))
    data = np.array(data)
    return data
#Used to load in test files.
def test_files():
    data = []
    test_files = [i for i in file_list if 'test_' in i]

    for file_path in test_files:
        data.extend(np.loadtxt(file_path, delimiter=','))
    data = np.array(data)
    return data

def main():
    #Note on how to use: Uncomment the training method you would like to use and only one, they will all run the testing data after training.
    #For batch training there are two algos Rprop and delta bar delta, you can only uncomment one at a time in batch training
    #Don't forget to uncomment whichever learning rule you want to use in batch training and then run batch training_files
    #For K-fold training just uncomment and run.
    #Lastly the let me explain the parameters for the neural net, I will list them off in order
    # Number Hidden layers, Number of output layers, number of epochs, Learning rate and Momentum.
    #NN = NeuralNetwork(25, 10, 300, 0.05, 0.08)
    #NN.holdout_batch_training(training_files())
    #NN.holdout_training(training_files())
    #NN.k_fold_training(training_files(), 10)
    data = test_files()
    print(data)


if __name__ == "__main__":
    main()


