import math
import random
from array import array
import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.learningRate = 0.001

        self.weights_ih = np.random.uniform(-0.75, 0.75, (self.hidden, self.inputs))
        self.weights_oh = np.random.uniform(-0.75, 0.75, (self.outputs, self.hidden))

        self.bias_h = np.random.uniform(-0.5, 0.5, (self.hidden, 1))
        self.bias_o = np.random.uniform(-0.5, 0.5, (self.outputs, 1))

    def dsigmoid(self, num):
        return num * (num - 1)

    def sigmoid(self, num):
        z = np.exp(-1 * num)
        return 1 / (1 + z)

    def relu(self, input):
        if (input >= 0):
            return input
        return 0

    def drelu(self, input):
        return(input > 0) * 1

    def feedForward(self, input):
        hiddenResult = np.dot(self.weights_ih, input)
        biasResult = np.add(hiddenResult, self.bias_h)


        for data in range(len(biasResult)):
          for subData in range(len(biasResult[data])):
              biasResult[data, subData] = self.relu(biasResult[data, subData])

        output = np.dot(self.weights_oh, biasResult)
        np.add(output, self.bias_o)

        for data in range(len(output)):
            for subData in range(len(output[data])):
                output[data, subData] = self.sigmoid(output[data, subData])

        return output


    def train(self, input, targets):
        trainingOutput = self.feedForward(input)
        trainingOutputError = np.subtract(targets, trainingOutput)

        who_t = np.transpose(self.weights_oh)
        hidden = np.dot(who_t, trainingOutputError)

        for data in range(len(trainingOutput)):
            for subData in range(len(trainingOutput[data])):
                trainingOutput[data, subData] = self.dsigmoid(trainingOutput[data, subData])

        trainingOutputChain = np.multiply(trainingOutput, trainingOutputError)

        for data in range(len(trainingOutputChain)):
            for subData in range(len(trainingOutputChain[data])):
                trainingOutputChain[data, subData] = trainingOutputChain[data, subData] * self.learningRate

        np.add(self.bias_o, trainingOutputChain)
        trainingHidden = np.transpose(hidden)
        who_t_deltas = np.dot(trainingOutputChain, trainingHidden)

        for data in range(len(who_t_deltas)):
            for subData in range(len(who_t_deltas[data])):
                self.weights_oh[data, subData] = who_t_deltas[data, subData] + self.weights_oh[data, subData]

        hiddenGradient = hidden

        for data in range(len(hidden)):
            for subData in range(len(hidden[data])):
                hiddenGradient[data, subData] = self.drelu(hidden[data, subData])

        hiddenErrors = np.dot(who_t, trainingOutputError)
        hiddenGradient = np.multiply(hiddenErrors, hiddenGradient)

        for data in range(len(hiddenGradient)):
            for subData in range(len(hiddenGradient[data])):
                hiddenGradient[data, subData] = hiddenGradient[data, subData] * self.learningRate

        np.add(self.bias_h, hiddenGradient)
        trainingInputs = np.transpose(input)
        whi_t_deltas = np.dot(hiddenGradient, trainingInputs)

        for data in range(len(whi_t_deltas)):
            for subData in range(len(self.weights_ih[data])):
                self.weights_ih[data, subData] = whi_t_deltas[data] + self.weights_ih[data, subData]


def readData():
    dataset = pd.read_csv('F:\\prog\\NeuralNetwork\\archive\\smoking.csv')
    return dataset

if __name__ == '__main__':
    dataSet = readData()
    df = readData()
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    ann = NeuralNetwork(df.shape[1], df.shape[1], 1)

    dataSet.replace(to_replace='F', value=0, inplace=True)
    dataSet.replace(to_replace='M', value=1, inplace=True)
    dataSet.replace(to_replace='Y', value=1, inplace=True)
    dataSet.replace(to_replace='N', value=0, inplace=True)

    df.replace(to_replace='M', value=1, inplace=True)
    df.replace(to_replace='F', value=0, inplace=True)
    df.replace(to_replace='Y', value=1, inplace=True)
    df.replace(to_replace='N', value=0, inplace=True)

    for trainingData in range(5000):
        trainingIndex = random.randint(0, df.shape[0] - 1)
        ann.train(df.loc[trainingIndex], dataSet['smoking'].loc[trainingIndex])


    print(np.average(ann.feedForward(df.loc[2])))






