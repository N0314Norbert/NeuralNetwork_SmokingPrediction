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
        self.learningRate = 0.0001

        self.weights_ih = np.random.uniform(-0.75, 0.75, (self.hidden, self.inputs))
        self.weights_oh = np.random.uniform(-0.75, 0.75, (self.outputs, self.hidden))

        self.bias_h = np.random.uniform(-0.7, 0.7, (self.hidden, 1))
        self.bias_o = np.random.uniform(-0.7, 0.7, (self.outputs, 1))

    def dsigmoid(self, num):
        return num * (num - 1)

    def sigmoid(self, num):
        z = np.exp(-num)
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


        index = 0
        for data in biasResult:
          indexInner = 0
          for subData in data:
              biasResult[index, indexInner] = self.relu(biasResult[index, indexInner])
              indexInner += 1
          index+=1

        output = np.dot(self.weights_oh, biasResult)
        np.add(output, self.bias_o)

        index = 0
        for data in output:
            indexInner = 0
            for subData in data:
                output[index, indexInner] = self.sigmoid(output[index, indexInner])
                indexInner += 1
            index += 1

        return output


    def train(self, input, targets):
        trainingOutput = self.feedForward(input)
        trainingOutputError = np.subtract(targets, trainingOutput)

        who_t = np.transpose(self.weights_oh)
        hidden = np.dot(who_t, trainingOutputError)

        index = 0
        for data in trainingOutput:
            indexInner = 0
            for subData in data:
                trainingOutput[index, indexInner] = self.dsigmoid(trainingOutput[index, indexInner])
                indexInner += 1
            index += 1

        trainingOutputChain = np.multiply(trainingOutput, trainingOutputError)

        index = 0
        for data in trainingOutputChain:
            indexInner = 0
            for subData in data:
                trainingOutputChain[index, indexInner] = trainingOutputChain[index, indexInner] * self.learningRate
                indexInner += 1
            index += 1

        np.add(self.bias_o, trainingOutputChain)
        trainingHidden = np.transpose(hidden)
        who_t_deltas = np.dot(trainingOutputChain, trainingHidden)

        index = 0
        for data in who_t_deltas:
            indexInner = 0
            for subData in data:
                self.weights_oh[index, indexInner] = who_t_deltas[index, indexInner] + self.weights_oh[index, indexInner]
                indexInner += 1
            index += 1

        hiddenGradient = hidden

        index = 0
        for data in hidden:
            indexInner = 0
            for subData in data:
                hiddenGradient[index, indexInner] = self.drelu(hidden[index, indexInner])
                indexInner += 1
            index += 1

        hiddenErrors = np.dot(who_t, trainingOutputError)
        hiddenGradient = np.multiply(hiddenErrors, hiddenGradient)

        index = 0
        for data in hiddenGradient:
            indexInner = 0
            for subData in data:
                hiddenGradient[index, indexInner] = hiddenGradient[index, indexInner] * self.learningRate
                indexInner += 1
            index += 1

        np.add(self.bias_h, hiddenGradient)
        trainingInputs = np.transpose(input)
        whi_t_deltas = np.dot(hiddenGradient, trainingInputs)

        index = 0
        for data in whi_t_deltas:

            self.weights_ih[index] = whi_t_deltas[index] + self.weights_ih[index]
            index += 1


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

    for trainingData in range(2000):
        trainingIndex = random.randint(0, df.shape[0] - 1)
        ann.train(df.loc[trainingIndex], dataSet['smoking'].loc[trainingIndex])


    print(np.average(ann.feedForward(df.loc[6])))






