import math
import random
import sys

NUM_INPUTS = 3                  # Input Nodes + Bias Input
NUM_PATTERNS = 4                # Input Patterns for XOr Experiment
NUM_HIDDEN = 4
NUM_EPOCHS = 200
LR_IH = 0.7                     # Learning Rate, Input -> Hidden Weights
LR_HO = 0.07                    # Learning Rate, Hidden -> Output Weights

# Data rescaled to range -1 to 1, w/ extra input value of 1 to act as bias
# Example: [Value1][Value2][Bias]
TRAINING_INPUT = [[1, -1, 1], [-1, 1, 1], [1, 1, 1], [-1, -1, 1]]

# The Output Must Lie In Range -1 to 1
TRAINING_OUTPUT = [1, 1, -1, -1]

class Backpropagation1:
    def __init__(self, numInputs, numPatterns, numHidden, numEpochs, i2hLearningRate, h2oLearningRate, inputValues, outputValues):
        self.mNumInputs = numInputs
        self.mNumPatterns = numPatterns
        self.mNumHidden = numHidden
        self.mNumEpochs = numEpochs
        self.mI2HLearningRate = i2hLearningRate
        self.mH2OLearningRate = h2oLearningRate
        self.hiddenVal = []     # Hidden Node Outputs
        self.weightsIH = []     # Input -> Hidden Weights
        self.weightsHO = []     # Hidden -> Output Weights
        self.trainInputs = inputValues
        self.trainOutput = outputValues     # "Actual" Output Values
        self.errThisPat = 0.0
        self.outPred = 0.0                  # "Expected" Output Values
        self.RMSerror = 0.0                 # Root Mean Squared Error
        return

    def initialize_arrays(self):
        # Initialize Weights to Random Values
        for j in range(self.mNumInputs):
            newRow = []
            for i in range(self.mNumHidden):
                self.weightsHO.append((random.random() - 0.5) / 2.0)
                weightValue = (random.random() - 0.5) / 5.0
                newRow.append(weightValue)
                sys.stdout.write("Weight = " + str(weightValue) + "\n")
            self.weightsIH.append(newRow)

        self.hiddenVal = [0.0] * self.mNumHidden

        return

    def calc_net(self, patNum):
        # Calculate Values for Hidden and Output Nodes
        for i in range(self.mNumHidden):
            self.hiddenVal[i] = 0.0
            for j in range(self.mNumInputs):
                self.hiddenVal[i] += (self.trainInputs[patNum][j] * self.weightsIH[j][i])

            self.hiddenVal[i] = math.tanh(self.hiddenVal[i])

        self.outPred = 0.0

        for i in range(self.mNumHidden):
            self.outPred += self.hiddenVal[i] * self.weightsHO[i]

        self.errThisPat = self.outPred - self.trainOutput[patNum] # Error = "Expected" - "Actual"
        return

    def adjust_hidden_to_output_weights(self):
        for i in range(self.mNumHidden):
            weightChange = self.mH2OLearningRate * self.errThisPat * self.hiddenVal[i]
            self.weightsHO[i] -= weightChange

            # Regularization of Output Weights
            if self.weightsHO[i] < -5.0:
                self.weightsHO[i] = -5.0
            elif self.weightsHO[i] > 5.0:
                self.weightsHO[i] = 5.0

        return

    def adjust_input_to_hidden_weights(self, patNum):
        for i in range(self.mNumHidden):
            for j in range(self.mNumInputs):
                x = 1 - math.pow(self.hiddenVal[i], 2)
                x = x * self.weightsHO[i] * self.errThisPat * self.mI2HLearningRate
                x = x * self.trainInputs[patNum][j]

                weightChange = x
                self.weightsIH[j][i] -= weightChange

        return

    def calculate_overall_error(self):
        errorValue = 0.0

        for i in range(self.mNumPatterns):
            self.calc_net(i)
            errorValue += math.pow(self.errThisPat, 2)

        errorValue /= self.mNumPatterns

        return math.sqrt(errorValue)

    def train_network(self):
        patNum = 0

        for j in range(self.mNumEpochs):
            for i in range(self.mNumPatterns):
                # Select Random Pattern
                patNum = random.randrange(0, 4)

                # Calculate Output and Error for That Pattern
                self.calc_net(patNum)

                # Adjust Weights
                self.adjust_hidden_to_output_weights()
                self.adjust_input_to_hidden_weights(patNum)

            self.RMSerror = self.calculate_overall_error()

            sys.stdout.write("Epoch = " + str(j) + " RMS Error = " + str(self.RMSerror) + "\n")

        return

    def display_results(self):
        for i in range(self.mNumPatterns):
            self.calc_net(i)
            sys.stdout.write("Pat = " + str(i + 1) + " Actual = " + str(self.trainOutput[i]) + " Neural Model = " + str(self.outPred) + "\n")
        return

if __name__ == '__main__':
    bp1 = Backpropagation1(NUM_INPUTS, NUM_PATTERNS, NUM_HIDDEN, NUM_EPOCHS, LR_IH, LR_HO, TRAINING_INPUT, TRAINING_OUTPUT)
    bp1.initialize_arrays()
    bp1.train_network()
    bp1.display_results()
