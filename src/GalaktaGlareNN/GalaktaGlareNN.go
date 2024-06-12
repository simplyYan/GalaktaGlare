package galaktaglarenn

import (
    "math"
    "math/rand"
)

type NeuralNetwork struct {
    inputNeurons  int
    hiddenNeurons int
    outputNeurons int
    weightsInput  [][]float64
    weightsOutput [][]float64
}

func New(input, hidden, output int) *NeuralNetwork {
    weightsInput := make([][]float64, input)
    for i := range weightsInput {
        weightsInput[i] = make([]float64, hidden)
        for j := range weightsInput[i] {
            weightsInput[i][j] = rand.Float64() - 0.5
        }
    }

    weightsOutput := make([][]float64, hidden)
    for i := range weightsOutput {
        weightsOutput[i] = make([]float64, output)
        for j := range weightsOutput[i] {
            weightsOutput[i][j] = rand.Float64() - 0.5
        }
    }

    return &NeuralNetwork{
        inputNeurons:  input,
        hiddenNeurons: hidden,
        outputNeurons: output,
        weightsInput:  weightsInput,
        weightsOutput: weightsOutput,
    }
}

func sigmoid(x float64) float64 {
    return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
    return x * (1 - x)
}

func (nn *NeuralNetwork) Train(inputs, targets [][]float64, iterations int, learningRate float64) {
    for i := 0; i < iterations; i++ {
        for n := 0; n < len(inputs); n++ {
            // Feedforward process
            hiddenLayer := make([]float64, nn.hiddenNeurons)
            for j := 0; j < nn.hiddenNeurons; j++ {
                var sum float64
                for k := 0; k < nn.inputNeurons; k++ {
                    sum += inputs[n][k] * nn.weightsInput[k][j]
                }
                hiddenLayer[j] = sigmoid(sum)
            }

            outputLayer := make([]float64, nn.outputNeurons)
            for j := 0; j < nn.outputNeurons; j++ {
                var sum float64
                for k := 0; k < nn.hiddenNeurons; k++ {
                    sum += hiddenLayer[k] * nn.weightsOutput[k][j]
                }
                outputLayer[j] = sigmoid(sum)
            }

            // Backpropagation process
            outputErrors := make([]float64, nn.outputNeurons)
            for j := 0; j < nn.outputNeurons; j++ {
                outputErrors[j] = targets[n][j] - outputLayer[j]
            }

            outputGradients := make([]float64, nn.outputNeurons)
            for j := 0; j < nn.outputNeurons; j++ {
                outputGradients[j] = outputErrors[j] * sigmoidDerivative(outputLayer[j])
            }

            hiddenErrors := make([]float64, nn.hiddenNeurons)
            for j := 0; j < nn.hiddenNeurons; j++ {
                var error float64
                for k := 0; k < nn.outputNeurons; k++ {
                    error += outputGradients[k] * nn.weightsOutput[j][k]
                }
                hiddenErrors[j] = error
            }

            hiddenGradients := make([]float64, nn.hiddenNeurons)
            for j := 0; j < nn.hiddenNeurons; j++ {
                hiddenGradients[j] = hiddenErrors[j] * sigmoidDerivative(hiddenLayer[j])
            }

            // Update weights
            for j := 0; j < nn.hiddenNeurons; j++ {
                for k := 0; k < nn.outputNeurons; k++ {
                    change := outputGradients[k] * hiddenLayer[j] * learningRate
                    nn.weightsOutput[j][k] += change
                }
            }

            for j := 0; j < nn.inputNeurons; j++ {
                for k := 0; k < nn.hiddenNeurons; k++ {
                    change := hiddenGradients[k] * inputs[n][j] * learningRate
                    nn.weightsInput[j][k] += change
                }
            }
        }
    }
}


func (nn *NeuralNetwork) Predict(input []float64) []float64 {
    hiddenLayer := make([]float64, nn.hiddenNeurons)
    for j := 0; j < nn.hiddenNeurons; j++ {
        var sum float64
        for k := 0; k < nn.inputNeurons; k++ {
            sum += input[k] * nn.weightsInput[k][j]
        }
        hiddenLayer[j] = sigmoid(sum)
    }

    outputLayer := make([]float64, nn.outputNeurons)
    for j := 0; j < nn.outputNeurons; j++ {
        var sum float64
        for k := 0; k < nn.hiddenNeurons; k++ {
            sum += hiddenLayer[k] * nn.weightsOutput[k][j]
        }
        outputLayer[j] = sigmoid(sum)
    }

    return outputLayer
}
