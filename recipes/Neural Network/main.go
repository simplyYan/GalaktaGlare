package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	inputSize := 3
	hiddenLayerSize := 4
	outputSize := 2

	initialLearningRate := 0.01
	epochs := 1000
	lrSchedule := galaktaglare.LearningRateSchedule{InitialRate: initialLearningRate, Decay: 0.01}

	hiddenLayer := galaktaglare.NewDenseLayer(inputSize, hiddenLayerSize, galaktaglare.Sigmoid, 0.01, 0.01, 0.5, lrSchedule)
	outputLayer := galaktaglare.NewDenseLayer(hiddenLayerSize, outputSize, galaktaglare.Sigmoid, 0.01, 0.01, 0.5, lrSchedule)

	nn := galaktaglare.NewNeuralNetwork(hiddenLayer, outputLayer)

	inputs := [][]float64{
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 1.0},
	}
	targets := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
		{0.0, 1.0},
		{1.0, 0.0},
	}

	nn.Train(inputs, targets, initialLearningRate, epochs)

	trainingMonitor := galaktaglare.NewTrainingMonitor(epochs, 100)
	trainingMonitor.MonitorTraining(nn, inputs, targets, initialLearningRate)

	testInputs := [][]float64{
		{0.0, 0.0, 0.0},
		{1.0, 1.0, 0.0},
	}
	for _, input := range testInputs {
		predicted := nn.Predict(input)
		fmt.Printf("Input: %v, Predicted: %v\n", input, predicted)
	}
}
