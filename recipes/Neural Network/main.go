package main

import (
	"fmt"
	"math/rand"
	"time"

	galaktaglare "github.com/simplyYan/GalaktaGlare"

	"github.com/pelletier/go-toml"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Define the training data
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Create the neural network
	inputSize := 2
	outputSize := 1
	activationFunc := galaktaglare.Sigmoid
	weightDecayL1 := 0.0
	weightDecayL2 := 0.0
	dropoutRate := 0.0
	lrSchedule := galaktaglare.LearningRateSchedule{
		InitialRate: 0.1,
		Decay:       0.0,
	}

	layer := galaktaglare.NewDenseLayer(inputSize, outputSize, activationFunc, weightDecayL1, weightDecayL2, dropoutRate, lrSchedule)
	neuralNetwork := galaktaglare.NewNeuralNetwork(layer)

	// Train the neural network
	epochs := 10000
	trainingMonitor := galaktaglare.NewTrainingMonitor(epochs, epochs/10)
	trainingMonitor.MonitorTraining(neuralNetwork, inputs, targets, lrSchedule.InitialRate)

	// Test the trained neural network
	fmt.Println("Testing the Neural Network:")
	for _, input := range inputs {
		prediction := neuralNetwork.Predict(input)
		fmt.Printf("Input: %v, Output: %.2f\n", input, prediction[0])
	}

	// Save the trained model
	modelFilename := "neural_network_model.gob"
	err := neuralNetwork.SaveModel(modelFilename)
	if err != nil {
		fmt.Println("Error saving template:", err)
	} else {
		fmt.Println("Model saved in", modelFilename)
	}
}
