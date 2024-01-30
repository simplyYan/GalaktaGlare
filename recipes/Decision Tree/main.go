package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	dataTraining := []galaktaglare.Example{
		{Features: []float64{10, 1}, Label: "No Spam"},
		{Features: []float64{50, 1}, Label: "Spam"},
		{Features: []float64{5, 0}, Label: "No Spam"},
		{Features: []float64{30, 1}, Label: "Spam"},
	}

	treeDecision := galaktaglare.DecisionTree{}
	treeDecision.Train(dataTraining, 3, 2)

	newFeatures := []float64{20, 1}
	forecast := treeDecision.Predict(newFeatures)

	fmt.Printf("Forecast for %v: %s\n", newFeatures, forecast)
}
