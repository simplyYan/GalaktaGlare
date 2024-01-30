package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {

	gg := galaktaglare.New()
	stopwords := map[string]bool{
		"as":      true,
		"follows": true,
	}

	entities := gg.ExtractEntities("The anomalies found were as follows", stopwords)
	data := []float64{1.0, 1.2, 1.1, 1.5, 1.3, 1.4, 9.0, 1.2, 1.3, 1.1}
	threshold := 2.0

	anomalies := gg.AnomalyDetector(data, threshold)

	fmt.Println(entities, ":", anomalies)

}
