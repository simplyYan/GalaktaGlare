package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	textToClassify := "This is an example of a text that will be classified. I am a go programmer. func printf"

	// And the path to the TOML configuration file with the categories and keywords
	configFilePath := "textcls.toml"

	// Create a GalaktaGlare instance
	gg := galaktaglare.New()

	// Call the TextClassifier function with the text and path of the configuration file
	category, err := gg.TextClassifier(textToClassify, configFilePath)
	if err != nil {
		fmt.Println("Error when classifying text:", err)
	} else {
		fmt.Println("The text was classified as:", category)
	}
}
