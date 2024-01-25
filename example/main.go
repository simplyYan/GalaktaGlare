package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	gg := galaktaglare.New()
	err := gg.ImageDB("./db/img")
	if err != nil {
		fmt.Println("Erro ao carregar o banco de dados de imagens:", err)
		return
	}
	similarity, err := gg.ImageScan("./test.png")
	if err != nil {
		fmt.Println("Erro ao comparar imagens:", err)
		return
	}

	fmt.Println("Similaridade da imagem:", similarity)
}
