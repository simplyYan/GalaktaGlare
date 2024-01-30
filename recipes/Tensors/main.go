package main

import (
	"fmt"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	tensor := galaktaglare.NewTensor([]int{2, 3}) // Creates a tensor with shape 2x3
	tensor.Set(1.0, 0, 0)
	tensor.Set(2.0, 0, 1)
	tensor.Set(3.0, 0, 2)
	tensor.Set(4.0, 1, 0)
	tensor.Set(5.0, 1, 1)
	tensor.Set(6.0, 1, 2)
	sum := tensor.Get(0, 0) + tensor.Get(1, 1)
	fmt.Println(sum)
}
