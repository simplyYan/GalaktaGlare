package main

import (
	"fmt"
	"os/exec"

	"github.com/simplyYan/GalaktaGlare"
)

func main() {
	gg := galaktaglare.New()
	gg.Speech(`
lang = "en"
text = "Hello World"
`)
	gg.SpeechCfg()
	cmd := exec.Command("/bin/sh", "runner.sh")
	out, err := cmd.CombinedOutput()
	if err != nil {
		panic(err)
	}

	fmt.Println(string(out))
}
