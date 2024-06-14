package galaktaglaretts

import (
    "fmt"
    "os/exec"
    "runtime"
)

func RecognizeSpeech() (string, error) {
    var cmd *exec.Cmd

    switch runtime.GOOS {
    case "linux":
        cmd = exec.Command("bash", "scripts/recognize_linux.sh")
    case "darwin":
        cmd = exec.Command("bash", "scripts/recognize_macos.sh")
    case "windows":
        cmd = exec.Command("cmd", "/C", "scripts/recognize_windows.bat")
    default:
        return "", fmt.Errorf("unsupported platform")
    }

    output, err := cmd.CombinedOutput()
    if err != nil {
        return "", fmt.Errorf("failed to recognize speech: %v, output: %s", err, output)
    }

    return string(output), nil
}
