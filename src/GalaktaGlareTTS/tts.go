package galaktaglaretts

import (
    "fmt"
    "os/exec"
    "runtime"
)

func TextToSpeech(text string) error {
    var cmd *exec.Cmd

    switch runtime.GOOS {
    case "linux":
        cmd = exec.Command("bash", "scripts/tts_linux.sh", text)
    case "darwin":
        cmd = exec.Command("bash", "scripts/tts_macos.sh", text)
    case "windows":
        cmd = exec.Command("cmd", "/C", "scripts/tts_windows.bat", text)
    default:
        return fmt.Errorf("unsupported platform")
    }

    output, err := cmd.CombinedOutput()
    if err != nil {
        return fmt.Errorf("failed to execute TTS: %v, output: %s", err, output)
    }

    return nil
}
