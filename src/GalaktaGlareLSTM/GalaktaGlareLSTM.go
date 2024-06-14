package GalaktaGlareLSTM

import (
	"strings"
)

type LSTM struct {
	knowledge map[string]string
}

func New() *LSTM {
	return &LSTM{
		knowledge: make(map[string]string),
	}
}

func (l *LSTM) Train(data string) {
	lines := strings.Split(data, "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.Split(line, "=")
		if len(parts) == 2 {
			question := strings.TrimSpace(parts[0])
			answer := strings.TrimSpace(parts[1])
			question = strings.Trim(question, "{}")
			answer = strings.Trim(answer, `[]"`)
			l.knowledge[question] = answer
		}
	}
}

func (l *LSTM) Run(question string) string {
	if answer, exists := l.knowledge[question]; exists {
		return answer
	}
	return "I don't know the answer to that."
}
