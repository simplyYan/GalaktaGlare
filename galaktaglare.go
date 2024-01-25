package galaktaglare

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/ioutil"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"

	"github.com/pelletier/go-toml"
)

type GalaktaGlare struct {
	imageDB []string
}

func New() *GalaktaGlare {
	return &GalaktaGlare{}
}

func (gg *GalaktaGlare) ImageDB(folderPath string) error {
	files, err := ioutil.ReadDir(folderPath)
	if err != nil {
		return err
	}

	var imageDB []string
	for _, file := range files {
		if strings.HasSuffix(strings.ToLower(file.Name()), ".png") || strings.HasSuffix(strings.ToLower(file.Name()), ".jpg") {
			imageDB = append(imageDB, filepath.Join(folderPath, file.Name()))
		}
	}

	gg.imageDB = imageDB
	return nil
}

func (gg *GalaktaGlare) ImageScan(imagePath string) (float64, error) {
	img, err := loadImage(imagePath)
	if err != nil {
		return 0, err
	}

	asciiImage := imageToASCII(img)

	var maxSimilarity float64
	for _, dbImagePath := range gg.imageDB {
		dbImg, err := loadImage(dbImagePath)
		if err != nil {
			return 0, err
		}

		dbAsciiImage := imageToASCII(dbImg)
		similarity := compareImages(asciiImage, dbAsciiImage)

		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
	}

	return maxSimilarity, nil
}

func loadImage(filename string) (image.Image, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

func imageToASCII(img image.Image) string {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var asciiArt string

	for y := 0; y < height; y += 2 {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := uint8(0.2126*float64(r)+0.7152*float64(g)+0.0722*float64(b)) / 256
			asciiArt += string(grayToChar(gray))
		}
		asciiArt += "\n"
	}

	return asciiArt
}

func grayToChar(gray uint8) rune {
	chars := []rune("@%#*+=-:. ")

	index := int(gray) * (len(chars) - 1) / 256
	return chars[index]
}

func compareImages(asciiImage1, asciiImage2 string) float64 {
	if len(asciiImage1) != len(asciiImage2) {
		return 0.0
	}

	totalDifference := 0

	for i := 0; i < len(asciiImage1); i++ {
		if asciiImage1[i] != asciiImage2[i] {
			totalDifference++
		}
	}

	averageDifference := float64(totalDifference) / float64(len(asciiImage1))
	similarity := 1.0 - averageDifference

	return similarity
}

func (gg *GalaktaGlare) TextClassifier(text, config string) (string, error) {
	configData, err := toml.LoadFile(config)
	if err != nil {
		return "", err
	}

	categories := getCategories(configData)

	categoryScores := make(map[string]int)
	for category, words := range categories {
		categoryScores[category] = countMatchingWords(text, words)
	}

	maxScore := 0
	maxCategory := "Unknown"
	for category, score := range categoryScores {
		if score > maxScore {
			maxScore = score
			maxCategory = category
		}
	}

	return maxCategory, nil
}

func getCategories(config *toml.Tree) map[string][]string {
	categories := make(map[string][]string)

	for _, key := range config.KeysHashed() {
		if key.IsTable() {
			category := key.String()
			wordsArray := config.GetArray(category + ".words")
			if wordsArray != nil {
				var words []string
				for _, word := range wordsArray {
					words = append(words, word.(string))
				}
				categories[category] = words
			}
		}
	}

	return categories
}

func countMatchingWords(text string, words []string) int {
	count := 0
	lowercaseText := strings.ToLower(text)

	for _, word := range words {
		if strings.Contains(lowercaseText, word) {
			count++
		}
	}

	return count
}

func (gg *GalaktaGlare) AnomalyDetector(data []float64, threshold float64) []int {
	var anomalies []int

	mean, stdDev := calculateStatistics(data)

	upperLimit := mean + threshold*stdDev
	lowerLimit := mean - threshold*stdDev

	for i, value := range data {
		if value > upperLimit || value < lowerLimit {
			anomalies = append(anomalies, i)
		}
	}

	return anomalies
}

func calculateStatistics(data []float64) (float64, float64) {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	mean := sum / float64(len(data))

	var squaredDiffs float64
	for _, value := range data {
		diff := value - mean
		squaredDiffs += diff * diff
	}
	variance := squaredDiffs / float64(len(data))
	stdDev := math.Sqrt(variance)

	return mean, stdDev
}

type DataAnalysis struct{}

func (da *DataAnalysis) Mean(data []float64) float64 {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}

func (da *DataAnalysis) Median(data []float64) float64 {
	// Ordena os dados
	sortedData := make([]float64, len(data))
	copy(sortedData, data)
	sort.Float64s(sortedData)

	n := len(sortedData)
	if n%2 == 0 {
		middle1 := sortedData[n/2-1]
		middle2 := sortedData[n/2]
		return (middle1 + middle2) / 2.0
	}
	return sortedData[n/2]
}

func (da *DataAnalysis) Mode(data []float64) []float64 {
	// Conta a frequência de cada valor
	frequency := make(map[float64]int)
	for _, value := range data {
		frequency[value]++
	}

	var modeValues []float64
	maxFrequency := 0
	for value, freq := range frequency {
		if freq > maxFrequency {
			modeValues = []float64{value}
			maxFrequency = freq
		} else if freq == maxFrequency {
			modeValues = append(modeValues, value)
		}
	}

	return modeValues
}

func (gg *GalaktaGlare) ExtractEntities(text string) []string {
	re := regexp.MustCompile(`\b\w+\b`)
	words := re.FindAllString(text, -1)

	stopwords := map[string]bool{
		"the": true,
		"is":  true,
	}

	var entities []string
	for _, word := range words {
		word = strings.ToLower(word)
		if !stopwords[word] {
			entities = append(entities, word)
		}
	}

	return entities
}

func (gg *GalaktaGlare) ExtractEntities(text string, customStopwords map[string]bool) []string {
	predefinedStopwords := map[string]bool{
		"the": true,
		"is":  true,
	}

	stopwords := make(map[string]bool)
	for word := range predefinedStopwords {
		stopwords[word] = true
	}
	for word := range customStopwords {
		stopwords[word] = true
	}

	re := regexp.MustCompile(`\b\w+\b`)
	words := re.FindAllString(text, -1)

	var entities []string
	for _, word := range words {
		word = strings.ToLower(word)
		if !stopwords[word] {
			entities = append(entities, word)
		}
	}

	return entities
}

type SpeechConfig struct {
	Lang string `toml:"lang"`
	Text string `toml:"text"`
	Reco string `toml:"reco"`
}

// Speech converte uma configuração TOML para JSON e cria o arquivo ggtts.json.
func (gg *GalaktaGlare) Speech(tomlConfig string) error {
	// Decodificar o arquivo TOML
	var config SpeechConfig
	if err := toml.Unmarshal([]byte(tomlConfig), &config); err != nil {
		return fmt.Errorf("erro ao decodificar o arquivo TOML: %v", err)
	}

	// Converter para JSON
	jsonConfig, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("erro ao converter para JSON: %v", err)
	}

	// Criar ou sobrescrever o arquivo ggtts.json
	if err := ioutil.WriteFile("ggtts.json", jsonConfig, 0644); err != nil {
		return fmt.Errorf("erro ao escrever o arquivo ggtts.json: %v", err)
	}

	return nil
}

func (gg *GalaktaGlare) SpeechCfg() error {
	var zipFileName string
	var executableName string

	// Check the operating system
	switch runtime.GOOS {
	case "linux":
		zipFileName = "linux_speech.zip"
		executableName = "galaktatts"
	case "windows":
		zipFileName = "windows_speech.zip"
		executableName = "galaktatts.exe"
	default:
		return fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}

	// Open the ZIP file
	zipFile, err := zip.OpenReader(zipFileName)
	if err != nil {
		return fmt.Errorf("failed to open ZIP file: %v", err)
	}
	defer zipFile.Close()

	// Extract files to the current directory
	for _, file := range zipFile.File {
		err := extractFile(file)
		if err != nil {
			return fmt.Errorf("failed to extract file %s: %v", file.Name, err)
		}
	}

	fmt.Printf("Files successfully extracted from ZIP file: %s\n", zipFileName)

	// Execute the galaktatts executable
	err = runExecutable(executableName)
	if err != nil {
		return fmt.Errorf("failed to execute executable %s: %v", executableName, err)
	}

	return nil
}

func extractFile(file *zip.File) error {
	src, err := file.Open()
	if err != nil {
		return err
	}
	defer src.Close()

	dst, err := os.Create(file.Name)
	if err != nil {
		return err
	}
	defer dst.Close()

	_, err = io.Copy(dst, src)
	if err != nil {
		return err
	}

	return nil
}

func runExecutable(executableName string) error {
	cmd := exec.Command(executableName)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to execute %s: %v", executableName, err)
	}

	return nil
}
