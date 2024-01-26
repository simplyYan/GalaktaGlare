package galaktaglare

import (
	"archive/zip"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/corona10/goimagehash"
	"github.com/nfnt/resize"
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

func dHash(img image.Image) (uint64, error) {
	hash, err := goimagehash.DifferenceHash(img)
	if err != nil {
		return 0, err
	}
	return hash.GetHash(), nil
}

func (gg *GalaktaGlare) ImageScan(imagePath string) (float64, error) {
	img, err := loadImage(imagePath)
	if err != nil {
		return 0, err
	}

	hash1, err := dHash(img)
	if err != nil {
		return 0, err
	}

	var maxSimilarity float64
	var wg sync.WaitGroup
	similarityChan := make(chan float64, len(gg.imageDB))

	for _, dbImagePath := range gg.imageDB {
		wg.Add(1)
		go func(dbImagePath string) {
			defer wg.Done()
			dbImg, err := loadImage(dbImagePath)
			if err != nil {
				// handle error
				return
			}

			hash2, err := dHash(dbImg)
			if err != nil {
				// handle error
				return
			}

			similarity := hashSimilarity(hash1, hash2)
			similarityChan <- similarity
		}(dbImagePath)
	}

	go func() {
		wg.Wait()
		close(similarityChan)
	}()

	for similarity := range similarityChan {
		if similarity > maxSimilarity {
			maxSimilarity = similarity
		}
	}

	return maxSimilarity, nil
}

func hashSimilarity(hash1, hash2 uint64) float64 {
	hashStr1 := fmt.Sprintf("%064b", hash1)
	hashStr2 := fmt.Sprintf("%064b", hash2)

	distance := hammingDistance(hashStr1, hashStr2)
	normalizedDistance := float64(distance) / 64.0
	similarity := 1.0 - normalizedDistance
	return similarity
}

func hammingDistance(str1, str2 string) int {
	if len(str1) != len(str2) {
		return -1
	}

	distance := 0
	for i := 0; i < len(str1); i++ {
		if str1[i] != str2[i] {
			distance++
		}
	}

	return distance
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

	resizedImg := resize.Resize(1000, 0, img, resize.Lanczos3)

	return resizedImg, nil
}

func imageToASCII(img image.Image) string {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	var asciiArt string

	for y := 0; y < height; y += 2 {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := uint8((0.2126*float64(r) + 0.7152*float64(g) + 0.0722*float64(b)) / 256.0) // Convert to float64 before division
			asciiArt += string(grayToChar(gray))
		}
		asciiArt += "\n"
	}

	return asciiArt
}

func grayToChar(gray uint8) rune {
	chars := []rune("@%#*+=-:. ")

	index := int(gray) * (len(chars) - 1) / 255
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

	for _, key := range config.Keys() {
		value := config.Get(key)
		if _, ok := value.(map[string]interface{}); ok {
			wordsArray := config.GetArray(key + ".words")
			if wordsArray != nil {
				var words []string
				for _, word := range wordsArray.([]interface{}) {
					if str, ok := word.(string); ok {
						words = append(words, str)
					}
				}
				categories[key] = words
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

type Tensor struct {
	shape []int
	data  []float64
}

func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	return &Tensor{shape, data}
}

func (t *Tensor) Set(value float64, indices ...int) {
	if len(indices) != len(t.shape) {
		panic("Incorrect number of indexes")
	}
	flatIndex := t.flatIndex(indices...)
	t.data[flatIndex] = value
}

func (t *Tensor) Get(indices ...int) float64 {
	if len(indices) != len(t.shape) {
		panic("Incorrect number of indexes")
	}
	flatIndex := t.flatIndex(indices...)
	return t.data[flatIndex]
}

func (t *Tensor) flatIndex(indices ...int) int {
	flatIndex := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] >= t.shape[i] || indices[i] < 0 {
			panic("Index out of bounds")
		}
		flatIndex += indices[i] * stride
		stride *= t.shape[i]
	}
	return flatIndex
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

func (gg *GalaktaGlare) Speech(tomlConfig string) error {
	var config SpeechConfig
	if err := toml.Unmarshal([]byte(tomlConfig), &config); err != nil {
		return fmt.Errorf("error decoding TOML file: %v", err)
	}

	jsonConfig, err := json.MarshalIndent(struct {
		Lang string `json:"lang"`
		Text string `json:"text"`
		Reco string `json:"reco"`
	}{
		Lang: config.Lang,
		Text: config.Text,
		Reco: config.Reco,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("error when converting to JSON: %v", err)
	}

	if err := ioutil.WriteFile("ggtts.json", jsonConfig, 0644); err != nil {
		return fmt.Errorf("error writing file ggtts.json: %v", err)
	}

	return nil
}

func (gg *GalaktaGlare) SpeechCfg() error {
	var zipFileName string
	var executableName string

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

	zipFile, err := zip.OpenReader(zipFileName)
	if err != nil {
		return fmt.Errorf("failed to open ZIP file: %v", err)
	}
	defer zipFile.Close()

	for _, file := range zipFile.File {
		err := extractFile(file)
		if err != nil {
			return fmt.Errorf("failed to extract file %s: %v", file.Name, err)
		}
	}

	fmt.Printf("Files successfully extracted from ZIP file: %s\n", zipFileName)

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
	var cmd *exec.Cmd

	if runtime.GOOS == "linux" {
		cmd = exec.Command("chmod", "+x", executableName)
		err := cmd.Run()
		if err != nil {
			return fmt.Errorf("failure to give execute permission: %v", err)
		}
	}

	cmd = exec.Command(executableName)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("failure to execute %s: %v", executableName, err)
	}

	return nil
}

type ActivationFunc func(float64) float64

type DenseLayer struct {
	InputSize            int
	OutputSize           int
	Weights              [][]float64
	Biases               []float64
	Activation           ActivationFunc
	Inputs               []float64
	WeightDecayL1        float64
	WeightDecayL2        float64
	DropoutRate          float64
	DropoutMask          []float64
	LearningRateSchedule LearningRateSchedule
	BatchNorm            *BatchNormalization
}

type LearningRateSchedule struct {
	InitialRate float64
	Decay       float64
}

type BatchNormalization struct {
	Gamma    float64
	Beta     float64
	Mean     []float64
	Variance []float64
	Epsilon  float64
	Input    []float64
	Output   []float64
}

func NewBatchNormalization(gamma, beta, epsilon float64) *BatchNormalization {
	return &BatchNormalization{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: epsilon,
	}
}

func (bn *BatchNormalization) Forward(input []float64) []float64 {
	bn.Input = input

	// Calcule a média e a variância
	mean := 0.0
	for _, x := range input {
		mean += x
	}
	mean /= float64(len(input))

	variance := 0.0
	for _, x := range input {
		variance += math.Pow(x-mean, 2)
	}
	variance /= float64(len(input))

	// Normalize as entradas
	bn.Mean = append(bn.Mean, mean)
	bn.Variance = append(bn.Variance, variance)
	normalized := make([]float64, len(input))
	for i, x := range input {
		normalized[i] = (x - mean) / math.Sqrt(variance+bn.Epsilon)
	}

	// Scale e shift usando Gamma e Beta
	bn.Output = make([]float64, len(input))
	for i, x := range normalized {
		bn.Output[i] = bn.Gamma*x + bn.Beta
	}

	return bn.Output
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func (l *DenseLayer) GetAdjustedLearningRate(epoch int) float64 {
	return l.LearningRateSchedule.InitialRate / (1 + l.LearningRateSchedule.Decay*float64(epoch))
}

func NewDenseLayer(inputSize, outputSize int, activation ActivationFunc, weightDecayL1, weightDecayL2, dropoutRate float64, lrSchedule LearningRateSchedule) *DenseLayer {
	rand.Seed(time.Now().UnixNano())

	weights := make([][]float64, outputSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}
	}

	biases := make([]float64, outputSize)
	for i := range biases {
		biases[i] = rand.Float64()
	}

	return &DenseLayer{
		InputSize:            inputSize,
		OutputSize:           outputSize,
		Weights:              weights,
		Biases:               biases,
		Activation:           activation,
		WeightDecayL1:        weightDecayL1,
		WeightDecayL2:        weightDecayL2,
		DropoutRate:          dropoutRate,
		DropoutMask:          nil,
		LearningRateSchedule: lrSchedule,
	}
}

type NeuralNetwork struct {
	Layers []*DenseLayer
}

func NewNeuralNetwork(layers ...*DenseLayer) *NeuralNetwork {
	return &NeuralNetwork{Layers: layers}
}

func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

func (nn *NeuralNetwork) Train(inputs, targets [][]float64, initialLearningRate float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		adjustedLearningRate := initialLearningRate
		if len(nn.Layers) > 0 {
			adjustedLearningRate = nn.Layers[0].GetAdjustedLearningRate(epoch)
		}

		for i, input := range inputs {
			target := targets[i]

			output := nn.Predict(input)

			nn.Backpropagate(output, target, adjustedLearningRate)
		}
	}
}

func (l *DenseLayer) Forward(input []float64) []float64 {
	if l.DropoutRate > 0.0 {
		l.DropoutMask = make([]float64, len(input))
		for i := range input {
			if rand.Float64() > l.DropoutRate {
				l.DropoutMask[i] = 1.0
			} else {
				l.DropoutMask[i] = 0.0
			}
			input[i] *= l.DropoutMask[i]
		}
	}

	l.Inputs = make([]float64, len(input))
	copy(l.Inputs, input)

	output := make([]float64, l.OutputSize)
	for i := range output {
		for j := range input {
			output[i] += input[j] * l.Weights[i][j]
		}
		output[i] += l.Biases[i]
		output[i] = l.Activation(output[i])
	}

	if l.BatchNorm != nil {
		normalized := l.BatchNorm.Forward(output)
		copy(output, normalized)
	}

	return output
}

func (l *DenseLayer) Backpropagate(output, target []float64, learningRate float64) {
	if l.DropoutRate > 0.0 {
		for i := range output {
			output[i] *= l.DropoutMask[i]
		}
	}
	outputError := make([]float64, len(output))
	for i := range output {
		outputError[i] = output[i] - target[i]
	}

	for i := len(nn.Layers) - 1; i >= 0; i-- {
		layer := nn.Layers[i]

		activationGradient := make([]float64, len(output))
		for j := range output {
			activationGradient[j] = layer.Activation(output[j]) * (1 - layer.Activation(output[j]))
		}

		for j := range output {
			for k := range layer.Weights[j] {
				layer.Weights[j][k] -= learningRate * outputError[j] * activationGradient[j] * layer.Inputs[k]
			}
			layer.Biases[j] -= learningRate * outputError[j] * activationGradient[j]
		}

		outputError = nn.MultiplyMatrixVector(layer.Weights, outputError)
	}
}

func (nn *NeuralNetwork) MultiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	result := make([]float64, rows)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i] += matrix[i][j] * vector[j]
		}
	}

	return result
}

func (nn *NeuralNetwork) MultiplyVectors(v1 []float64, v2 []float64) []float64 {
	result := make([]float64, len(v1))
	for i := range v1 {
		result[i] = v1[i] * v2[i]
	}
	return result
}

func LoadModel(filename string) (*NeuralNetwork, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var nn NeuralNetwork
	if err := decoder.Decode(&nn); err != nil {
		return nil, err
	}

	fmt.Printf("Model loaded with %s\n", filename)
	return &nn, nil
}

func (nn *NeuralNetwork) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(nn); err != nil {
		return err
	}

	fmt.Printf("Model saved in %s\n", filename)
	return nil
}

type Variable struct {
	Value     float64
	Gradient  float64
	Children  []*Variable
	Operation string
}

func NewVariable(value float64, operation string) *Variable {
	return &Variable{
		Value:     value,
		Gradient:  0,
		Children:  make([]*Variable, 0),
		Operation: operation,
	}
}

func Add(a, b *Variable) *Variable {
	result := NewVariable(a.Value+b.Value, "Add")
	result.Children = append(result.Children, a, b)
	return result
}

func Mul(a, b *Variable) *Variable {
	result := NewVariable(a.Value*b.Value, "Mul")
	result.Children = append(result.Children, a, b)
	return result
}

func Sin(a *Variable) *Variable {
	result := NewVariable(math.Sin(a.Value), "Sin")
	result.Children = append(result.Children, a)
	return result
}

func Cos(a *Variable) *Variable {
	result := NewVariable(math.Cos(a.Value), "Cos")
	result.Children = append(result.Children, a)
	return result
}

func Backward(result *Variable) {
	result.Gradient = 1
	backwardHelper(result)
}

func backwardHelper(curr *Variable) {
	switch curr.Operation {
	case "Add":
		for _, child := range curr.Children {
			child.Gradient += curr.Gradient
			backwardHelper(child)
		}
	case "Mul":
		for _, child := range curr.Children {
			child.Gradient += curr.Gradient * productOfOthers(curr, child)
			backwardHelper(child)
		}
	case "Sin":
		child := curr.Children[0]
		child.Gradient += curr.Gradient * math.Cos(child.Value)
		backwardHelper(child)
	case "Cos":
		child := curr.Children[0]
		child.Gradient += -curr.Gradient * math.Sin(child.Value)
		backwardHelper(child)
	}
}

func productOfOthers(curr, exclude *Variable) float64 {
	product := 1.0
	for _, child := range curr.Children {
		if child != exclude {
			product *= child.Value
		}
	}
	return product
}

type TrainingMonitor struct {
	Epochs        int
	DisplayPeriod int
}

func NewTrainingMonitor(epochs, displayPeriod int) *TrainingMonitor {
	return &TrainingMonitor{
		Epochs:        epochs,
		DisplayPeriod: displayPeriod,
	}
}

func (tm *TrainingMonitor) MonitorTraining(nn *NeuralNetwork, inputs, targets [][]float64, initialLearningRate float64) {
	// Training loop
	for epoch := 0; epoch < tm.Epochs; epoch++ {
		adjustedLearningRate := nn.Layers[0].GetAdjustedLearningRate(epoch)

		// Training step
		for i, input := range inputs {
			target := targets[i]
			output := nn.Predict(input)

			nn.Backpropagate(output, target, adjustedLearningRate)
		}

		// Display metrics at specified intervals
		if (epoch+1)%tm.DisplayPeriod == 0 {
			accuracy, loss := tm.calculateMetrics(nn, inputs, targets)
			fmt.Printf("Epoch %d - Accuracy: %.2f%%, Loss: %.4f\n", epoch+1, accuracy*100, loss)
		}
	}
}

func (tm *TrainingMonitor) calculateMetrics(nn *NeuralNetwork, inputs, targets [][]float64) (accuracy, loss float64) {
	var correctPredictions int
	totalLoss := 0.0

	for i, input := range inputs {
		target := targets[i]
		predicted := nn.Predict(input)

		// Calculate accuracy
		if isPredictionCorrect(predicted, target) {
			correctPredictions++
		}

		// Calculate loss (using mean squared error)
		totalLoss += calculateLoss(predicted, target) // Example loss, replace with your own loss function
	}

	accuracy = float64(correctPredictions) / float64(len(inputs))
	loss = totalLoss / float64(len(inputs))

	return accuracy, loss
}

func isPredictionCorrect(predicted, target []float64) bool {
	// Comparing the index with the maximum value
	predictedLabel := argmax(predicted)
	targetLabel := argmax(target)

	return predictedLabel == targetLabel
}

func calculateLoss(predicted, target []float64) float64 {
	// Example loss: mean squared error
	loss := 0.0
	for i := range predicted {
		loss += math.Pow(predicted[i]-target[i], 2)
	}
	return loss / float64(len(predicted))
}

func argmax(values []float64) int {
	maxIndex := 0
	maxValue := values[0]

	for i, value := range values {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}

	return maxIndex
}
