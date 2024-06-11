package GalaktaGlareDT

import (
	"errors"
	"math"
)

type Node struct {
	IsLeaf       bool
	Prediction   interface{}
	SplitFeature int
	SplitValue   interface{}
	Left         *Node
	Right        *Node
}

type DecisionTree struct {
	Root *Node
}

func NewDecisionTree() *DecisionTree {
	return &DecisionTree{}
}

func (dt *DecisionTree) Fit(data [][]interface{}, labels []interface{}, features []int, maxDepth int) error {
	if len(data) == 0 || len(data) != len(labels) {
		return errors.New("dados ou rótulos inválidos")
	}
	dt.Root = dt.buildTree(data, labels, features, maxDepth, 1)
	return nil
}

func (dt *DecisionTree) buildTree(data [][]interface{}, labels []interface{}, features []int, maxDepth, currentDepth int) *Node {
	if len(data) == 0 || currentDepth >= maxDepth {
		return &Node{IsLeaf: true, Prediction: majorityVote(labels)}
	}

	bestFeature, bestValue := chooseBestFeatureToSplit(data, labels, features)
	leftData, leftLabels, rightData, rightLabels := splitData(data, labels, bestFeature, bestValue)

	leftChild := dt.buildTree(leftData, leftLabels, features, maxDepth, currentDepth+1)
	rightChild := dt.buildTree(rightData, rightLabels, features, maxDepth, currentDepth+1)

	return &Node{
		SplitFeature: bestFeature,
		SplitValue:   bestValue,
		Left:         leftChild,
		Right:        rightChild,
	}
}

func (dt *DecisionTree) Predict(input []interface{}) (interface{}, error) {
	currentNode := dt.Root
	for !currentNode.IsLeaf {
		value := input[currentNode.SplitFeature]
		if compareValues(value, currentNode.SplitValue) {
			currentNode = currentNode.Left
		} else {
			currentNode = currentNode.Right
		}
	}
	return currentNode.Prediction, nil
}

func allSame(items []interface{}) bool {
	first := items[0]
	for _, item := range items[1:] {
		if item != first {
			return false
		}
	}
	return true
}

func majorityVote(items []interface{}) interface{} {
	counts := make(map[interface{}]int)
	for _, item := range items {
		counts[item]++
	}
	var majority interface{}
	maxCount := 0
	for key, count := range counts {
		if count > maxCount {
			majority = key
			maxCount = count
		}
	}
	return majority
}

func chooseBestFeatureToSplit(data [][]interface{}, labels []interface{}, features []int) (int, interface{}) {
	bestFeature := -1
	var bestValue interface{}
	bestImpurity := math.Inf(1)

	for _, featureIndex := range features {
		for _, row := range data {
			value := row[featureIndex]
			_, leftLabels, _, rightLabels := splitData(data, labels, featureIndex, value)
			impurity := calculateImpurity(leftLabels, rightLabels)

			if impurity < bestImpurity {
				bestImpurity = impurity
				bestFeature = featureIndex
				bestValue = value
			}
		}
	}

	return bestFeature, bestValue
}

func calculateImpurity(labels ...[]interface{}) float64 {
	totalSamples := 0
	labelCounts := make(map[interface{}]int)

	for _, labelSet := range labels {
		for _, label := range labelSet {
			totalSamples++
			labelCounts[label]++
		}
	}

	entropy := 0.0
	for _, count := range labelCounts {
		probability := float64(count) / float64(totalSamples)
		entropy -= probability * math.Log2(probability)
	}

	return entropy
}

func splitData(data [][]interface{}, labels []interface{}, featureIndex int, value interface{}) ([][]interface{}, []interface{}, [][]interface{}, []interface{}) {
	leftData, rightData := [][]interface{}{}, [][]interface{}{}
	leftLabels, rightLabels := []interface{}{}, []interface{}{}
	for i, row := range data {
		if compareValues(row[featureIndex], value) {
			leftData = append(leftData, row)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightData = append(rightData, row)
			rightLabels = append(rightLabels, labels[i])
		}
	}
	return leftData, leftLabels, rightData, rightLabels
}

func compareValues(a, b interface{}) bool {
	switch a.(type) {
	case int:
		return a.(int) < b.(int)
	case float64:
		return a.(float64) < b.(float64)
	case string:
		return a.(string) < b.(string)
	case bool:
		return !a.(bool) && b.(bool) 
	default:
		return false 
	}
}