package GalaktaGlareDATA

import (
    "math"
    "sort"
)

func Mean(numbers []float64) float64 {
    total := 0.0
    for _, num := range numbers {
        total += num
    }
    return total / float64(len(numbers))
}

func Median(numbers []float64) float64 {
    sort.Float64s(numbers)
    mid := len(numbers) / 2
    if len(numbers)%2 == 0 {
        return (numbers[mid-1] + numbers[mid]) / 2
    }
    return numbers[mid]
}

func StandardDeviation(numbers []float64) float64 {
    mean := Mean(numbers)
    variance := 0.0
    for _, num := range numbers {
        variance += math.Pow(num-mean, 2)
    }
    variance /= float64(len(numbers))
    return math.Sqrt(variance)
}

func Max(numbers []float64) float64 {
    max := numbers[0]
    for _, num := range numbers {
        if num > max {
            max = num
        }
    }
    return max
}

func Min(numbers []float64) float64 {
    min := numbers[0]
    for _, num := range numbers {
        if num < min {
            min = num
        }
    }
    return min
}

func Mode(numbers []float64) []float64 {
    frequency := make(map[float64]int)
    maxFrequency := 0
    for _, num := range numbers {
        frequency[num]++
        if frequency[num] > maxFrequency {
            maxFrequency = frequency[num]
        }
    }
    var modes []float64
    for num, freq := range frequency {
        if freq == maxFrequency {
            modes = append(modes, num)
        }
    }
    return modes
}

func Quartiles(numbers []float64) (float64, float64, float64) {
    sort.Float64s(numbers)
    n := len(numbers)
    q1 := Median(numbers[:n/2])
    q2 := Median(numbers)
    q3 := Median(numbers[(n + 1) / 2:])
    return q1, q2, q3
}

func Percentile(numbers []float64, p float64) float64 {
    sort.Float64s(numbers)
    n := float64(len(numbers))
    rank := p / 100 * (n - 1)
    lower := math.Floor(rank)
    upper := math.Ceil(rank)
    if lower == upper {
        return numbers[int(rank)]
    }
    return numbers[int(lower)] + (rank - lower) * (numbers[int(upper)] - numbers[int(lower)])
}

func Correlation(x, y []float64) float64 {
    if len(x) != len(y) {
        panic("Tamanho das listas de números não corresponde")
    }
    n := len(x)
    sumX, sumY, sumXY, sumXSquare, sumYSquare := 0.0, 0.0, 0.0, 0.0, 0.0
    for i := 0; i < n; i++ {
        sumX += x[i]
        sumY += y[i]
        sumXY += x[i] * y[i]
        sumXSquare += math.Pow(x[i], 2)
        sumYSquare += math.Pow(y[i], 2)
    }
    numerator := float64(n)*sumXY - sumX*sumY
    denominator := math.Sqrt((float64(n)*sumXSquare - math.Pow(sumX, 2)) * (float64(n)*sumYSquare - math.Pow(sumY, 2)))
    return numerator / denominator
}

func DetectAnomalies(numbers []float64, threshold float64) []float64 {
    anomalies := make([]float64, 0)
    mean := Mean(numbers)
    stdDev := StandardDeviation(numbers)
    for _, num := range numbers {
        zScore := (num - mean) / stdDev
        if math.Abs(zScore) > threshold {
            anomalies = append(anomalies, num)
        }
    }
    return anomalies
}