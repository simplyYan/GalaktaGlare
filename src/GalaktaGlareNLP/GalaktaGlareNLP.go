package GalaktaGlareNLP

import (
    "strings"
    "unicode"
)

func Lowercase(text string) string {
    return strings.ToLower(text)
}

func RemovePunctuation(text string) string {
    var sb strings.Builder
    for _, char := range text {
        if !unicode.IsPunct(char) {
            sb.WriteRune(char)
        }
    }
    return sb.String()
}

func Tokenize(text string) []string {
    return strings.Fields(text)
}

func RemoveStopWords(tokens []string, stopWords map[string]bool) []string {
    var result []string
    for _, token := range tokens {
        if !stopWords[token] {
            result = append(result, token)
        }
    }
    return result
}

func Lemmatize(word string, lemmatizationRules map[string]string) string {
    if lemma, ok := lemmatizationRules[word]; ok {
        return lemma
    }
    return word 
}

func Stem(word string, stemmingRules map[string]string) string {
    if stem, ok := stemmingRules[word]; ok {
        return stem
    }
    return word 
}