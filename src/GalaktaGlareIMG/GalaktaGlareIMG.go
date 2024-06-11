package galaktaglareimg

import (
	"errors"
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
)

func LoadImage(path string) (image.Image, error) {
	img, err := imaging.Open(path)
	if err != nil {
		return nil, err
	}
	return img, nil
}

func SaveImage(img image.Image, path string) error {
	return imaging.Save(img, path)
}

func ResizeImage(img image.Image, width, height int) image.Image {
	return imaging.Resize(img, width, height, imaging.Lanczos)
}

func ConvertToGrayscale(img image.Image) image.Image {
	return imaging.Grayscale(img)
}

func CompareImages(img1, img2 image.Image) (float64, error) {
	bounds1 := img1.Bounds()
	bounds2 := img2.Bounds()

	if bounds1 != bounds2 {
		return 0, errors.New("images must have the same dimensions")
	}

	var sum float64
	var count int

	for y := bounds1.Min.Y; y < bounds1.Max.Y; y++ {
		for x := bounds1.Min.X; x < bounds1.Max.X; x++ {
			r1, g1, b1, _ := img1.At(x, y).RGBA()
			r2, g2, b2, _ := img2.At(x, y).RGBA()
			dr := float64(r1) - float64(r2)
			dg := float64(g1) - float64(g2)
			db := float64(b1) - float64(b2)
			sum += dr*dr + dg*dg + db*db
			count++
		}
	}

	mse := sum / float64(count)
	return 10 * math.Log10(65535*65535/mse), nil
}

func ImageToGrayscaleArray(img image.Image) [][]uint8 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	grayscaleArray := make([][]uint8, height)
	for y := 0; y < height; y++ {
		grayscaleArray[y] = make([]uint8, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			gray := color.GrayModel.Convert(color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 255})
			grayscaleArray[y][x] = gray.(color.Gray).Y
		}
	}
	return grayscaleArray
}