package main

import (
	"fmt"
	G "github.com/chewxy/gorgonia"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"io/ioutil"
	"testing"
)

func TestNewSymplecticMatrix(t *testing.T) {
	assert := assert.New(t)

	sympl := NewSymplecticMatrix(tensor.Float64, 4)

	correct := []float64{0, 0, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, 0, 0}
	assert.Equal(correct, sympl.Float64s())
}

func TestNewWeightedDiagMatrix(t *testing.T) {
	assert := assert.New(t)

	// One weight in Slice
	diagSlice := []float64{42}

	diag := NewWeightedDiagMatrix(2, diagSlice)

	correct := []float64{42, 0, 0, 42}
	assert.Equal(correct, diag.Float64s())

	// One weight as Scalar
	diagfloat := float64(42)

	diag = NewWeightedDiagMatrix(2, diagfloat)

	correct = []float64{42, 0, 0, 42}
	assert.Equal(correct, diag.Float64s())

	// Two weights for 2x2 matrix

	diagSlice = []float64{1, 2}

	diag = NewWeightedDiagMatrix(2, diagSlice)

	correct = []float64{1, 0, 0, 2}
	assert.Equal(correct, diag.Float64s())

	// Two weights for 4x4 matrix
	diagSlice = []float64{1, 2}

	diag = NewWeightedDiagMatrix(4, diagSlice)

	correct = []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 2, 0,
		0, 0, 0, 2,
	}
	assert.Equal(correct, diag.Float64s())

	// Two weights for 6x6 matrix
	diagSlice = []float64{1, 2}

	diag = NewWeightedDiagMatrix(6, diagSlice)

	correct = []float64{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0,
		0, 0, 0, 2, 0, 0,
		0, 0, 0, 0, 2, 0,
		0, 0, 0, 0, 0, 2,
	}
	assert.Equal(correct, diag.Float64s())
}

func TestNewMassWeightedSumVector(t *testing.T) {
	assert := assert.New(t)

	vec := NewMassWeightedSumVector([]float64{1, 2, 3, 4}, 6)

	correct := []float64{2, 3, 4, 6, 8, 12}
	assert.Equal(correct, vec.Float64s())
}

func TestNumberOfEntries(t *testing.T) {
	assert := assert.New(t)

	correct := int(15)
	assert.Equal(correct, BinomialChooseTwoOutOf(6))
}

func TestNewDifferenceGeneratorMatrix(t *testing.T) {
	assert := assert.New(t)

	DiffGenMatrix := NewDifferenceGeneratorMatrix(3, 2)

	correct := []float64{
		1, 0, -1, 0, 0, 0,
		0, 1, 0, -1, 0, 0,
		1, 0, 0, 0, -1, 0,
		0, 1, 0, 0, 0, -1,
		0, 0, 1, 0, -1, 0,
		0, 0, 0, 1, 0, -1,
	}

	assert.Equal(correct, DiffGenMatrix.Float64s())
}

func TestNewPartialVectorSumMatrix(t *testing.T) {
	assert := assert.New(t)

	PartVecSumMatrix := NewPartialVectorSumMatrix(2, 6)

	correct := []float64{
		1, 1, 0, 0, 0, 0,
		0, 0, 1, 1, 0, 0,
		0, 0, 0, 0, 1, 1,
	}

	assert.Equal(correct, PartVecSumMatrix.Float64s())
}

func TestConstDerivative(t *testing.T) {
	assert := assert.New(t)

	// Build the graph
	g := G.NewGraph()

	x := G.NewVector(g, tensor.Float64, G.WithName("x"), G.WithShape(3), G.WithName("x"))
	a := G.NewConstant(tensor.New(tensor.WithBacking([]float64{2.0, 2.0, 2.0}), tensor.WithShape(3)), G.WithName("a"))

	b := G.NewScalar(g, tensor.Float64, G.WithName("b"))

	xT := tensor.New(tensor.WithBacking([]float64{1, 1, 1}), tensor.WithShape(3))

	y, err := G.Mul(x, a)

	z, err := G.Mul(y, b)

	dz, err := G.Grad(z, x)

	ioutil.WriteFile("solarSystem.dot", []byte(g.ToDot()), 0644)

	prog, locMap, err := G.CompileFunction(g, G.Nodes{x}, G.Nodes{y, dz[0]})
	handleError(err)

	machine := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	machine.Let(x, xT)
	machine.Let(b, -0.5)
	for turns := 0; turns < 4; turns++ {
		machine.Reset()

		err = machine.RunAll()
		handleError(err)

		fmt.Println("x", x.Value())
		fmt.Println("y", y.Value())
		fmt.Println("a", a.Value())
		fmt.Println("dz", dz[0].Value())
		fmt.Println()
	}

	correct := []float64{1, 1, 1}

	assert.Equal(correct, dz[0].Value().Data().([]float64))

}
