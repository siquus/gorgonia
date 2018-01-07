package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	G "github.com/chewxy/gorgonia"
	"gorgonia.org/tensor"
	"io/ioutil"
	"log"
	"math"
	"math/big"
	"os"
	"os/exec"
	"reflect"
)

/* Constants **********************************************************************************************************/

const ConstNodesUse = false

const simulationTime = 200000.0
const timeStep = 1.0

// See "Geometric Numerical Integration" p. 13ff for the data for "The Outer Solar System"

var GravConst float64 = 2.95912208286 * math.Pow10(-4)

const (
	DimQx = iota
	DimQy
	DimQz
	DimPx
	DimPy
	DimPz
	DimNrOf
)

const DimQNrOf = DimQz + 1
const DimPNrOf = DimPz - DimQz

const (
	objectSun = iota
	objectJupiter
	objectSaturn
	objectUranus
	objectNeptune
	objectPluto
	objectNrOf
)

var objectNames = [objectNrOf]string{
	objectSun:     "Sun",
	objectJupiter: "Jupiter",
	objectSaturn:  "Saturn",
	objectUranus:  "Uranus",
	objectNeptune: "Neptune",
	objectPluto:   "Pluto",
}

var objectMasses = [objectNrOf]float64{
	objectSun:     1.00000597682, // sun = 1 and then account for the inner planets
	objectJupiter: 0.000954786104043,
	objectSaturn:  0.000285583733151,
	objectUranus:  0.0000437273164546,
	objectNeptune: 0.0000517759138449,
	objectPluto:   1.0 / (1.3 * math.Pow10(8)),
}

var objectInitData = [objectNrOf][DimNrOf]float64{
	objectSun: {
		DimQx: 0,
		DimQy: 0,
		DimQz: 0,
		DimPx: 0 * objectMasses[objectSun],
		DimPy: 0 * objectMasses[objectSun],
		DimPz: 0 * objectMasses[objectSun],
	},

	objectJupiter: {
		DimQx: -3.5023653,
		DimQy: -3.8169847,
		DimQz: -1.5507963,
		DimPx: 0.00565429 * objectMasses[objectJupiter],
		DimPy: -0.00412490 * objectMasses[objectJupiter],
		DimPz: -0.00190589 * objectMasses[objectJupiter],
	},

	objectSaturn: {
		DimQx: 9.0755314,
		DimQy: -3.0458353,
		DimQz: -1.6483708,
		DimPx: 0.00168318 * objectMasses[objectSaturn],
		DimPy: 0.00483525 * objectMasses[objectSaturn],
		DimPz: 0.00192462 * objectMasses[objectSaturn],
	},

	objectUranus: {
		DimQx: 8.3101420,
		DimQy: -16.2901086,
		DimQz: -7.2521278,
		DimPx: 0.00354178 * objectMasses[objectUranus],
		DimPy: 0.00137102 * objectMasses[objectUranus],
		DimPz: 0.00055029 * objectMasses[objectUranus],
	},

	objectNeptune: {
		DimQx: 11.4707666,
		DimQy: -25.7294829,
		DimQz: -10.8169456,
		DimPx: 0.00288930 * objectMasses[objectNeptune],
		DimPy: 0.00114527 * objectMasses[objectNeptune],
		DimPz: 0.00039677 * objectMasses[objectNeptune],
	},

	objectPluto: {
		DimQx: -15.5387357,
		DimQy: -25.2225594,
		DimQz: -3.1902382,
		DimPx: 0.00276725 * objectMasses[objectPluto],
		DimPy: -0.00170702 * objectMasses[objectPluto],
		DimPz: -0.00136504 * objectMasses[objectPluto],
	},
}

func main() {
	// Create a single vector (q_1x, q_1y, q1_z, q2x, ... , p_1x, p_1y, ...) from objectInitData
	objectInitSlice := make([]float64, objectNrOf*DimNrOf)

	for object, coordSlice := range objectInitData {
		for dim, coordValue := range coordSlice {
			if dim > DimQz { // it's a DimP..
				objectInitSlice[objectNrOf*DimQNrOf+object*DimPNrOf+dim-DimQNrOf] = coordValue
			} else { // it's a DimQ..
				objectInitSlice[object*DimQNrOf+dim] = coordValue
			}
		}
	}

	// x ... coordinate in phase space
	xT := tensor.New(tensor.WithBacking(objectInitSlice), tensor.WithShape(objectNrOf*DimNrOf))

	// Build the graph
	g := G.NewGraph()

	// Generate the Hamiltonian
	// H =           T         -                   V
	//   = \Sum_i p_i^2 / 2m_i - G * \Sum_{i<j} m_i*m_j / |q_i - q_j|

	x := G.NewVector(g, tensor.Float64, G.WithName("x"), G.WithShape(objectNrOf*DimNrOf))

	// T part of the Hamiltonian: \Sum_i p_i^2 / 2m_i
	p, err := G.Slice(x, G.S(objectNrOf*DimQNrOf, objectNrOf*DimNrOf))
	handleError(err)

	// Create Diag. Matrix Diag(1/2m_1, 1/2m_1, 1/2m_1, 1/2m_2, ...)
	pWeightSlice := make([]float64, len(objectMasses))
	for index := range pWeightSlice {
		pWeightSlice[index] = 1.0 / (2.0 * objectMasses[index])
	}

	// TODO: Make it constant again
	var diagMasses *G.Node
	diagMassesT := NewWeightedDiagMatrix(objectNrOf*DimPNrOf, pWeightSlice)
	if ConstNodesUse {
		diagMasses = G.NewConstant(diagMassesT, G.WithName("Diag Masses"))
	} else {
		diagMasses = G.NewTensor(g, G.Float64, 2, G.WithShape(diagMassesT.Shape()...), G.WithName("Diag Masses"), G.WithValue(diagMassesT))
	}

	pWeighted, err := G.Mul(diagMasses, p)
	handleError(err)

	T, err := G.Mul(p, pWeighted)

	handleError(err)

	// V part of the Hamiltonian:  G * \Sum_{i<j} m_i*m_j / |q_i - q_j|

	q, err := G.Slice(x, G.S(0, objectNrOf*DimQNrOf))
	handleError(err)

	// Create vector of unique differences
	// (q_1_1 - q_2_1, q_1_2 - q_2_2, q_1_3 - q_2_3, q_1_1 - q_3_1, ...), where q_[ObjectNr]_[DimQ]
	// TODO: Make it constant again
	var diffGenMatrix *G.Node
	diffGenMatrixT := NewDifferenceGeneratorMatrix(objectNrOf, DimQNrOf)
	if ConstNodesUse {
		diffGenMatrix = G.NewConstant(diffGenMatrixT, G.WithName("diffGenMatrix"))
	} else {
		diffGenMatrix = G.NewTensor(g, G.Float64, 2, G.WithShape(diffGenMatrixT.Shape()...), G.WithName("diffGenMatrix"), G.WithValue(diffGenMatrixT))
	}

	qDiffs, err := G.Mul(diffGenMatrix, q)
	handleError(err)

	// Square the vector
	qDiffsSquared, err := G.HadamardProd(qDiffs, qDiffs)
	handleError(err)

	// Create vector of |q_i - q_j|^2 terms: (|q_1 - q_2|^2, |q_1 - q_3|^2, ...)
	partVecSumMatrixT := NewPartialVectorSumMatrix(DimQNrOf, qDiffsSquared.Shape()[0])
	var partVecSumMatrix *G.Node
	if ConstNodesUse {
		partVecSumMatrix = G.NewConstant(partVecSumMatrixT, G.WithName("partVecSumMatrix")) // TODO: Make it constant again
	} else {
		partVecSumMatrix = G.NewTensor(g, G.Float64, 2, G.WithShape(partVecSumMatrixT.Shape()...), G.WithName("partVecSumMatrix"), G.WithValue(partVecSumMatrixT))
	}

	objectDistsSquared, err := G.Mul(partVecSumMatrix, qDiffsSquared)
	handleError(err)

	// Create vector of |q_i - q_j| terms: (|q_1 - q_2|, |q_1 - q_3|, ...)
	objectDists, err := G.Sqrt(objectDistsSquared)
	handleError(err)

	// Create a vector of 1 / |q_i - q_j| terms:
	objectDistsInv, err := G.Inverse(objectDists)
	handleError(err)

	// Create V / G = \Sum_{i<j} m_i*m_j / |q_i - q_j|
	massWeightedSumVecT := NewMassWeightedSumVector(objectMasses[:], objectDistsInv.Shape()[0])
	var massWeightedSumVec *G.Node
	if ConstNodesUse {
		massWeightedSumVec = G.NewConstant(massWeightedSumVecT, G.WithName("massWeightedSumVec")) // TODO: Make it constant again
	} else {

		massWeightedSumVec = G.NewTensor(g, G.Float64, 1, G.WithShape(massWeightedSumVecT.Shape()...), G.WithName("massWeightedSumVec"), G.WithValue(massWeightedSumVecT))
	}

	VmissingG, err := G.Mul(massWeightedSumVec, objectDistsInv)
	handleError(err)

	// Create V = G * \Sum_{i<j} m_i*m_j / |q_i - q_j|
	var gravConst *G.Node
	if ConstNodesUse {
		gravConst = G.NewConstant(GravConst, G.WithName("Gravitational Constant")) // TODO: Make it constant again
	} else {
		gravConst = G.NewScalar(g, G.Float64, G.WithName("Gravitational Constant"), G.WithValue(GravConst))
	}

	V, err := G.Mul(VmissingG, gravConst)
	handleError(err)

	// Finally, create H = T - V
	H, err := G.Sub(T, V)
	handleError(err)

	// Calculate the Hamiltonian Vector Field XH = J * dH, where J is the symplectic matrix
	dH, err := G.Grad(H, x) // TODO: When restricting to two objects, this panics (shape 0 scalar vs shape {1} scalar?!)
	handleError(err)

	JT := NewSymplecticMatrix(tensor.Float64, objectNrOf*DimNrOf)
	var J *G.Node
	if ConstNodesUse {
		J = G.NewConstant(JT, G.WithName("J")) // TODO: Make this constant again
	} else {

		J = G.NewTensor(g, G.Float64, 2, G.WithShape(JT.Shape()...), G.WithName("J"), G.WithValue(JT))
	}

	XH, err := G.Mul(J, dH[0])
	handleError(err)

	var ones *G.Node
	onesT := NewWeightedDiagMatrix(XH.Shape()[0], 1.0)
	if ConstNodesUse {
		ones = G.NewConstant(onesT, G.WithName("ones")) // TODO: Make this constant again
	} else {
		ones = G.NewTensor(g, G.Float64, 2, G.WithShape(onesT.Shape()...), G.WithName("ones"), G.WithValue(onesT))
	}

	XHprintout, err := G.Mul(ones, XH)

	// Calculate new coordinates
	var epsilon *G.Node
	if ConstNodesUse {
		epsilon = G.NewConstant(timeStep) // TODO: Make this constant again
	} else {
		epsilon = G.NewScalar(g, G.Float64, G.WithName("Epsilon"), G.WithValue(timeStep))
	}

	step, err := G.Mul(XH, epsilon)
	handleError(err)

	xUpdated, err := G.Add(x, step)
	handleError(err)

	// Create Picture of Graph
	// e.g. do
	// # dot solarSystem.dot -Tpng -o solarSystem.png
	// To convert to png
	ioutil.WriteFile("solarSystem.dot", []byte(g.ToDot()), 0644)

	// Run the Graph
	prog, locMap, err := G.CompileFunction(g, G.Nodes{x}, G.Nodes{xUpdated, XHprintout})
	handleError(err)

	machine := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	timeSteps := int(math.Ceil(simulationTime / timeStep))
	trajectoryPointsNrOf := timeSteps * objectNrOf * DimQNrOf
	trajectoryX := make([]float64, trajectoryPointsNrOf, trajectoryPointsNrOf)
	trajectoryTime := make([]float64, timeSteps, timeSteps)
	time := 0.0

	machine.Let(x, xT)

	for stepNr := 0; stepNr < timeSteps; stepNr++ {
		machine.Reset()

		err = machine.RunAll()
		handleError(err)

		machine.Set(x, xUpdated)

		time += timeStep

		// Save to trajectory data
		trajectoryTime[stepNr] = time
		xFloats := x.Value().Data().([]float64)
		copy(trajectoryX[stepNr*objectNrOf*DimQNrOf:], xFloats[0:objectNrOf*DimQNrOf])

		/*
				fmt.Println("H", H.Value())
				fmt.Println("dH", dH[0].Value())
				fmt.Println("XH", XH.Value())

			xPluto := xFloats[objectPluto*DimQNrOf : objectPluto*DimQNrOf+DimQNrOf]
			fmt.Println("Pluto", xPluto)

			xUpdFloats := x.Value().Data().([]float64)
			xUpdPluto := xUpdFloats[objectPluto*DimQNrOf : objectPluto*DimQNrOf+DimQNrOf]
			fmt.Println("Pluto", xUpdPluto)
		*/
		//XHFloats := XH.Value().Data().([]float64)
		//dHFloats := dH[0].Value().Data().([]float64)
		//fmt.Println("Jupiter qdot", XHFloats[3:6])
		//fmt.Println("Jupiter pdot", XHFloats[9:12])
		//fmt.Println("Saturn qdot", XHFloats[6:9])
		//fmt.Println("Saturn pdot", XHFloats[12:15])
		//fmt.Println("Jupiter dH/dp", dHFloats[9:12])
		//fmt.Println("Jupiter dH/dq", dHFloats[3:6])

		//qDiffsFloats := qDiffs.Value().Data().([]float64)
		//fmt.Println("qDiffs", qDiffsFloats)

		//objectDistsFl := XHTest.Value().Data().([]float64)
		//fmt.Println("objDists", objectDistsFl)

		/*
			XHFloats := XHprintout.Value().Data().([]float64)

			fmt.Println("q", xFloats[3:6])
			fmt.Println("qdot", xFloats[12]/objectMasses[objectJupiter], xFloats[13]/objectMasses[objectJupiter], xFloats[14]/objectMasses[objectJupiter])
			fmt.Println("qdotdot", XHFloats[12]/objectMasses[objectJupiter], XHFloats[13]/objectMasses[objectJupiter], XHFloats[14]/objectMasses[objectJupiter])
		*/
	}

	// Export Data
	err = TrajectoryJsonExport(trajectoryTime, trajectoryX)
	handleError(err)

	// Show graph
	fmt.Println("Plotting...")
	filepath := "/home/ommo/GoglandProjects/src/github.com/chewxy/gorgonia/examples/solarsystem/trajectoryPlot.py"
	cmd := exec.Command(filepath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	log.Println(cmd.Run())

}

func TrajectoryJsonExport(time, x []float64) error {
	file, err := os.Create("trajectoryData.json")
	defer file.Close()

	if nil != err {
		return err
	}

	writer := bufio.NewWriter(file)
	encoder := json.NewEncoder(writer)

	// Save general information
	type SimulationInformation struct {
		Name       string
		Objects    [objectNrOf]string
		Dimensions int
		Time       []float64
		Trajectory []float64
	}

	simulationInformation := SimulationInformation{
		Name:       "Simulation Information",
		Objects:    objectNames,
		Dimensions: DimQNrOf,
		Time:       time,
		Trajectory: x,
	}

	err = encoder.Encode(simulationInformation)

	if nil != err {
		return err
	}

	// Save Object Information
	type ObjectInformation struct {
		Name       string
		Trajectory [DimQNrOf]float64
	}

	writer.Flush()
	return nil
}

func BinomialChooseTwoOutOf(n int) int {
	var retBigInt big.Int
	retBigInt.Binomial(int64(n), 2)
	retInt := int(retBigInt.Int64())

	return retInt
}

// Creates the vector that multiplies with the vector ( 1 / |q_1 - q_2|, 1 / |q_1 - q_3, ...)
// to form \Sum_{i<j} m_i*m_j / |q_i - q_j|
// i.e.
// (m_1 * m_2, m_1 * m_3, ...)
//
func NewMassWeightedSumVector(masses []float64, vectorLen int) *tensor.Dense {

	if vectorLen != BinomialChooseTwoOutOf(len(masses)) {
		panic("Vector Length needs to equal to ''[Masses Length] choose two''")
	}

	vectorSlice := make([]float64, vectorLen)

	index := 0
	for mass1 := 0; mass1 < len(masses); mass1++ {
		for mass2 := mass1 + 1; mass2 < len(masses); mass2++ {
			vectorSlice[index] = masses[mass1] * masses[mass2]
			index++
		}
	}

	return tensor.New(tensor.WithBacking(vectorSlice), tensor.WithShape(vectorLen))
}

// Creates a matrix that sums every sumLen vector elements, i.e. for sumLen = 3
//
// 1 1 1 0 0 0 0 0 0 0 0 ...
// 0 0 0 1 1 1 0 0 0 0 0 ...
// ...
//
// TODO: Make this sparse
func NewPartialVectorSumMatrix(sumLen, vectorLen int) *tensor.Dense {
	if 0 != vectorLen%sumLen {
		panic("Vector length needs to be divisible by sum length.")
	}

	rows := vectorLen / sumLen
	columns := vectorLen

	vectorSumMatrixSlice := make([]float64, rows*columns)

	for row := 0; row < rows; row++ {
		for sum := 0; sum < sumLen; sum++ {
			column := sumLen*row + sum
			vectorSumMatrixSlice[row*columns+column] = 1
		}
	}

	return tensor.New(tensor.WithBacking(vectorSumMatrixSlice), tensor.WithShape(rows, columns))
}

// Creates a matrix that will generate a vector of unique differences (q_1_1 - q_2_1, q_1_2 - q_2_2, ..),
// where q_[ObjectNr]_[DimQ] :
//
// 1 0 0 -1  0  ...
// 0 1 0  0 -1  ...
// ....
// i.e. it will then be multiplied on the right by (q_1_1, q_1_2, q_1_3, q_2_1, ...)
// TODO: Make this sparse
func NewDifferenceGeneratorMatrix(objectLen, qLen int) *tensor.Dense {
	DiffGenMatrixRows := qLen * BinomialChooseTwoOutOf(objectLen)
	DiffGenMatrixColumns := objectLen * qLen
	DiffGenMatrixSlice := make([]float64, DiffGenMatrixRows*DiffGenMatrixColumns)

	currentRow := 0
	for object1 := 0; object1 < objectLen; object1++ {
		for object2 := object1 + 1; object2 < objectLen; object2++ {
			for coord := 0; coord < qLen; coord++ {
				// object1 Coordinate
				colObj1 := object1*qLen + coord
				DiffGenMatrixSlice[currentRow*DiffGenMatrixColumns+colObj1] = 1

				// - object2 Coordinate
				colObj2 := object2*qLen + coord
				DiffGenMatrixSlice[currentRow*DiffGenMatrixColumns+colObj2] = -1

				currentRow++
			}
		}
	}

	return tensor.New(tensor.WithBacking(DiffGenMatrixSlice), tensor.WithShape(DiffGenMatrixRows, DiffGenMatrixColumns))
}

// Creates a diagonal matrix from the list of weights:
// - If one weight w is supplied, it will create diag(w, w, ..., w)
// - If two weights are supplied...diag(w1, w1, .... w1, w2, ..., w2, w2)
// and so forth.
// TODO: Make this sparse
func NewWeightedDiagMatrix(dimensions int, weights interface{}) *tensor.Dense {

	weightsType := reflect.TypeOf(weights)

	switch weightsType.Kind() {
	case reflect.Float64:
		weightFloat := weights.(float64)
		matrixSlice := make([]float64, dimensions*dimensions)

		for index := range matrixSlice {
			row := int(index / dimensions) // square matrix, no more columns than rows
			column := int(index - row*dimensions)

			if row == column {
				matrixSlice[index] = weightFloat
			}
		}

		return tensor.New(tensor.WithBacking(matrixSlice), tensor.WithShape(dimensions, dimensions))

	case reflect.Slice:
		switch weightsType.Elem().Kind() {
		case reflect.Float64:
			weightsSlice := weights.([]float64)
			if 0 != dimensions%len(weightsSlice) {
				panic("Number of weights does not divide number of diag. elements")
			}

			matrixSlice := make([]float64, dimensions*dimensions)

			var weightSelect int
			for index := 0; index < len(matrixSlice); index++ {
				row := int(index / dimensions) // square matrix, no more columns than rows
				column := int(index - row*dimensions)

				if row == column {
					if (0 == row%(dimensions/len(weightsSlice))) && (0 != row) {
						weightSelect++
					}

					matrixSlice[index] = weightsSlice[weightSelect]
				}
			}

			return tensor.New(tensor.WithBacking(matrixSlice), tensor.WithShape(dimensions, dimensions))

		default:
			panic("Element Kind not implemented")
		}

	default:
		panic("Kind not implemented")
	}

	panic("Unintendedly reached end of function")

	return nil
}

// Creates a Matrix containing the Symplectic Matrix for the standard coordinates
// (q1, q2, ..., qn, p1, p2, ..., pn). It has the form:
//
//      0 id
//    -id  0
//
// TODO: Make this sparse
func NewSymplecticMatrix(t tensor.Dtype, dimensions int) *tensor.Dense {
	if 0 != dimensions%2 {
		panic("Dimension needs to be an even number.")
	}

	if t.Kind() != reflect.Float64 {
		panic("Type not supported")
	}

	// Create the corresponding slice
	sliceLen := dimensions * dimensions
	matrixSlice := make([]float64, sliceLen, sliceLen)

	for index := range matrixSlice {
		row := int(index / dimensions) // square matrix, no more columns than rows
		column := int(index - row*dimensions)

		if row < dimensions/2 {
			if column == row+dimensions/2 {
				matrixSlice[index] = 1
			}
		} else {
			if column == row-dimensions/2 {
				matrixSlice[index] = -1
			}
		}
	}

	return tensor.New(tensor.WithBacking(matrixSlice), tensor.WithShape(dimensions, dimensions))
}

func handleError(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
