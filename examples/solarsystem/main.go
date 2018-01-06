package solarsystem

import "math"

/* Constants **********************************************************************************************************/
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

const (
	objectSun = iota
	objectJupiter
	objectSaturn
	objectUranus
	objectNeptune
	objectPluto
	objectNrOf
)

var objectMasses = [objectNrOf]float64{
	objectSun:     1.00000597682, // sun = 1 and then account for the inner planets
	objectJupiter: 0.000954786104043,
	objectSaturn:  0.000285583733151,
	objectUranus:  0.0000437273164546,
	objectNeptune: 0.0000517759138449,
	objectPluto:   1.0 / (1.3 * 100000000),
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
		DimPx: 0.00168318 * objectMasses[objectJupiter],
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
