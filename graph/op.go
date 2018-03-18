package graph

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// DimSizer is any type (typically a tensor.Shape) that allows querying for a dimension size given an input dimension.
type DimSizer interface {
	DimSize(int) (int, error)
}

// ShapesToDimSizers is a convenience function to convert a slice of tensor.Shape to a slice of DimSizer
func ShapesToDimSizers(shapes []tensor.Shape) []DimSizer {
	retVal := make([]DimSizer, len(shapes))
	for i, s := range shapes {
		retVal[i] = s
	}
	return retVal
}

// DimSizersToShapes is a convenience function to convert a slice of DimSizer to a slice of tensor.Shape. It will return an error if any of them isn't a tensor.Shape
func DimSizersToShapes(ds []DimSizer) ([]tensor.Shape, error) {
	retVal := make([]tensor.Shape, len(ds))
	var ok bool
	for i, d := range ds {
		if retVal[i], ok = d.(tensor.Shape); !ok {
			return nil, errors.Errorf("Dimsizer %d is not a Shape.", i)
		}
	}
	return retVal, nil
}

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//		OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
type Op interface {
	/* Graph Building Related Methods */

	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
	Arity() int

	// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
	Type() hm.Type

	// returns the output shape as a function of the inputs
	InferShape(...DimSizer) (tensor.Shape, error)

	/* Machine related */

	// executes the op
	Do(...values.Value) (values.Value, error)

	/* Analysis Related Methods */

	// indicates if the Op will return a pointer (allowing possible inplace edits) or by value
	// if it's false, the return value of the Op will be a copy of its input
	ReturnsPtr() bool

	// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool

	// overwriteInput() is a method which states which input the output will be overwriting.
	// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
	// The method returns an int instead of a bool because potentially different operations may be allowed
	// to overwrite certain inputs. For example, consider an operation to increment a value:
	// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
	// the retVal of overwriteInput() will be 0 (inputs[0]).
	// -1 is returned if overwriting of input is disallowed
	OverwritesInput() int

	/* Other methods */
	WriteHash(h hash.Hash)
	Hashcode() uint32
	fmt.Stringer
}

// A UnaryOp is an Op that takes only one input
type UnaryOp interface {
	Op

	IsUnary() bool
}

// A BinaryOp is an Op that takes only two inputs
type BinaryOp interface {
	Op

	IsBinary() bool
}

type BestDoer interface {
	Op

	BestDo(prealloc values.Value, vals ...values.Value) (values.Value, error)
}

// A NoRetOp is an Op that reads a value, but does not return any value. It's a representation of a not-pure function
type NoRetOp interface {
	Op

	ReturnsNothing() bool
}

// An ADOp is an Op that supports automatic differentiation.
type ADOp interface {
	Op

	DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error
}

// A SDOp is an Op that supports symbolic differentiation
type SDOp interface {
	Op

	// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
	// returns []bool to indicate which input it is differentiable to
	DiffWRT(inputs int) []bool

	// SymDiff symbolically differentiates the op
	SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error)
}

// ReductionOp changes the shape of the node
type ReductionOp interface {
	Op

	IsReduction() bool
}

// IncrDoer increments the toIncr with the result of doing
type IncrDoer interface {
	IncrDo(toIncr values.Value, inputs ...values.Value) error
}

// UsePreallocDoer is an op that works when a preallocated value is provided
type UsePreallocDoer interface {
	UsePreallocDo(prealloc values.Value, inputs ...values.Value) (values.Value, error)
}

// UnsafeDoer is an op that will overwrite the underlying value.
type UnsafeDoer interface {
	UnsafeDo(inputs ...values.Value) (values.Value, error)
}

// CUDADoer uses CUDA to perform the Op.
type CUDADoer interface {
	CUDADo(extern External, dev Device, prealloc values.Value, inputs ...values.Value) (retVal values.Value, err error)
}

// CLDoer uses OpenCL to perform the Op. As of now, there are NO Ops that support OpenCL
type CLDoer interface {
	CLDo(inputs ...values.Value) (values.Value, error)
}

type CUDAADOp interface {
	ADOp
	CUDADoDiff(extern External, dev Device, inputs Nodes, output *Node) error
}

type StmtOp interface {
	Op
	IsStmt() bool
}
