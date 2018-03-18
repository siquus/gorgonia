package stdops

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/graph"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// a constant is an unchanging value. I think everyone would know what a constant is
// a constant op is an op that creates a constant. It is also a values.Value of a constant value
type constant interface {
	graph.Op

	isconstant() bool
	Value() values.Value
}

type constantScalar struct {
	v values.Scalar
}

func (c constantScalar) Arity() int                                   { return 0 }
func (c constantScalar) Type() hm.Type                                { return TypeOf(c.v) }
func (c constantScalar) InferShape(...DimSizer) (tensor.Shape, error) { return scalarShape, nil }
func (c constantScalar) ReturnsPtr() bool                             { return false }
func (c constantScalar) CallsExtern() bool                            { return false }
func (c constantScalar) OverwritesInput() int                         { return -1 }
func (c constantScalar) DiffWRT(i int) []bool                         { return nil }
func (c constantScalar) SymDiff(Nodes, *Node, *Node) (Nodes, error)   { return nil, nil }

func (c constantScalar) Do(...values.Value) (values.Value, error) { return c.v, nil }
func (c constantScalar) String() string                           { return fmt.Sprintf("const %s", c.v) }

func (c constantScalar) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "const %v: %v", TypeOf(c.v), c.v)
}

func (c constantScalar) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c constantScalar) isconstant() bool    { return true }
func (c constantScalar) Value() values.Value { return c.v }

type constantTensor struct {
	v tensor.Tensor
}

func (c constantTensor) Arity() int                                   { return 1 }
func (c constantTensor) Type() hm.Type                                { return TypeOf(c.v) }
func (c constantTensor) InferShape(...DimSizer) (tensor.Shape, error) { return c.v.Shape(), nil }

// danger! The only reason why this is the case is because matrices may be too large. copying is costly.
// constants should return value but for the sake of memory, we're going to return pointers
func (c constantTensor) ReturnsPtr() bool                           { return true }
func (c constantTensor) OverwritesInput() int                       { return -1 }
func (c constantTensor) CallsExtern() bool                          { return false }
func (c constantTensor) DiffWRT(i int) []bool                       { return nil }
func (c constantTensor) SymDiff(Nodes, *Node, *Node) (Nodes, error) { return nil, nil }
func (c constantTensor) Do(...values.Value) (values.Value, error)   { return c.v, nil }
func (c constantTensor) String() string                             { return fmt.Sprintf("const %s", TypeOf(c.v)) }

func (c constantTensor) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "const %v:%v", c.Type(), c.v)
}

func (c constantTensor) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c constantTensor) isconstant() bool    { return true }
func (c constantTensor) Value() values.Value { return c.v }
