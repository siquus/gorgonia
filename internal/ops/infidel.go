package stdops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

var (
	_ graph.op = LetOp{}
)

// LetOp is not really a function. It's more of a binding statement.
// However, it's implemented as a Op so that it can be counted for register allocation and liveness
type LetOp struct{}

func (op LetOp) Arity() int                                                      { return 0 }
func (op LetOp) Type() hm.Type                                                   { return nil }
func (op LetOp) ReturnsPtr() bool                                                { return true }
func (op LetOp) OverwritesInput() int                                            { return 0 }
func (op LetOp) CallsExtern() bool                                               { return false }
func (op LetOp) InferShape(...DimSizer) (tensor.Shape, error)                    { return nil, nil }
func (op LetOp) DiffWRT(int) []bool                                              { return nil }
func (op LetOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op LetOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op LetOp) String() string                                                  { return "=" }
func (op LetOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("let")) }
func (op LetOp) Hashcode() uint32                                                { return simpleHash(op) }
func (op letOp) IsStmt() bool                                                    { return true }

// ReadOp reads a value off the input. This op ensures that a value used, and hence codegen'd out
type ReadOp struct {
	into *values.Value // no, it's not a mistake. It's a pointer to a Value (which is an interface{} type)
}

func (op readOp) Arity() int                                                      { return 0 }
func (op readOp) Type() hm.Type                                                   { return nil }
func (op readOp) ReturnsPtr() bool                                                { return true }
func (op readOp) OverwritesInput() int                                            { return 0 }
func (op readOp) CallsExtern() bool                                               { return false }
func (op readOp) InferShape(...DimSizer) (tensor.Shape, error)                    { return nil, nil }
func (op readOp) DiffWRT(int) []bool                                              { return nil }
func (op readOp) SymDiff(inputs Nodes, outputNode, gradNode *Node) (Nodes, error) { return nil, nil }
func (op readOp) Do(vals ...Value) (Value, error)                                 { return nil, nil }
func (op readOp) String() string                                                  { return "print" }
func (op readOp) WriteHash(h hash.Hash)                                           { h.Write([]byte("print")) }
func (op readOp) Hashcode() uint32                                                { return simpleHash(op) }

func (op readOp) IsStmt() bool { return true }

// DevTrans is a dummy Op, used to aid in creating the program that is run in a *tapeMachine. It is inserted not into the graph, but into a slice of sorted nodes, and will not show up in thegraph.
type DevTrans struct {
	from, to Device
	toNode   *Node
}

func (op DevTrans) Arity() int                                   { panic("not implemented") }
func (op DevTrans) Type() hm.Type                                { panic("not implemented") }
func (op DevTrans) InferShape(...DimSizer) (tensor.Shape, error) { panic("not implemented") }
func (op DevTrans) Do(...Value) (Value, error)                   { panic("not implemented") }
func (op DevTrans) ReturnsPtr() bool                             { return false }
func (op DevTrans) CallsExtern() bool                            { return true }
func (op DevTrans) OverwritesInput() int                         { return -1 }
func (op DevTrans) WriteHash(h hash.Hash)                        { fmt.Fprintf(h, "from:%vto%v", op.from, op.to) }
func (op DevTrans) Hashcode() uint32                             { return simpleHash(op) }
func (op DevTrans) String() string                               { return fmt.Sprintf("[CP %v %v]", op.from, op.to) }
func (op DevTrans) IsStmt() bool                                 { return true }

func (op DevTrans) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	return nil, nil
}
func (op DevTrans) CUDAFuncName() string { return op.String() }
