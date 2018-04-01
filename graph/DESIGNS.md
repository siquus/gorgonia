# Current Design #

The current design of a node in the graph is as a struct containing all its parts:

```go
// A Node is a node in the computation graph
type Node struct {
	// metadata of the node
	t     hm.Type // pruned types only plz
	shape tensor.Shape

	// this node is the result of applying the op to the children
	op       Op
	children Nodes // shortcut, instead of having to go through the graph

	// For nicely grouping stuff in graphviz.
	// TODO: Should this be in *Node?
	name  string
	group string

	g *ExprGraph // this node belongs in this graph

	// value bondage
	// inputs are bound to values directly
	boundTo Value
	dataOn  Device // where is the data on

	// to track derivations
	derivOf Nodes
	deriv   *Node

	// for hashing nodes
	id   int64 // id is the ID at which the node is added to the graph
	hash uint32

	hashed        bool
	inferredShape bool // is shape inferred?
	unchanged     bool // has this node been modified
	isStmt        bool // is this a statement node
	ofInterest    bool // is this node of particular interest? (for debugging)
}
```

For completeness, this is the definition of an `ExprGraph`:

```go
type ExprGraph struct {
	name string

	all Nodes

	byHash map[uint32]*Node
	evac   map[uint32]Nodes
	to     map[*Node]Nodes

	leaves    Nodes
	constants Nodes
	roots     Nodes
	counter   uint
}
```

Let's walk through each field to explain its necessity and see how we can refactor this:

| Field | Type | Purpose | Necessity |
|-------|------|---------|-----------|
| `t`   | `hm.Type` | Stores the type of the Node. Note this is the expected type of the correctly executed `Op`. | Used as input to the Node's parent's type inference phase|
| `shape` | `tensor.Shape` | Stores the shape of the Node. This is the expected shape of the correctly executed `Op`. | Used as an input to the Node's parent's shape inference phase |
| `op`  | `Op` | The op that was applied to the children. May be nil to indicate input. | Necessary |
| `children` | `Nodes` | The inputs that generated this node | Necessary |
| `name, group` | string | Drawing graphs | ??? |
| `g` | `ExprGraph` | Stores the graph that this Node is in | Necessary |
| `boundTo` | `Value` | The "executed" value. | Necessary for `LispMachine` | 
| `dataOn` | `Device` | The device that the data will sit in | Necessary for execution state |
| `derivOf` | Nodes | Tracks which derivation of this node is | Necessary |
| `deriv` | `*Node` | Tracks the derivation of this node | Necessary |
| `id` | `int64` | ID of the node in the graph | Necessary |
| `hash` | `uint32` | The hash code. This allows only unique nodes to be added. Lazily calculated, once | Necessary |
| `hashed` | `bool` | Is this node hashed? | ??? |
| `unchanged` |  `bool` | Has this node been changed?  | ??? |
| `isStmt` | `bool` | Is this a statement node? | Necessary for performance. This used to be a method, but as it turns out it was too much to keep computing in a tight loop | 
| `ofInterest` | `bool` | For debugging neural networks | Necessary, but probably not here |

As you may note, there are multiple concepts all mashed into one data structure:

* Expression graph - a graph of operations and their types (lambda calculus, if you will)
* Execution state tracking - `boundTo` tracks the values of the Node. It's primarily for user friendliness. `isStmt` is a more or less static execution state tracking.
* Execution pragmas - `dataOn` tells the executing machine where to store the data
* Differentiation graph - which nodes are derivatives of which nodes
* Drawing state 

# Refactor #

Now there are multiple axes of refactoring. Having run a LOT of neural networks using Gorgonia, I've noticed one of the issues is pointer chasing. The GC chases pointers, because there are pointers EVERYWHERE. This somewhat degrades performance. In particular, I noticed it may be caused by the copious amounts of `Nodes` and `*Node` everywhere

Here are some solutions:

## Simplified Node ##

Let there be another data type called `Data`, which stores the things necessary, in a "global" `ExprGraph` data structure. Then `Node` can simply be:

```
type Node struct {
	ID int64
	Graph *ExprGraph
}

// A Data is a node in the computation graph
type Data struct {
	// metadata of the node
	t     hm.Type // pruned types only plz
	shape tensor.Shape

	// this node is the result of applying the op to the children
	op       Op
	children Nodes // shortcut, instead of having to go through the graph

	// For nicely grouping stuff in graphviz.
	// TODO: Should this be in *Node?
	name  string
	group string

	g *ExprGraph // this node belongs in this graph

	// value bondage
	// inputs are bound to values directly
	boundTo Value
	dataOn  Device // where is the data on

	// to track derivations
	derivOf Nodes
	deriv   *Node

	// for hashing nodes
	id   int64 // id is the ID at which the node is added to the graph
	hash uint32

	hashed        bool
	inferredShape bool // is shape inferred?
	unchanged     bool // has this node been modified
	isStmt        bool // is this a statement node
	ofInterest    bool // is this node of particular interest? (for debugging)
}


type ExprGraph struct {
	// elided

	all []Data // note, not pointers. this forms an "arena" of sorts

	byHash map[uint32]int64
	evac   map[uint32][]int64
	to     map[int64]int64

	leaves    []int64
	constants []int64
	roots     []int64
	counter   uint
}
```

Instead of passing around pointers, one can just pass around a two word value. The reason for storing the `*ExprGraph` is for accesbility to the data of the node.

This will slow things down for things like back propagation, where analysis of a node and its children are needed. But that's an acceptable slowdown, if the execution of the instructions in the VM speeds up due to less pointer chasing and GC pauses. 

The problem with this simplified node structure is that it is incompatible with the notion of a `Tensor` interface to be defined in Gorgonia proper. The `Node` of an `ExprGraph` *will* implement a `Tensor` interface - either exactly the same interface as `tensor.Tensor` or a reduced variant. 

When wrapping things in an interface, Go will automatically allocate the struct on the heap and use the pointer to the object. Now we're back at the problem of pointer chasing. 

## Two Node Structures ## 

So we arrive at the conclusion we need TWO types of Node structure. `Node` is just a dumb data structure with access to certain things. It doesn't implement the `Tensor` interface.  `*Data` on the other hand implements a `Tensor` interface. Then we write a method to get a `*Data` from a Node. 

This is clearly however, a HUGE usability problem. Having two kinds of node representation is terrible. 


# AoS vs SoA #

The `ExprGraph` is currently a AoS-ish structure. Is there a point in destructuring the current implementation of `*Node` ?

What I mean is making the data structure something like this (with a simplified `Data`):

```go
type Data struct {
	// metadata of the node
	t     hm.Type // pruned types only plz
	shape tensor.Shape

	// this node is the result of applying the op to the children
	op       Op

	// unique name
	name  string

	// for hashing nodes
	id   int64 // id is the ID at which the node is added to the graph
	hash uint32

	hashed        bool
	inferredShape bool // is shape inferred?
	unchanged     bool // has this node been modified
	isStmt        bool // is this a statement node
	ofInterest    bool // is this node of particular interest? (for debugging)
}

type ExprGraph struct {
	// elided

	all      []Data 
	children map[int64][]int64
	valuesOf map[int64]Value
	devices  map[int64]Device
	derivs   map[int64]int64
	derivOf  map[int64][]int64

	byHash map[uint32]int64
	evac   map[uint32][]int64
	to     map[int64]int64

	leaves    []int64
	constants []int64
	roots     []int64
	counter   uint
}
```


# TO DO : INVESTIGATION #

The next step in the investigation is to enumerate the phases at which each field is used. Then try to separate them out.