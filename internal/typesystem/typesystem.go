package typesystem

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Infer infers the type of the expression
func Infer(expr interface{}) (retVal hm.Type, err error) {
	switch e := expr.(type) {
	case *Node:
		// stop the recursive inference early - if the node already has a type, return it
		if typ := e.Type(); typ != nil {
			return typ, nil
		}
		// if e.IsInput() || e.IsConstant() {
		// 	// Var (and Let const)
		// 	return e.Type(), nil
		// }

		return inferNodeType(e.Op(), e.children...)
	case hm.Typer:
		return e.Type(), nil
	default:
		err = errors.Errorf(nyiTypeFail, "inferType", expr)
		return
	}
}

// Instead of using hm's Infer function, since all the nodes are pretty much hm.Apply, we write our own.
func inferNodeType(op Typer, children ...*Node) (retVal hm.Type, err error) {
	fnType := op.Type()
	if fnt, ok := fnType.(*hm.FunctionType); ok {
		defer hm.ReturnFnType(fnt)
	}

	argTypes := hm.BorrowTypes(len(children) + 1)
	defer hm.ReturnTypes(argTypes)
	for i, child := range children {
		if argTypes[i], err = Infer(child); err != nil {
			return nil, errors.Wrapf(err, "Failed to infer type of %v", child)
		}
	}

	b := hm.TypeVariable('b')
	argTypes[len(argTypes)-1] = b

	fn := hm.NewFnType(argTypes...)
	defer hm.ReturnFnType(fn)

	// var t0 hm.Type
	var sub hm.Subs
	if sub, err = hm.Unify(fn, fnType); err != nil {
		return nil, errors.Wrapf(err, "Unable to unify while inferring type of %v", op)
	}

	var ok bool
	if retVal, ok = sub.Get(b); !ok {
		return nil, errors.Errorf("Expected a replacement for %v", b)
	}

	// return pruneReturn(t0.(*hm.FunctionType).ReturnType()), nil
	return retVal, nil
}

// IsScalarType returns true if the type is a scalar type
func IsScalarType(t hm.Type) bool {
	switch tt := t.(type) {
	case tensor.Dtype:
		return true
	case TensorType:
		if tt.Dims == 0 {
			return true
		}
		return false
	case hm.TypeVariable:
		panic("Type Variable is a type that is not yet known.")
	default:
		panic("Unhandled type")
	}
}

// DtypeOf returns the data type of the type.
func DtypeOf(t hm.Type) (retVal tensor.Dtype, err error) {
	switch p := t.(type) {
	case tensor.Dtype:
		return p, nil
	case TensorType:
		return DtypeOf(p.Of)
	case hm.TypeVariable:
		return nil, errors.Errorf("instance %v does not have a dtype", p)
	default:
		return nil, errors.Errorf(nyiFail, "dtypeOf", p)
	}
}
