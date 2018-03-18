package dual

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/tensor"
)

func typeCheckTypeOf(v Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return types.NewTensorType(dim, dt)
	case Scalar:
		return t.Dtype()
	case Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

func tensorInfo(t tensor.Tensor) (dt tensor.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}
