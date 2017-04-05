// +build cuda

package gorgonia

import (
	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

// UseCudaFor is an option for *tapeMachine only. At the moment users should pass in strings of the op name ("add", "sub"...)
// Do not pass in the types (for example, don't pass in "add64")
func UseCudaFor(ops ...string) VMOpt {
	f := func(m VM) {
		// switch v := m.(type) {
		// case *tapeMachine:
		// 	if v.c == nil {
		// 		// v.init()
		// 		v.f = make(map[string][]cu.Function)
		// 	}

		// 	if len(ops) == 0 {
		// 		v.loadDummyStdLib()
		// 		return
		// 	}

		// 	for _, op := range ops {
		// 		op64 := op + "64"
		// 		op32 := op + "32"

		// 		cudaLogf("Trying to load %q and %q. m.c = %v", op64, op32, v.c)

		// 		if _, ok := cudaStdLib[op64]; ok {
		// 			// if err := v.LoadCUDAFunc(op64, data); err != nil {
		// 			// 	log.Printf("Unable to load %q: %v", op64, err)
		// 			// }
		// 			v.loadDummyFunc(op64)
		// 		}

		// 		if _, ok := cudaStdLib[op32]; ok {
		// 			// if err := v.LoadCUDAFunc(op32, data); err != nil {
		// 			// 	log.Printf("Unable to load %q: %v", op32, err)
		// 			// }
		// 			v.loadDummyFunc(op32)
		// 		}
		// 	}
		// }
	}
	return f
}

func finalizeTapeMachine(m *tapeMachine) {
	cudaLogf("Finalizing tape machine %p", m)
	for i, c := range m.c {
		cu.SetCurrent(c.Context)
		for _, v := range m.m {
			mod := v[i]
			cu.Unload(mod)
		}
		cu.DestroyContext(&c.Context)
	}
	m.Cleanup()
	m.initFail() // not really a failure. Just call to detroy all the contexts and shit
}

func (m *tapeMachine) init() {
	var initCUDA bool
	cudaLogf("instructions %v", len(m.p.instructions))
	for _, instr := range m.p.instructions {
		if eo, ok := instr.(*execOp); ok {
			if _, ok := eo.op.(CUDADoer); ok {
				initCUDA = true
				break
			}
		}
	}

	// don't bother initializing contexts if no instructions were CUDA based
	if !initCUDA {
		cudaLogf("No CUDA ops")
		return
	}
	// functions to be loaded
	cudaLogf("%v", m.f)
	funcs := make([]string, 0, len(m.ExternMetadata.f))
	for fn := range m.f {
		funcs = append(funcs, fn)
	}

	m.ExternMetadata.init(m.p.gpumem)
	m.loadStdLib()

	// cudaLogf("funcs %v", funcs)
	// for _, f := range funcs {
	// 	m.LoadCUDAFunc(f, cudaStdLib[f])
	// }
	cudaLogf("m.c = %v", m.c)
	cudaLogf("m.f = %v", m.f)
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the machine.
//
// The convention is to have one function per module, sharing the same name.
func (m *tapeMachine) LoadCUDAFunc(name, data string) (err error) {
	if len(m.c) == 0 {
		return nil
	}

	mods := make([]cu.Module, len(m.c))
	fns := make([]cu.Function, len(m.c))
	for i, c := range m.c {
		if err = cu.SetCurrent(c.Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current context when loading module %q at context %d", name, i)
			return
		}

		var mod cu.Module
		if mod, err = cu.LoadData(data); err != nil {
			err = errors.Wrapf(err, "Failed to load module %q data for %dth context %x", name, i, c)
			return
		}

		var fn cu.Function
		if fn, err = mod.Function(name); err != nil {
			err = errors.Wrapf(err, "Unable to get function %q in %dth context %x", name, i, c)
			return
		}
		mods[i] = mod
		fns[i] = fn
	}

	// set the first to current
	if len(m.c) > 0 {
		if err = cu.SetCurrent(m.c[0].Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current")
			return
		}
	}

	m.m[name] = mods
	m.f[name] = fns
	cudaLogf("Loaded %q", name)
	return nil
}

func (m *tapeMachine) loadDummyFunc(name string) {
	m.f[name] = nil
}

// loads the standardlib
func (m *tapeMachine) loadStdLib() {
	if cudaStdLib == nil {
		return
	}

	for name, data := range cudaStdLib {
		if err := m.LoadCUDAFunc(name, data); err != nil {
			cudaLogf("Unable to load %q.: %v", name, err)
		}
	}
}

func (m *tapeMachine) loadDummyStdLib() {
	if cudaStdLib == nil {
		return
	}
	for name := range cudaStdLib {
		m.loadDummyFunc(name)
	}
}

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	enterLoggingContext()
	defer leaveLoggingContext()

	m.watchedLogf("Inputs:")
	m.enterLoggingContext()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.getValue(reg)
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLoggingContext()

	toDev := instr.writeTo.device
	var v Value
	switch op := instr.op.(type) {
	case CUDADoer:
		prealloc := m.getValue(instr.writeTo)
		if v, err = op.CUDADo(m, toDev, prealloc, inputs...); err != nil {
			return errors.Wrapf(err, "Happened while attempting to use CUDA to execute %v. Node is %x. Register was %v", instr, instr.id, instr.writeTo.id)
		}
	case CLDoer:
	default:
		switch {
		case instr.preAllocated:
			if pd, ok := instr.op.(UsePreallocDoer); ok {
				p := m.cpumem[instr.writeTo.id]
				if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
					return errors.Wrapf(err, "Happened while attempting to execute %v. Node is %x. Register was: %v ", instr, instr.id, instr.writeTo.id)
				}
			} else {
				// TODO: maybe warn?
				if v, err = instr.op.Do(inputs...); err != nil {
					return errors.Wrap(err, opDoFail)
				}
			}
		case instr.useUnsafe:
			if ud, ok := instr.op.(UnsafeDoer); ok {
				if v, err = ud.UnsafeDo(inputs...); err != nil {
					return errors.Wrap(err, "Failed to carry UnsafeDo()")
				}
			} else {
				// TODO: warn?
				if v, err = instr.op.Do(inputs...); err != nil {
					return errors.Wrap(err, opDoFail)
				}
			}
		default:
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}

	}
	m.watchedLogf("Result:")
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()
	// TODO: type and shape checks

	// Write
	m.writeValue(instr.writeTo, v)
	node := m.p.g.Node(instr.id).(*Node)

	if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
		m.Signal()
		<-m.Sync()
		if err = node.bindCopy(v); err != nil {
			return errors.Wrapf(err, "TraceExec failed to bind copy")
		}
	} else {
		node.bind(v)
	}

	// this is a gradient node then, we should also bind the value to the node's dualValue
	if m.bindDV() && node.derivOf != nil {
		for _, src := range node.derivOf {
			if len(m.bindNodesDV) > 0 && !m.bindNodesDV.Contains(src) {
				continue
			}

			if src.boundTo != nil {
				dv := dvUnit(src.boundTo)
				cudaLogf("dv.d 0x%x v 0x%x | writeTo: %v", dv.d.Uintptr(), v.Uintptr(), instr.writeTo)
				dev := instr.writeTo.device
				switch dev {
				case CPU:
					add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))
					if d, err := add.UnsafeDo(dv.d, v); err == nil {
						dv.SetDeriv(d)
						src.bind(dv)
					} else {
						return err
					}
				default:
					// the CPU method is correct. This method is correct for MOST cases, but will not be correct under some other circumstances
					ctx := m.Contexts()[int(dev)]
					ctx.MemcpyDtoH(dv.d.Pointer(), cu.DevicePtr(v.Uintptr()), instr.size)

				}

				// switch cd := op.(type) {
				// case CUDADoer:
				// 	cudaLogf("CUDADOING CD")
				// 	if d, err := cd.CUDADo(m, 0, dv.d, dv.d, v); err == nil {
				// 		dv.SetDeriv(d)
				// 		src.bind(dv)
				// 	} else {
				// 		return err
				// 	}
				// case UnsafeDoer:
				// 	if d, err := cd.UnsafeDo(dv.d, v); err == nil {
				// 		dv.SetDeriv(d)
				// 		src.bind(dv)
				// 	} else {
				// 		return err
				// 	}

				// }

			}
		}

	}

	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()

	return nil
}

func (instr deviceTransport) exec(m *tapeMachine) (err error) {
	from := m.getValue(instr.from)
	to := m.getValue(instr.to)

	var ctx *cu.BatchedContext
	switch {
	case instr.from.device == CPU && instr.to.device != CPU:
		memsize := int64(from.MemSize())
		ctx = m.Contexts()[int(instr.to.device)]
		ctx.MemcpyHtoD(cu.DevicePtr(to.Uintptr()), from.Pointer(), memsize)
	case instr.from.device != CPU && instr.to.device == CPU:
		dt := from.Dtype()
		memsize := calcMemSize(dt, from.Shape())
		ctx = m.Contexts()[int(instr.from.device)]
		ctx.MemcpyDtoH(to.Pointer(), cu.DevicePtr(from.Uintptr()), memsize)

		// when copying from device to host, it's assumed that the host will want to immediately use
		// so signal the DoWork
		m.Signal()
		<-m.Sync()
	}

	return nil
}
