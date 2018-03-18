package stdops

import "hash/fnv"

func simpleHash(op hashWriter) uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}
