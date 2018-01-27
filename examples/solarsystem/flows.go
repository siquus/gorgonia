package main

import (
	G "github.com/chewxy/gorgonia"
)

// If a system is in state s_1 at time t_1 and this changes to s_2 at a later time
// t_2, and this change is given by
// F_{t_1, t_2}(s_1) = s_2
// We call F the evolution operator; it maps a state at t_1 to what the state would be after
// time t_2 - t_1 as elapsed.
//
// If F_{t_1, t_2} only depends on t_2 - t_1 the operation is time independent.
// F_{t}(s) := F_{t}(s} is then called the flow
// else, F_{t_1, t_2} is callled the time-dependent flow
//
// If the system is non-reversible, that is, defined only for t_2 >= t_1, we speak of
// a semi-flow.
//
// See e.g. p.60 in Abraham, Ralph and Marsden, Jerrold E. (1987) Foundations of Mechanics
// available (legally) for free https://authors.library.caltech.edu/25029/

type EvolutionOperator interface {
	Step(G.Nodes) error
}
