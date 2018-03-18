# graph

`package graph` implements the expression graph that Gorgonia uses.

# Developer Notes #

Developers familiar with lambda calculus or have built lisps will note that there are similarities between `graph` types and lambda calculus. Here are the equivalencies:

| λ-calculus | Gorgonia |
|:----------:|:--------:|
| Var  | values.Value |
| Lambda | graph.Ops |
| Apply | graph.Node |

The part which diverges from traditional implementation of lambda calculus based system is that at definition, the tree is already partially evaluated. This is why instead of proper lambda abstractions, we arrive at a delta-ish lambda calculus, where all lambdas are externally defined (as are primitives).

The reason for doing this is mainly performance - performant on-the-fly graph reduction is surprisingly harder than expected to be implemented by one person. A naive version would take too long. So instead of a graph reduction, a graph walking algorithm is used, and the graph is reduced when the graph is defined.

See the ALTERNATIVE DESIGNS document.