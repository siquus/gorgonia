The internal directory is used to hide away implementation details like:

* ops
* type system

The main reason for implementing thse in `internal` is because if they were implemented in `gorgonia.org/ops` or `gorgonia.org/types`, the API would be exposed. That's poor design.

Furthermore it allows for untangling of possible import cycles