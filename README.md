# polynomial_opt

## First Time Setup
```julia
import Pkg; Pkg.activate(".")
Pkg.instantiate()
```
You may encounter issues with the `TSSOS` and `lazy_nlp_qd` packages. For now I have removed `lazy_nlp_qd`. To install TSSOS, use
```julia
import Pkg; Pkg.add("https://github.com/wangjie212/TSSOS")
```