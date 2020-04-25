# Defining objectives/constraints in optimization packages

Would love to get your thoughts on the below - send me an email at 96imranahmed@gmail.com :smile:

## Problem 
I have a pet-peeve with optimization packages in that they force very complex code-interfaces that often vary with the optimization method being used.

For example, take a look at `scipy.optimize` docs [here](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html). For the same optimization objective (e.g., `Rosenbrock`), there are multiple possible function signatures  often with additional arguments (e.g., explicitly adding `hessian` functions for Newton Conjucate-Gradient methods). The world becomes even more complex with constrained optimization, where you need to define even more package-specific code to represent basic constraints.

Case in point, see the snippet below to define a simple less than equal to constraint in PuLP:
```python
# Less than equal constraints
constraints = {j : opt_model.addConstraint(
plp.LpConstraint(
             e=m(a[i,j] * x_vars[i,j] for i in set_I),
             sense=plp.plp.LpConstraintLE,
             rhs=b[j],
             name="constraint_{0}".format(j)))
       for j in set_J}
```

^ Do you really need this much complexity to do "<=" in constrained optimization?


I see this as a challenge for two reasons: 
1. **Limited interoperability between optimization methods** - The number of optimization packages / approaches grows every day. Writing code that is very method/package specific prevents the quick 'plug-and-play' of different optimization approaches, limiting iteration and building unnecessary reliance on certain optimization packages.
2. **Limited flexibility for new optimization use-cases** - Optimization techniques are now being used in situations that many packages weren't originally designed for (e.g., optimizing hyperparameters of a machine learning model, optimizing thresholds for 'fair' machine learning). Because of the strict interfaces that older packages impose, existing code-bases require substantial rework before they can be passed to an optimization package. 

In summary, optimization packages broadly aren't 'fit for the future'!

## Potential solution

It's all very well talking about problems, but let's try to propose a solution. I want to preface this by saying that I have not fully fleshed out the below (I'm sure I'm missing edge-cases etc.), but hopefully it should encourage a healthy discussion about what the best way forward should be. 

**Key design decisions:**
1. A standardized interface for optimization packages. I leave package-specific code to the packages, while surfacing an easy-to-use interface to map onto existing code-bases.
2. Represent the `objective` as a class. Each class has a function `f(X)` as well as set of constraints. In a world where `f(x)` if differentiable, you can also consider creating additional requirements for `f_hessian(X)` which can be used by different optimization packages where available. Finally `f(X)` can be *any* function (e.g., an ML pipeline for use in hyperparameter search)

```python
class Objective(object):
    f = None
    constraints = []
    
    def __init__(self,
                 f, 
                 constraints,
                 n_dimensions_in,
                 n_dimensions_out = 1,
                 **kwargs):
```

3. Represent the `constraint` as a class. Each constraint represents a function `g(X)` that returns a `True` or `False` depending on whether that constraint has been satisfied. For more flexibility, it is left up to the implementor to decide whether to write the constraint more efficiently (e.g., via vectorization). Finally, there is support for specific `kwargs` to be passed to that function (if needed). One possible extension would be to also include `f(x)` in the constraint function (e.g., `g(X) -> g(X, f(X))`) in order to support a wider range of constraints.   

With this proposed approach, packages can interop with existing code-bases in a way that would allow developers 'plug-and-play' optimization options across a range of algorithms. The one draw-back is that function and constraint evaluation is now forced to be in python, but I suspect the efficient use of C-accelerated python libraries will mean evaluations can still be made efficiently. 

**What do you think about this approach? Do you share similar frustrations with these optimization packages?**

## Example

The following example illustrates how simple it would be to define an optimization task in you code-base (example for the 2D Eggholder function): 

```python

# Sub-class Objective class and merge it with your existing code-base
class Eggholder(Objective):
    def __init__(self): 

        def f(X, **kwargs): 
            return -(X[1] + 47)*np.sin(np.sqrt(np.abs(X[0]/2 + (X[1]+47)))) + \
            -X[0]*np.sin(np.sqrt(np.abs(X[0] - (X[1]+ 47))))
             

        def bound_constraint(X, **kwargs):
            return np.all(np.logical_or(X < -512, X > 512))
        
        constraints = [
            Constraint(bound_constraint, **{})
        ]

        super().__init__(f, constraints, 
                            n_dimensions_in = 2, 
                            n_dimensions_out = 1, 
                            **{})

# Define your optimization objective
obj = Eggholder()

# Initializate your favourite optimization package with a start value and optimize
my_fav_optimizer = OptimizePackage(obj, seed_X = X)
optimized_X = my_fav_optimizer.optimize(params = my_fav_optimizer_params)
```

Wouldn't life be simple if optimizers could work like this?