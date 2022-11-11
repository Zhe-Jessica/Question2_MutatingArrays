I am trying to use sciml_train in Flux to train a neural network model using the data from an engineering system. The data contains many groups of measurements with different initial states. I gave these initial states to the neural network individually and calculated the loos function by summing the mean squared error of all measurements.
I want to use standardization for my training data to see if my training result gets better or not. I did data transformation before I calculated the loss function. To scale back to the original range, I saved the transformation. I got this error:
```
ERROR: Mutating arrays is not supported -- called copyto!(LinearAlgebra.Transpose{Float64, Matrix{Float64}}, ...)
This error occurs when you ask Zygote to differentiate operations that change
the elements of arrays in place (e.g. setting values with x .= ...)

Possible fixes:
- avoid mutating operations (preferred)
- or read the documentation and solutions for this error
https://fluxml.ai/Zygote.jl/latest/limitations
```
Here is my code. I demonstrate two groups of measurements here. Hope it makes my question clear.
I tried to use `{Zygote.Buffer()` for `trans`, `ynm`, `nny` in loss function, `loss_neuralode`, but it did not solve the problem. 
