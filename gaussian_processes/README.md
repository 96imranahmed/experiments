# Gaussian Process implementation in Torch

A simple `Gaussian Process` implementation in [pytorch](https://pytorch.org/) to test the speed-up when using vectorized code to calculate kernels (both for a full recalculation of kernal matrix, and calculating the kernel increment with new data). Also, I wanted to learn why everyone keeps talking about torch!

GP plotted to fit `sin(x)` below:

![Alt text](./test.png?raw=true)

## Kernel calculation speed-up  

Significant speed-up with vectorized code with substantially higher speed-up for full kernel recalcuation as opposed to kernel increment methods:

```
---------------------------------------------
Timing test (fit method)...
Average pairwise compute time: 1.1958807500000002, Average vectorised compute time: 0.0036964700000000406
---------------------------------------------
Timing test (fit_increment method)...
Average pairwise compute time: 0.02540175999999992, Average vectorised compute time: 0.0027035149999999676
```

## So-what? 

It looks like this torch vectorization is *really* efficient (and, moreover, torch is *really* easy to use). However, if you need to use GPs in your pipelines, it's probably a good idea to just use [GPyTorch](https://gpytorch.ai/) going forwards.
