# PRISM module

PRISM separates the expensive, task-specific execution from the search
feedback used by an AAD method.

1. Evaluate distinct initial algorithms on all training instances.
2. Build the instance-by-algorithm response matrix and select `k` rows by
   greedy leave-one-algorithm-out affine-ridge reconstruction RMSE.
3. During formal search, execute every new algorithm only on those `k` rows.
4. Use `feedback="mean"` for EoH, or `feedback="cpi_augmented"` for EoH-S.

The augmented policy always contains the `k` raw instance axes and fills the
remaining directions (300 by default) with a fixed uniform simplex design.
The module assumes larger fitness is better.  For gap minimization, pass
`fitness = -gap`.

```python
prism = PRISM(128, k=5, target_algorithms=10, feedback="cpi_augmented")
for algorithm_id, full_fitness in initialization_measurements:
    prism.observe_initial(algorithm_id, full_fitness)

selected = prism.selected_indices
search_fitness = prism.transform_selected(run_on_instances(program, selected))
```

`PRISM` is framework-independent: the caller owns program execution and can
attach the scalar/vector returned by `transform_selected` to EoH, EoH-S, or
another search controller.
