# MicroBooNE 200-Toy profiled CLs band

Select every production-grid point whose reference CLs lies in [0.005, 0.1].
At every selected point, generate 200 Toys under each hypothesis and profile
every Toy separately. Clopper-Pearson tail-probability intervals are propagated
to pointwise CLs limits; their CLs=0.05 crossings define two contour curves.

Results are checkpointed after every completed parameter point and the run is
resumable. This is a study and does not modify active inference.

