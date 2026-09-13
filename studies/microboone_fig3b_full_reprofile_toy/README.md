# MicroBooNE Fig.3a/Fig.3b full-grid per-Toy reprofile

This research runner uses the active BNB+NuMI prediction, registered
`psi(E,L,flavour)` input, covariance and Fig.3b constrained-profile definition.
It does not replace the active fixed-hypothesis Toy implementation.

Fig.3a uses 61x61 logarithmic points over `0.01--100 eV2` and
`sin2(2theta_mue)=1e-4--1`. Fig.3b uses 74x61 logarithmic points over
`0.1--40 eV2` and `sin2(2theta_ee)=0.01--1`. At every plotted point it generates
the requested number of Toys under each of the 3nu and tested 4nu hypotheses.
Every Toy repeats the same fixed-coordinate profile over `sin2(theta24)` and
both physical `sin2(theta14)` branches.  Summary and raw Toy statistics are
appended immediately, so completed points survive an interrupted worker.

The runner accepts `--figures fig3a`, `fig3b`, or `both`; the production default
is both figures and 5000 Toys per generating hypothesis. Toy precision must
still be assessed from the actual tail counts.

For a downloaded server batch, `render_shards.py --batch-directory <batch>`
validates and merges the selected point shards, then writes the merged CSV,
heatmap and contour-only figures, and binomial counting diagnostics.
