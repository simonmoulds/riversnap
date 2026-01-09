Distance components and diagnostics
===================================

Snapping in ``riversnap`` is formulated as a distance-based matching problem.
For each gauging site, multiple candidate locations on a river network
are evaluated using a set of *distance components*, which are combined to form
a composite distance used for ranking candidates.

In addition to these ranking distances, ``riversnap`` can compute a set of
*diagnostic metrics* that express mismatches on interpretable scales
(e.g. metres, percentage difference, or percentage points). This separation
allows users to define their own quality-control (QC) criteria without relying
on the composite distance itself.

Overview
--------

Each snapping component has two distinct roles:

* **Distance**: a dimensionless quantity used internally to rank candidates.
* **Diagnostics**: human-interpretable metrics derived from the same inputs,
  intended for QC, filtering, and inspection.

Distances are optimised; diagnostics are reported.

This distinction is central to the design of ``riversnap``.

Distance components
-------------------

A distance component compares a *candidate* value with a *reference* value
associated with a gauging site (or, in some cases, with a pre-computed similarity
score). Each component distance is computed using a formulation appropriate for
the type of variable being compared.

Common examples include:

* Scaled spatial distance between the gauge location and a candidate river reach.
* Log-ratio distance for strictly positive, scale-dependent attributes such as
  drainage area or mean annual discharge.
* Absolute difference for bounded indices such as baseflow index.

Each component produces a column named ``d_<name>``, where ``<name>`` is the
logical name of the component. Distance components are user-defined.

Diagnostics
-----------

Diagnostics are additional quantities computed alongside each distance component.
They are not used directly in candidate ranking, but are intended to support
quality control using thresholds that are meaningful to users.

Examples of diagnostic quantities include:

* Spatial offset in metres.
* Percentage mismatch in drainage area.
* Multiplicative factor difference between candidate and reference values.
* Difference in percentage points for bounded indices.

Diagnostics are always derived from the same inputs as the distance component
(candidate values, reference values, and/or the component distance), ensuring
consistency between ranking and QC.

Context-based diagnostics
-------------------------

In ``riversnap``, diagnostics are implemented using a *context-based* approach.
Each diagnostic function receives a context object containing all relevant
information for a given component:

* the candidate values,
* the reference values (if applicable),
* the computed component distance.

Diagnostic functions may use any subset of this information. For example:

* Percentage-point diagnostics require candidate and reference values.
* Percentage and factor diagnostics for log-ratio distances require only the
  component distance.
* Spatial diagnostics may simply return the raw candidate distance in metres.

This design avoids complex class hierarchies while keeping diagnostic logic
explicit, testable, and extensible.

Configuring diagnostics
-----------------------

Diagnostics are specified per distance component using short string keys.
For example:

.. code-block:: python

   specs = [
       DistanceSpec(
           name="space",
           cand_col="gauge_to_reach_m",
           dist_fn=d_spatial_scaled,
           kwargs={"scale_m": 500.0},
           diagnostics=("m",),
       ),
       DistanceSpec(
           name="area",
           cand_col="cand_area_km2",
           ref_col="ref_area_km2",
           dist_fn=d_log_ratio,
           diagnostics=("pct", "factor"),
       ),
       DistanceSpec(
           name="bfi",
           cand_col="cand_bfi",
           ref_col="ref_bfi",
           dist_fn=d_abs_diff,
           diagnostics=("pp",),
       ),
   ]

Each diagnostic key corresponds to a predefined diagnostic computation.
If a diagnostic cannot be computed for a given component (e.g. because a
reference value is unavailable), the resulting column will contain missing
values.

Naming convention
-----------------

For a component named ``<name>``, the following naming conventions are used:

* ``d_<name>`` — component distance (used for ranking)
* ``m_<name>`` — spatial distance in metres
* ``pct_<name>`` — percentage mismatch
* ``factor_<name>`` — multiplicative factor mismatch
* ``pp_<name>`` — percentage-point difference
* ``err_<name>`` — absolute difference in native units

This consistent naming scheme allows users to easily construct QC rules using
standard pandas operations.

Example: quality control
------------------------

A typical workflow is:

1. Rank candidates using the composite distance.
2. Apply QC criteria using diagnostic columns.
3. Accept or flag snapped locations accordingly.

For example:

.. code-block:: python

   good = (
       (df["m_space"] <= 250) &
       (df["pct_area"] <= 25) &
       (df["pp_bfi"] <= 5)
   )

This approach avoids imposing hard-coded QC rules within ``riversnap`` while
ensuring that QC thresholds are defined on interpretable scales.

Design rationale
----------------

The composite distance produced by ``riversnap`` is intentionally unitless and
configuration-dependent. As a result, it is not well suited to absolute QC
thresholds. Diagnostics provide a complementary view of candidate quality that:

* is interpretable without knowledge of distance weights,
* remains stable across different snapping configurations,
* allows users to define dataset- and application-specific QC schemes.

By separating optimisation (distance minimisation) from evaluation (diagnostics),
``riversnap`` provides a flexible and transparent framework for snapping river
gauges to digital hydrographies.
