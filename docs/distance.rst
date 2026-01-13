Distance components and diagnostics
===================================

Overview 
--------

Snapping in ``riversnap`` is formulated as a distance-based matching problem.
For each gauging site, multiple candidate locations on a river network
are evaluated using a set of *distance components*, which are combined to form
a composite distance used for ranking candidates.

In addition to these ranking distances, each distance component can compute a 
set of *diagnostic metrics* that express mismatches on interpretable scales
(e.g. metres, percentage difference, or percentage points). This separation
allows users to define their own quality-control (QC) criteria without relying
on the composite distance itself, which can be difficult to interpret. 

Distance
--------

A distance component compares a *candidate* value with a *reference* value
associated with a gauging site (or, possibly, with a pre-computed similarity
score). Each component distance is computed using a formulation appropriate for
the type of variable being compared. 

Distance functions are specified by the user as a key, with each key
corresponding to a predefined distance function. Valid keys are as follows:

* :py:func:`euclid_scaled_dist <riversnap.distances.d_spatial_distance_scaled>` -
  Scaled Euclidean distance between the gauge location and a candidate river reach.

* :py:func:`log_ratio <riversnap.distances.d_log_ratio>` -
  Log-ratio distance for strictly positive, scale-dependent attributes that span
  orders of magnitude, such as drainage area or mean annual discharge.

* :py:func:`abs_diff <riversnap.distances.d_abs_diff>` -
  Absolute difference for bounded indices such as baseflow index.

* :py:func:`sim_01 <riversnap.distances.d_one_minus_similarity01>` -
  One minus a similarity score bounded on [0, 1], such as those computed by
  matching algorithms.

Each component produces a column named ``d_<name>``, where ``<name>`` is the
logical name of the component. 

Diagnostics
-----------

Diagnostics are additional quantities computed alongside each distance component.
They are not used directly in candidate ranking, but are intended to support
quality control using thresholds that are meaningful to users. Diagnostics provide 
a complementary view of candidate quality that: 

* is interpretable without knowledge of distance weights,
* remains stable across different snapping configurations,
* allows users to define dataset- and application-specific QC schemes.

By separating optimisation (distance minimisation) from evaluation (diagnostics),
``riversnap`` provides a flexible and transparent framework for snapping river
gauges to river hydrographies.

Diagnostics are implemented using a *context-based* approach.
Each diagnostic function receives a context object containing all relevant
information for a given component, including:

* the candidate values,
* the reference values (if applicable),
* the computed component distance.

Diagnostic functions may use any subset of this information. This design avoids 
complex class hierarchies while keeping diagnostic logic explicit, testable, and 
extensible.

Like the distance functions, diagnostics are specified by the user as keys, with 
each key corresponding to a predefined diagnostic function. Currently supported 
diagnostic keys are as follows:

* ``m`` - Euclidean distance in metres (e.g. gauge-to-reach offset).
* ``pct`` - Percentage mismatch (derived from log-ratio distances as
  ``(exp(d) - 1) * 100``).
* ``factor`` - Multiplicative factor mismatch (derived from log-ratio distances
  as ``exp(d)``).
* ``pp`` - Difference in percentage points for bounded indices (e.g. variables
  in [0, 1]).
* ``err`` - Absolute difference in native units.
* ``sim`` - Candidate similarity score (copies the candidate values to the
  diagnostic column; useful when candidate values are already interpretable and
  bounded in [0, 1]).

Diagnostics are always derived from the same inputs as the distance component
(candidate values, reference values, and/or the component distance), ensuring
consistency between ranking and QC.

Naming convention
-----------------

For a distance component named ``<name>``, the following naming conventions 
are used:

* ``d_<name>`` — component distance (used for ranking candidates)
* ``diag_<diagnostic-key>_<name>`` — component diagnostic (used for QC)

This consistent naming scheme allows users to easily construct QC rules using
standard Pandas operations.

Example: distance component specification
-----------------------------------------

Distance components are specified as follows:

.. code-block:: python

   specs = [
       DistanceSpec(
           name="gauge_dist",
           cand_col="distance_m",
           dist_fn="spatial",
           kwargs={"scale_m": 500.0},
           diagnostics=("m",),
       ),
       DistanceSpec(
           name="drainage_area",
           cand_col="cand_area_km2",
           ref_col="ref_area_km2",
           dist_fn="scale",
           diagnostics=("pct", "factor"),
       ),
   ]

If a diagnostic cannot be computed for a given component (e.g. because a
reference value is unavailable), the resulting column will contain missing
values. 

Example: quality control
------------------------

A typical workflow is:

1. Rank candidates using the composite distance.
2. Apply QC criteria using diagnostic columns.
3. Accept or flag snapped locations accordingly.

For example:

.. code-block:: python

   good = (
       (df["diag_m_gauge_dist"] <= 250) &
       (df["diag_pct_drainage_area"] <= 25)
   )

This approach avoids imposing hard-coded QC rules within ``riversnap`` while
ensuring that QC thresholds are defined on interpretable scales.