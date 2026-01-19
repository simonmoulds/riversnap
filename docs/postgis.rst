Working with GeoDataFrames and PostGIS
======================================

``riversnap`` supports two complementary workflows for snapping gauging sites to
river networks:

1. **In-memory snapping** using :class:`geopandas.GeoDataFrame` objects.
2. **Database-backed snapping** using **PostGIS** for scalable candidate
   generation on large river vector datasets.

Both workflows share the same high-level snapping logic (distance components,
diagnostics, and ranking). The main difference is how **candidate river reaches**
are generated.

Overview
--------

In ``riversnap``, snapping is typically performed in two stages:

1. **Candidate generation**
   Find all (or the top-:math:`k`) river reaches within a given search radius of
   each gauge location.

2. **Candidate scoring**
   Compute one or more distance components (e.g. spatial distance, log-ratio
   distance for drainage area) and combine these into an overall score to select
   the best candidate.

The package provides two candidate-generation backends:

* **GeoDataFrame backend** (filesystem / in-memory)
* **PostGIS backend** (database / scalable)

GeoDataFrame backend
--------------------

The GeoDataFrame backend is the simplest option, and requires no external
database.

Typical use cases include:

* regional river networks that fit comfortably in memory
* prototyping a snapping strategy
* interactive exploration in notebooks

Inputs
^^^^^^

You provide:

* a GeoDataFrame of **river reaches** (lines)
* a GeoDataFrame of **gauges** (points)

The snapping workflow will:

* buffer gauges by a user-defined distance threshold
* spatially join reaches intersecting each buffer
* compute the point-to-line Euclidean distance for each candidate reach

Example
^^^^^^^

.. code-block:: python

   import geopandas as gpd
   from riversnap import GRIT

   target_crs = 3857 

   pts = gpd.read_file("my_gauges.gpkg").to_crs(epsg=3857)

   ds = GRIT(backend="filesystem", segments=True, continents=['EU'])
   fs = ds.get_files(root=Path('/path/to/GRITv1/'), continents=['EU'], grit_version=1, srid=target_crs)
   ds.prepare_data(files=fs, target_crs=srid)

   # Candidate generation is handled internally by the filesystem backend
   out = ds.snap(
       pts,
       id_column="gauge_id",
       distance_threshold=1500,
       distance_specification=[...],
       return_all=False,
   )

PostGIS backend
---------------

For very large river vector datasets (global hydrographies, many gauges, or
high-resolution networks), candidate generation can become the dominant
computational cost. In these cases, PostGIS can provide a major speedup by:

* using spatial indexes (GiST) on large line datasets
* efficiently evaluating proximity queries (e.g. ``ST_DWithin``)
* returning the top-:math:`k` nearest candidate reaches per gauge using indexed
  nearest-neighbour ordering

Typical use cases include:

* global river datasets (e.g. GRIT)
* tens of thousands of gauge points
* repeated snapping experiments with different thresholds and distance plans

If you are working with global river datasets, the PostGIS backend is strongly
recommended. The initial setup cost of loading the hydrography into PostGIS is 
offset by the performance gains during candidate generation.

Requirements
^^^^^^^^^^^^

To use the PostGIS backend you need:

* a running PostgreSQL database with the PostGIS extension enabled
* a SQLAlchemy engine connected to that database

Example engine creation:

.. code-block:: python

   from sqlalchemy import create_engine

   engine = create_engine(
       "postgresql+psycopg2://postgres:postgres@localhost:5432/riversnap"
   )

Candidate generation in PostGIS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``riversnap`` uses a proximity query of the form:

* ``ST_DWithin`` to find reaches within a radius of each point
* ``ORDER BY <->`` to efficiently rank nearby candidates using the spatial index
* optionally ``LIMIT k`` to restrict the number of candidates per gauge

If ``k`` is set to ``None``, the query returns **all** candidate reaches within
the radius.

Coordinate reference systems (CRS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For performance and interpretability, the PostGIS backend is typically used with
a **projected CRS in metres** (e.g. EPSG:3857). In this case:

* ``distance_threshold`` is interpreted in **metres**
* ``ST_DWithin`` and ``ST_Distance`` operate on planar geometry

If your data are stored in geographic coordinates (EPSG:4326), you may prefer to
use PostGIS ``geography`` operations instead, but this is slower and is not the
default workflow in ``riversnap``.

Preparing a PostGIS hydrography table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common workflow is to load a hydrography product (e.g. GRIT) into PostGIS once,
then reuse it across many snapping runs.

In ``riversnap``, heavy setup operations should be performed in a dedicated
preparation step (rather than inside object construction). This keeps dataset
objects lightweight and makes the workflow easier to reason about.

A typical preparation step includes:

* creating (or validating) the hydrography table
* creating GiST indexes on geometry columns
* optional ingestion tracking and verification

Example
^^^^^^^

.. code-block:: python

   from riversnap import GRIT

   ds = GRIT(root="/path/to/GRITv1", segments=True, continents=["EU"])

   # Prepare PostGIS backend: load segments into a table and build indexes
   ds.prepare_postgis(
       engine=engine,
       table="gritv1_segments_eu_3857",
       target_crs=3857,
       if_exists="fail",
   )

   # Snap using points already stored in PostGIS
   out = ds.snap(
       pts="ohdb_nrfa",           # name of points table in PostGIS
       id_column="ohdb_id",
       distance_threshold=5000,
       distance_specification=[...],
       return_all=False,
       engine=engine,             # required for PostGIS backend
   )

Candidate output format
-----------------------

Regardless of backend, candidate generation returns a tabular object that
includes:

* the gauge identifier column
* reach attributes for candidate line features
* a computed ``distance_m`` column

When using PostGIS, it is possible for the result to contain duplicate column
names if both the points table and the lines table share column names
(e.g. ``id`` or ``geometry``). In these cases, Pandas will preserve the column
order (points first) and allow duplicate names. To avoid ambiguity when working
with the output DataFrame, we recommend that you either rename or alias columns
in your SQL queries.

Docker-based PostGIS quickstart (optional)
------------------------------------------

For local development and reproducible examples, a containerised PostGIS
instance can be started with Docker Compose. A reference ``docker-compose.yml``
file is provided in the repository (``examples/postgis/``)
and can be used as a quickstart for local testing and benchmarking.

.. note::

   The Docker Compose file is intended for development and examples. It is not
   part of the installed Python package.

Troubleshooting
---------------

Engine required for PostGIS
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an error indicating an engine is missing, ensure that:

* you are using the PostGIS backend, and
* you provided a valid SQLAlchemy engine

Example:

.. code-block:: python

   out = ds.snap(..., engine=engine)

Table existence checks
^^^^^^^^^^^^^^^^^^^^^^

A quick way to check if a table exists in the connected database is:

.. code-block:: python

   from sqlalchemy import inspect
   exists = inspect(engine).has_table("gritv1_segments", schema="public")

Or, search-path aware:

.. code-block:: python

   from sqlalchemy import text
   with engine.connect() as con:
       exists = con.execute(
           text("SELECT to_regclass(:tname) IS NOT NULL"),
           {"tname": "public.gritv1_segments"},
       ).scalar()

Indexes
^^^^^^^

For large datasets, ensure you have a GiST index on geometry columns:

.. code-block:: python 

   from sqlalchemy import text
   with engine.begin() as con:
        con.execute(
             text("CREATE INDEX IF NOT EXISTS grit_geom_gix ON gritv1_segments USING GIST (geometry);")
        )

Without this, candidate generation will be slow.

