Quick Start 
=========== 

Prerequisites
-------------
Before you can start using riversnap you need a Python environment with the 
required dependencies installed. The recommended way to manage your Python 
environment is to use `conda <https://docs.conda.io/en/latest/>`_, because it 
makes it easy to install geospatial packages with external dependencies. 

You can create a new conda environment with the required dependencies by using 
our provided environment file. From the root of the repository run:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate riversnap

riversnap should be installed within this environment.

Installation
------------

Currently riversnap can only be installed from source. Start by cloning or 
downloading the repository from GitHub: 

.. code-block:: bash

   git clone https://github.com/simonmoulds/riversnap.git

This will create a directory called `riversnap`. Change into this directory and 
install a local, editable version of riversnap using pip:

.. code-block:: bash
   cd riversnap
   pip install -e .

Data
----

To use riversnap you will need a river hydrography dataset and a set of river 
gauge sites. We suggest starting with the Global River Topology (GRIT) 
hydrography, and the EStreams gauge dataset. 

GRIT can be downloaded from `Zenodo <https://doi.org/10.5281/zenodo.7629907>`_. 
For this example we will retrieve `GRITv1.0_segments_EU_EPSG4326.gpkg.zip <https://zenodo.org/records/17435232/files/GRITv1.0_segments_EU_EPSG4326.gpkg.zip?download=1>`_ 
which contains a GeoPackage file with river segments for Europe in EPSG:4326 
coordinate reference system.

The EStreams gauge dataset can also be downloaded from `Zenodo <https://doi.org/10.5281/zenodo.10733141>`_. 
We require the gauging stations shapefile, which is contained with the `shapefiles.zip <https://zenodo.org/records/17598150/files/shapefiles.zip?download=1>`_
archive.

Download these two archives and extract their contents to a suitable location. Now 
you might have something like this: 

.. code-block:: text
   EStreams/
      shapefiles.zip
      shapefiles/ 
         estreams_gauging_stations.cpg
         estreams_gauging_stations.dbf
         estreams_gauging_stations.prj
         estreams_gauging_stations.shp
         estreams_gauging_stations.shx
   GRIT/ 
      GRITv1.0_segments_EU_EPSG4326.gpkg.zip
      GRITv1.0_segments_EU_EPSG4326.gpkg

Snapping
--------

Now you are ready to start using riversnap to snap the EStreams gauging stations to the GRIT river hydrography. 


