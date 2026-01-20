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
hydrography, and the EStreams gauge dataset. We provide a `tutorial <tutorials/hydrography-data.nblink>`_ 
with download instructions for these datasets. 


Snapping
--------

Now you are ready to start using riversnap to snap the EStreams gauging stations to the GRIT river hydrography. 


