Create a clustering visualization app with Bokeh and scikit-learn

Set up
=======

To install the required dependencies to run the demo, install 
the packages in the ``environment.yml`` file:

    conda env create

Then activate the environment:

    source activate bokeh-learn


Running
=======

Bokeh Server
------------

To view the apps directly from a bokeh server, simply run:

    bokeh serve cluster1.py


Then navigate to the following URL in a browser:

    http://localhost:5006/cluster1


Demo Content
============

- ``cluster_plot.ipynb``: A simple notebook demonstrating how to create a 
     plot in Bokeh to visualize the output of a scikit-learn clustering 
     algorithm 

- ``cluster1.py``: A clustering visualization app with one selection widget
    for the clustering algorithm.

- ``cluster2.py``: A clustering visualization app with two selection widgets,
    one for the clustering algorithm and one for the dataset, and two slider
    widgets, one for the number of samples and one for the number of clusters.
