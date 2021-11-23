:orphan:

.. _ug-visualization:

=================================
Visualization within TensorBoard
=================================

One may log various graphs and data collected in different phases of the model quantization and display them within the Tensorboard UI.
To use it, all you have to do is to set a logger path. Setting a path is done by calling :ref:`set_log_folder<ug-set_logger_path>`.

.. code-block:: python

   import model_compression_toolkit as mct
   mct.set_log_folder('/logger/dir/path')

Then, by calling :ref:`keras_post_training_quantization<ug-keras_post_training_quantization>`, a TensorBoard writer will log graphs of the model at different stages.
To visualize them, TensorBoard should be launched with:

tensorboard --logdir=/logger/dir/path

|


The graphs representing the model can be seen under the Graphs tab:

.. image:: ../../images/tbwriter/tbwriter_graphs.png
  :scale: 40%

|


To observe the model at different stages of the quantization process, change the 'Run':

.. image:: ../../images/tbwriter/tbwriter_stages.png
  :scale: 50%

|


To display the required memory (in bytes) of the graph at different stages, change the 'Tag' from 'Default' to 'Resources'

.. image:: ../../images/tbwriter/tbwriter_resources.png
  :scale: 60%

|

By clicking a node, its statistics will show up:

.. image:: ../../images/tbwriter/tbwriter_resources_node.png
  :scale: 60%


|


During the quantization process, statistics are gathered at some layers' output: histograms, min/max per channel and mean per channel.
These statistics can be viewed under 'Histograms' (histograms) or 'Scalars' (min/max/mean per channel) for each layer that statistics were gathered in its output:

.. image:: ../../images/tbwriter/tbwriter_histograms.png
  :scale: 50%

.. image:: ../../images/tbwriter/tbwriter_scalars.png
  :scale: 50%


|


=================================
Cosine Similarity Comparison
=================================

Computing a cosine-similarity is a way to quantify the similarity between two vectors.
Mathematically, the cosine similarity is the division between the dot product of the vectors and the product of the euclidean norms of each vector.
Thus, we can use it to measure the two models similarity, by measuring the cosine similarity
of tensors along the networks in different pairs of points in the networks, where we would expect them to
output similar tensors.

.. image:: ../../images/cs_compare.png

|

There are many ways to measure similarity between two models (or tensors). To name a few: MSE, MAE, KL-Divergence, etc.
As for now, mct uses cosine-similarity to compare the tensors along the models and display its changes within the
TensorBoard UI.

|

Several plots comparing the cosine similarity between the original float model and the
final quantized model at different points can be viewed under 'Images'.
More specifically, 20 samples from the provided representative dataset generator, are inserted to both of
the models, and the cosine similarity at the output of different layers are computed and displayed.

.. image:: ../../images/tbwriter/tbwriter_cosinesimilarity.png
  :scale: 50%
