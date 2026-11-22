GRAIL: Generating Humanoid Loco-Manipulation from World Models
==============================================================

.. raw:: html

   <span id="grail-documentation"></span>

Start with a video. Recover the human and object motion, transfer it to a Unitree G1, and train a policy to follow it in simulation. The two examples include the video, camera and metric object mesh.

.. tip::

   **First time here?** Check :doc:`GPU, disk and time requirements <requirements>`, then :doc:`start the GPU container <installation>` and :doc:`download the weights <downloads>`.

Tutorials
---------

.. raw:: html

   <span id="choose-a-tutorial"></span>

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Stair Climbing
      :link: tutorials/stairs
      :link-type: doc
      :img-top: _static/examples/stairs.jpg

      Watch the fitted feet land on each tread, then train a simulated G1 to climb the same stairs.

   .. grid-item-card:: Kettlebell Lifting
      :link: tutorials/kettlebell
      :link-type: doc
      :img-top: _static/examples/kettlebell.jpg

      Inspect the human grasp, then watch the robot fingers close around the handle. Continue to simulation to measure how far the policy follows the lift.

For a new case, see :doc:`the generation route and real-footage limits <tutorials/own-video>`. You can finish with the G1 motion and preview, or continue to simulation. See the :doc:`glossary <configuration>` and :doc:`outputs reference <outputs>`.

Pipeline at a glance
--------------------

.. raw:: html

   <p>
     <a href="https://research.nvidia.com/labs/dair/grail/">Project</a> ·
     <a href="https://arxiv.org/abs/2606.05160">Paper</a> ·
     <a href="https://github.com/NVlabs/GRAIL">Source code</a>
   </p>

.. image:: ../../assets/videos/teaser.gif
   :alt: GRAIL teaser
   :align: center
   :width: 100%

.. image:: ../../assets/images/hoigen.jpeg
   :alt: HOI generation pipeline
   :align: center
   :width: 100%

.. image:: ../../assets/images/motion_tracking.jpeg
   :alt: Task-general loco-manipulation tracking pipeline
   :align: center
   :width: 100%

The bundled examples start at the video and scene. If you are generating a new
scene, first :doc:`create the mesh <gen_3d_assets>` and :doc:`generate the video <gen_2dhoi>`.

1. :doc:`Reconstruct <recon_4dhoi>` the human and object, then inspect the fit against the source video.
2. :doc:`Retarget <retargeting>` the reviewed motion to the Unitree G1, preserving the finger joints for the kettlebell grasp.
3. :doc:`Train and evaluate <tracking>` a terrain or manipulation policy with the matching scene and motion. Inspect its rollout metrics before exporting a trajectory.

.. raw:: html

   <span id="bundled-dependencies"></span><span id="id1"></span>

See :doc:`the component guide <resources>` for bundled dependencies and :doc:`installation` for submodules and downloads.

Sim-to-Real Deployment
----------------------

Representative real Unitree G1 deployment results are included for object
pick-up and stair-climbing, alongside rendered egocentric views used for
visual-policy training.

Rendered Egocentric Views
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../assets/videos/deployment_egocentric_views.gif
   :alt: Rendered egocentric views for policy training
   :align: center
   :width: 100%

Real-Robot Deployment
~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <span id="real-robot-clips"></span>

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Pick-up
     - Stair-Climbing
   * - .. image:: ../../assets/videos/deployment_pickup.gif
          :alt: Pick-up deployment
          :width: 100%
     - .. image:: ../../assets/videos/deployment_stairs.gif
          :alt: Stair-climbing deployment
          :width: 100%

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   requirements
   installation
   downloads
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/stairs
   tutorials/kettlebell
   tutorials/own-video
   results

.. toctree::
   :maxdepth: 2
   :caption: Pipelines

   gen_3d_assets
   gen_2dhoi
   recon_4dhoi
   retargeting
   tracking

.. toctree::
   :maxdepth: 2
   :caption: Data Exploration

   data_export
   visualization
   web_visualizer

.. toctree::
   :maxdepth: 1
   :caption: Reference and Support

   outputs
   configuration
   troubleshooting
   manual_installation
   resources

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
