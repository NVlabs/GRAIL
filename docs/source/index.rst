GRAIL: Generating Humanoid Loco-Manipulation from World Models
==============================================================

.. raw:: html

   <p>
     <a href="https://nvlabs.github.io/GRAIL/"><img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page"></a>
     <a href="https://arxiv.org/pdf/2606.05160"><img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square" alt="Paper"></a>
     <a href="https://github.com/NVlabs/GRAIL/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-NVIDIA-76B900?style=flat-square" alt="License"></a>
     <a href="https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2"><img src="https://img.shields.io/badge/IsaacLab-2.3.2-76B900?style=flat-square" alt="IsaacLab"></a>
   </p>

.. image:: ../../assets/videos/teaser.gif
   :alt: GRAIL teaser
   :align: center
   :width: 100%

GRAIL — *Generating Humanoid Loco-Manipulation from World Models* —
is a fully digital data-generation pipeline for robot-compatible humanoid
loco-manipulation. It composes 3D assets, simulator-ready scenes,
robot-proportioned characters, and video foundation model priors to synthesize
interactions in known metric 3D configurations, then reconstructs 4D
human-object trajectories, retargets them to the Unitree G1, and trains
task-general manipulation and locomotion trackers. The resulting egocentric
visual policies transfer from GRAIL-generated data to real-world object
pick-up and stair-climbing.

.. tip::

   **New here?** Start with :doc:`installation`, then
   :doc:`quick_start` for a 5-minute smoke run.

Pipeline at a glance
--------------------

.. image:: ../../assets/images/hoigen.jpeg
   :alt: HOI generation pipeline
   :align: center
   :width: 100%

.. image:: ../../assets/images/motion_tracking.jpeg
   :alt: Motion tracking pipeline
   :align: center
   :width: 100%

.. code-block:: text

   Text / asset prompt
           ↓
   python -m grail.pipelines.gen_terrain / python -m grail.pipelines.gen_3d_assets
           →  3D object meshes
           ↓
   python -m grail.pipelines.gen_2dhoi
           →  Blender + Kling AI / MiniMax-H3 HOI videos
           ↓
   python -m grail.pipelines.recon_4dhoi
           →  GEM-X native SOMA + MoGe-3 depth + hybrid object trajectories
           ↓
   imports/soma-retargeter + GMR MJCF   →  Unitree G1+DEx3 joint trajectories
           ↓
   imports/SONIC                        →  Task-general tracking policy training

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

Sim-to-Real Deployment
~~~~~~~~~~~~~~~~~~~~~~

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

Bundled Dependencies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Path
     - Upstream
     - Purpose
   * - ``imports/SONIC``
     - GRAIL-vendored SONIC release tree
     - RL training, inference, deploy stack
   * - ``imports/GMR``
     - `YanjieZe/GMR <https://github.com/YanjieZe/GMR>`_
     - Legacy SMPL-X → robot retargeting engine
   * - ``imports/GEM-X``, ``imports/soma-retargeter``
     - `NVlabs/GEM-X <https://github.com/NVlabs/GEM-X>`_ / `NVIDIA SOMA Retargeter <https://github.com/NVIDIA/soma-retargeter>`_
     - Native SOMA prediction and source-bound SOMA → Unitree G1+DEx3 retargeting
   * - ``imports/MoSE3``, ``imports/FoundationPose``
     - `MoSE3 <https://github.com/mose3-tracker/MoSE3>`_ / `NVlabs/FoundationPose <https://github.com/NVlabs/FoundationPose>`_
     - Hybrid object 6-DOF tracking
   * - ``imports/MoGe``, ``imports/Hunyuan3D-2.1``
     - Microsoft, Tencent
     - MoGe-3 monocular depth + 3D asset generation
   * - ``imports/Sana``
     - `NVlabs/Sana <https://github.com/NVlabs/Sana>`_
     - MiniMax-H3 video generation through Sol-Engine

Initialize with:

.. code-block:: bash

   git submodule update --init --recursive

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
