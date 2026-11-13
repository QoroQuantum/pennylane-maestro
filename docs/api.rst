API reference
=============

.. module:: pennylane_maestro

Most code needs only the device name, ``qp.device("maestro.qubit", ...)``, which
creates a :class:`MaestroQubitDevice`. Its keyword arguments are the options below;
the guide's :doc:`guide/options` page explains them in more detail.

Device
------

.. autoclass:: MaestroQubitDevice
   :members: incremental_evolve, seed, backend

Functions
---------

.. autofunction:: incremental_evolve
