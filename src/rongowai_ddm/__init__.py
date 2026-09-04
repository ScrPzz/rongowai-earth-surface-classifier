"""Land/water classification from Rongowai GNSS-R Delay-Doppler Maps.

The package is organised along the pipeline:

- :mod:`rongowai_ddm.netcdf` reads one L1 flight file into arrays;
- :mod:`rongowai_ddm.data` indexes flights, filters DDMs, labels them, samples
  them, splits them by flight and writes the sample table;
- :mod:`rongowai_ddm.features` turns DDMs into feature groups;
- :mod:`rongowai_ddm.models` wraps the model families and calibration;
- :mod:`rongowai_ddm.evaluation` computes metrics and figures.
"""

__version__ = "0.1.0"
