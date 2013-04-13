#!/usr/bin/env bash
numpy_dir=/usr/local/lib/python2.7/site-packages/numpy
python setup.py build_ext --inplace --include-dirs=$numpy_dir/core/include
cmd="PYTHONPATH=. python tests/test_growcut_py.py"
echo running $cmd
eval $cmd
