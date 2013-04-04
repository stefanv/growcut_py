#!/usr/bin/env zsh
numpy_dir=/usr/local/lib/python2.7/site-packages/numpy
python setup.py build_ext --include-dirs=$numpy_dir/core/include
cmd="PYTHONPATH=.:./build/lib.macosx-10.8-x86_64-2.7 python tests/test_growcut_py.py"
echo running $cmd
eval $cmd
