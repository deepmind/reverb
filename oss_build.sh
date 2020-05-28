#!/bin/bash
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Designed to work with ./docker/release.dockerfile to build reverb for multiple
# Python versions. It can work locally but is not tested for that use case.
#
# Example usage after building release docker:
#   docker run --rm -it -v ${REVERB_DIR}:/tmp/reverb tensorflow:reverb_release \
#   bash oss_build.sh --python 3.8

# Exit if any process returns non-zero status.
set -e
set -o pipefail

# Flags
PYTHON_VERSION=3.6 # Options 3.6 (default), 3.7, 3.8
CLEAN=false # Set to true to run bazel clean.
OUTPUT_DIR=/tmp/reverb/dist/
PYTHON_TESTS=true

ABI=cp36

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--python [3.6(default)|3.7|3.8]"
  echo "--clean  [true to run bazel clean]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  case $key in
      --python)
      PYTHON_VERSION="$2" # Python version to build against.
      shift
      ;;
      --clean)
      CLEAN="$2" # `true` to run bazel clean. False otherwise.
      shift
      ;;
      --python_tests)
      PYTHON_TESTS="$2"
      shift
      ;;
      --output_dir)
      OUTPUT_DIR="$2"
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      ;;
  esac
  shift # past argument or value
done

# Cleans the environment.
if [ "$CLEAN" = "true" ]; then
  bazel clean
fi

if [ "$PYTHON_VERSION" = "3.6" ]; then
  export PYTHON_BIN_PATH=/usr/bin/python3.6 && export PYTHON_LIB_PATH=/usr/local/lib/python3.6/dist-packages
elif [ "$PYTHON_VERSION" = "3.7" ]; then
  export PYTHON_BIN_PATH=/usr/local/bin/python3.7 && export PYTHON_LIB_PATH=/usr/local/lib/python3.7/dist-packages
  ABI=cp37
elif [ "$PYTHON_VERSION" = "3.8" ]; then
  export PYTHON_BIN_PATH=/usr/bin/python3.8 && export PYTHON_LIB_PATH=/usr/local/lib/python3.8/dist-packages
  ABI=cp38
else
  echo "Error unknown --python. Only [3.6|3.7|3.8]"
  exit
fi

# Configures Bazel environment for selected Python version.
$PYTHON_BIN_PATH configure.py

# Runs bazel tests for cc.
# Only run cc tests because `bazel test` seems to ignore bazelrc and only uses
# /usr/bin/python3. A solution is to swap symbolic links for each version of
# python to be tested. This works well in docker but would make a mess of
# someone's system unexpectedly. We are executing the python tests after
# installing the final package making this approach satisfactory.
# TODO(b/157223742): Execute Python tests as well.
bazel test -c opt --copt=-mavx --config=manylinux2010 --test_output=errors //reverb/cc/...

# Builds Reverb and creates the wheel package.
bazel build -c opt --copt=-mavx --config=manylinux2010 reverb/pip_package:build_pip_package
./bazel-bin/reverb/pip_package/build_pip_package --dst $OUTPUT_DIR

# Installs pip package.
$PYTHON_BIN_PATH -mpip install ${OUTPUT_DIR}*${ABI}*.whl

if [ "$PYTHON_TESTS" = "true" ]; then
  echo "Run Python tests..."
  set +e

  bash run_python_tests.sh |& tee ./unittest_log.txt
  UNIT_TEST_ERROR_CODE=$?
  set -e
  if [[ $UNIT_TEST_ERROR_CODE != 0 ]]; then
    echo -e "\n\n\n===========Error Summary============"
    grep -E 'ERROR:|FAIL:' ./unittest_log.txt
    exit $UNIT_TEST_ERROR_CODE
  else
    echo "Python tests successful!!!"
  fi
fi