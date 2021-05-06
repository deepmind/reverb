# python3
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

"""Build and installs dm-reverb."""
import argparse
import codecs
import datetime
import fnmatch
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

import reverb_version

# Defaults if doing a release build.
TENSORFLOW_VERSION = 'tensorflow>=2.3.0'


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = super().finalize_options()
    # We need to set this manually because we are not using setuptools to
    # compile the shared libraries we are distributing.
    self.install_lib = self.install_platlib
    return ret


class SetupToolsHelper(object):
  """Helper to execute `setuptools.setup()`."""

  def __init__(self, release=False, tf_version_override=None):
    """Initialize ReleaseBuilder class.

    Args:
      release: True to do a release build. False for a nightly build.
      tf_version_override: Set to override the tf_version dependency.
    """
    self.release = release
    self.tf_version_override = tf_version_override

  def _get_version(self):
    """Returns the version and project name to associate with the build."""
    if self.release:
      project_name = 'dm-reverb'
      version = reverb_version.__rel_version__
    else:
      project_name = 'dm-reverb-nightly'
      version = reverb_version.__dev_version__
      version += datetime.datetime.now().strftime('%Y%m%d')

    return version, project_name

  def _get_required_packages(self):
    """Returns list of required packages."""
    required_packages = [
        'dataclasses; python_version < "3.7.0"',  # Back-port for Python 3.6.
        'dm-tree',
        'portpicker',
    ]
    return required_packages

  def _get_tensorflow_packages(self):
    """Returns list of required packages if using reverb."""
    tf_packages = []
    if self.release:
      tf_version = TENSORFLOW_VERSION
    else:
      tf_version = 'tf-nightly'

    # Overrides required versions if tf_version_override is set.
    if self.tf_version_override:
      tf_version = self.tf_version_override

    tf_packages.append(tf_version)
    return tf_packages

  def run_setup(self):
    # Builds the long description from the README.
    root_path = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(root_path, 'README.md'), encoding='utf-8') as f:
      long_description = f.read()

    version, project_name = self._get_version()

    so_lib_paths = [
        i for i in os.listdir('.')
        if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
    ]

    matches = []
    for path in so_lib_paths:
      matches.extend(['../' + x for x in find_files('*', path) if '.py' not in x])

    setup(
        name=project_name,
        version=version,
        description=('Reverb is an efficient and easy-to-use data storage and '
                     'transport system designed for machine learning research.'),
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='DeepMind',
        author_email='DeepMind <no-reply@google.com>',
        url='https://github.com/deepmind/reverb',
        license='Apache 2.0',
        packages=find_packages(),
        headers=list(find_files('*.proto', 'reverb')),
        include_package_data=True,
        package_data={
            'reverb': matches,
        },
        install_requires=self._get_required_packages(),
        extras_require={
            'tensorflow': self._get_tensorflow_packages(),
        },
        distclass=BinaryDistribution,
        cmdclass={
            'install': InstallCommand,
        },
        python_requires='>=3',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='tensorflow deepmind reinforcement learning machine replay jax',
    )


if __name__ == '__main__':
  # Hide argparse help so `setuptools.setup` help prints. This pattern is an
  # improvement over using `sys.argv` and then `sys.argv.remove`, which also
  # did not provide help about custom arguments.
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '--release',
      action='store_true',
      help='Pass as true to do a release build')
  parser.add_argument(
      '--tf-version',
      type=str,
      default=None,
      help='Overrides TF version required when Reverb is installed, e.g.'
      'tensorflow>=2.3.0')
  FLAGS, unparsed = parser.parse_known_args()
  # Go forward with only non-custom flags.
  sys.argv.clear()
  # Downstream `setuptools.setup` expects args to start at the second element.
  unparsed.insert(0, 'foo')
  sys.argv.extend(unparsed)
  setup_tools_helper = SetupToolsHelper(release=FLAGS.release,
                                        tf_version_override=FLAGS.tf_version)
  setup_tools_helper.run_setup()
