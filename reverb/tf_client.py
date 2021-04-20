# Lint as: python3
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

"""TFClient provides tf-ops for interacting with Reverb."""

from typing import Any, Optional, Sequence

from absl import logging
from reverb import dataset
from reverb import replay_sample
import tensorflow.compat.v1 as tf
import tree

from reverb.cc.ops import gen_client_ops

# TODO(b/153616873): Remove this alias once users have been refactored.
ReplayDataset = dataset.ReplayDataset


class TFClient:
  """Client class for calling Reverb replay servers from a TensorFlow graph."""

  def __init__(self,
               server_address: str,
               shared_name: Optional[str] = None,
               name: str = 'reverb'):
    """Creates the client TensorFlow handle.

    Args:
      server_address: Address of the server.
      shared_name: (Optional) If non-empty, this client will be shared under the
        given name across multiple sessions.
      name: Optional name for the Client operations.
    """
    self._name = name
    self._server_address = server_address
    self._handle = gen_client_ops.reverb_client(
        server_address=server_address, shared_name=shared_name, name=name)

  def sample(self,
             table: str,
             data_dtypes,
             name: Optional[str] = None) -> replay_sample.ReplaySample:
    """Samples an item from the replay.

    This only allows sampling items with a data field.

    Args:
      table: Probability table to sample from.
      data_dtypes: Dtypes of the data output. Can be nested.
      name: Optional name for the Client operations.

    Returns:
      A ReplaySample with data nested according to data_dtypes. See ReplaySample
      for more details.
    """
    with tf.name_scope(name, f'{self._name}_sample', ['sample']) as scope:
      key, probability, table_size, priority, data = gen_client_ops.reverb_client_sample(
          self._handle, table, tree.flatten(data_dtypes), name=scope)
      return replay_sample.ReplaySample(
          replay_sample.SampleInfo(
              key=key,
              probability=probability,
              table_size=table_size,
              priority=priority), tree.unflatten_as(data_dtypes, data))

  def insert(self,
             data: Sequence[tf.Tensor],
             tables: tf.Tensor,
             priorities: tf.Tensor,
             name: Optional[str] = None):
    """Inserts a trajectory into one or more tables.

    The content of `tables` and `priorities` are zipped to create the
    prioritized items. That is, an item with priority `priorities[i]` is
    inserted into `tables[i]`.

    Args:
      data: Tensors to insert as the trajectory.
      tables: Rank 1 tensor with the names of the tables to create prioritized
        items in.
      priorities: Rank 1 tensor with priorities of the new items.
      name: Optional name for the client operation.

    Returns:
      A tf-op for performing the insert.

    Raises:
      ValueError: If tables is not a string tensor of rank 1.
      ValueError: If priorities is not a float64 tensor of rank 1.
      ValueError: If priorities and tables does not have the same shape.
    """
    if tables.dtype != tf.string or tables.shape.rank != 1:
      raise ValueError('tables must be a string tensor of rank 1')
    if priorities.dtype != tf.float64 or priorities.shape.rank != 1:
      raise ValueError('priorities must be a float64 tensor of rank 1')
    if not tables.shape.is_compatible_with(priorities.shape):
      raise ValueError('priorities and tables must have the same shape')

    with tf.name_scope(name, f'{self._name}_insert', ['insert']) as scope:
      return gen_client_ops.reverb_client_insert(
          self._handle, data, tables, priorities, name=scope)

  def update_priorities(self,
                        table: str,
                        keys: tf.Tensor,
                        priorities: tf.Tensor,
                        keys_to_delete: Optional[tf.Tensor] = None,
                        name: str = None):
    """Creates op for updating priorities of existing items in the replay.

    Not found elements for `keys` are silently ignored.

    Args:
      table: Probability table to update.
      keys: Keys of the items to update. Must be same length as `priorities`.
      priorities: New priorities for `keys`. Must be same length as `keys`.
      keys_to_delete: Keys of the items to delete
      name: Optional name for the operation.

    Returns:
      A tf-op for performing the update.
    """

    if keys_to_delete is None:
      keys_to_delete = tf.constant([], dtype=tf.uint64)

    with tf.name_scope(name, f'{self._name}_update_priorities',
                       ['update_priorities']) as scope:
      return gen_client_ops.reverb_client_update_priorities(
          self._handle, table, keys, priorities, keys_to_delete, name=scope)

  def dataset(self,
              table: str,
              dtypes: Sequence[Any],
              shapes: Sequence[Any],
              capacity: int = 100,
              num_workers_per_iterator: int = -1,
              max_samples_per_stream: int = -1,
              sequence_length: Optional[int] = None,
              emit_timesteps: bool = True,
              rate_limiter_timeout_ms: int = -1) -> dataset.ReplayDataset:
    """DEPRECATED, please use dataset.ReplayDataset instead.

    A tf.data.Dataset which samples timesteps from the ReverbService.

    Note: Uses of Python lists are converted into tuples as nest used by the
    tf.data API doesn't have good support for lists.

    See dataset.ReplayDataset for detailed documentation.

    Args:
      table: Probability table to sample from.
      dtypes: Dtypes of the data output. Can be nested.
      shapes: Shapes of the data output. Can be nested. When `emit_timesteps` is
        True this is the shape of a single timestep in the sampled items; when
        it is False shapes must include `sequence_length`.
      capacity: (Defaults to 100) Maximum number of samples requested by the
        workers with each request. Higher values give higher throughput but too
        big values can result in skewed sampling distributions as large number
        of samples are fetched from single snapshot of the replay (followed by a
        period of lower activity as the samples are consumed). A good rule of
        thumb is to set this value to 2-3x times the batch size used.
      num_workers_per_iterator: (Defaults to -1, i.e auto selected) The number
        of worker threads to create per dataset iterator. When the selected
        table uses a FIFO sampler (i.e a queue) then exactly 1 worker must be
        used to avoid races causing invalid ordering of items. For all other
        samplers, this value should be roughly equal to the number of threads
        available on the CPU.
      max_samples_per_stream: (Defaults to -1, i.e auto selected) The maximum
        number of samples to fetch from a stream before a new call is made.
        Keeping this number low ensures that the data is fetched uniformly from
        all server.
      sequence_length: (Defaults to None, i.e unknown) The number of timesteps
        that each sample consists of. If set then the length of samples received
        from the server will be validated against this number.
      emit_timesteps: (Defaults to True) If set, timesteps instead of full
        sequences are retturned from the dataset. Returning sequences instead of
        timesteps can be more efficient as the memcopies caused by the splitting
        and batching of tensor can be avoided. Note that if set to False then
        then all `shapes` must have dim[0] equal to `sequence_length`.
      rate_limiter_timeout_ms: (Defaults to -1: infinite).  Timeout (in
        milliseconds) to wait on the rate limiter when sampling from the table.
        If `rate_limiter_timeout_ms >= 0`, this is the timeout passed to
        `Table::Sample` describing how long to wait for the rate limiter to
        allow sampling. The first time that a request times out (across any of
        the workers), the Dataset iterator is closed and the sequence is
        considered finished.

    Returns:
      A ReplayDataset with the above specification.
    """
    logging.warning(
        'TFClient.dataset is DEPRECATED! Please use ReplayDataset (see '
        './dataset.py) instead.')
    return dataset.ReplayDataset(
        server_address=self._server_address,
        table=table,
        dtypes=dtypes,
        shapes=shapes,
        max_in_flight_samples_per_worker=capacity,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples_per_stream=max_samples_per_stream,
        sequence_length=sequence_length,
        emit_timesteps=emit_timesteps,
        rate_limiter_timeout_ms=rate_limiter_timeout_ms)
