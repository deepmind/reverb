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

"""Tests for tf_client."""

from concurrent import futures

import numpy as np
from reverb import client as reverb_client
from reverb import item_selectors
from reverb import rate_limiters
from reverb import server
from reverb import tf_client
import tensorflow.compat.v1 as tf

def make_server():
  return server.Server(
      tables=[
          server.Table(
              'dist',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1)),
          server.Table(
              'dist2',
              sampler=item_selectors.Prioritized(priority_exponent=1),
              remover=item_selectors.Fifo(),
              max_size=1000000,
              rate_limiter=rate_limiters.MinSize(1)),
      ],
      port=None,
  )


class SampleOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def test_sets_meta_data_fields(self):
    input_data = [np.ones((81, 81), dtype=np.float64)]
    self._client.insert(input_data, {'dist': 1})
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      sample = session.run(client.sample('dist', [tf.float64]))
      np.testing.assert_equal(input_data, sample.data)
      self.assertNotEqual(sample.info.key, 0)
      self.assertEqual(sample.info.probability, 1)
      self.assertEqual(sample.info.table_size, 1)
      self.assertEqual(sample.info.priority, 1)

  def test_dtype_mismatch_result_in_error_raised(self):
    data = [np.zeros((81, 81))]
    self._client.insert(data, {'dist': 1})
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with self.assertRaises(tf.errors.InternalError):
        session.run(client.sample('dist', [tf.float32]))

  def test_forwards_server_error(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with self.assertRaises(tf.errors.NotFoundError):
        session.run(client.sample('invalid', [tf.float64]))

  def test_retries_until_success_or_fatal_error(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      with futures.ThreadPoolExecutor(max_workers=1) as executor:
        sample = executor.submit(session.run,
                                 client.sample('dist', [tf.float64]))
        input_data = [np.zeros((81, 81))]
        self._client.insert(input_data, {'dist': 1})
        np.testing.assert_equal(input_data, sample.result().data)


class UpdatePrioritiesOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def test_shape_result_in_error_raised(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      update_op = client.update_priorities(
          tf.constant('dist'), tf.constant([1, 2], dtype=tf.uint64),
          tf.constant([1], dtype=tf.float64))
      with self.assertRaises(tf.errors.InvalidArgumentError):
        session.run(update_op)

  def test_priority_update_is_applied(self):
    # Start with uniform distribution
    for i in range(4):
      self._client.insert([np.array([i], dtype=np.uint32)], {'dist': 1})

    # Until we have recieved all 4 items.
    items = {}
    while len(items) < 4:
      item = next(self._client.sample('dist'))[0]
      items[item.info.key] = item.info.probability
      self.assertEqual(item.info.probability, 0.25)

    # Update the priority of one of the items.
    update_key = next(iter(items.keys()))
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      update_op = client.update_priorities(
          table=tf.constant('dist'),
          keys=tf.constant([update_key], dtype=tf.uint64),
          priorities=tf.constant([3], dtype=tf.float64))
      self.assertIsNone(session.run(update_op))

    # The updated item now has priority 3 and the other 3 items have priority 1
    # each. The probability of sampling the new item should thus be 50%. We
    # sample until the updated item is seen and check that the probability (and
    # thus the priority) has been updated.
    for _ in range(1000):
      item = next(self._client.sample('dist'))[0]
      if item.info.key == update_key:
        self.assertEqual(item.info.probability, 0.5)
        break
    else:
      self.fail('Updated item was not found')


  def test_delete_key_is_applied(self):
    # Start with 4 items
    for i in range(4):
      self._client.insert([np.array([i], dtype=np.uint32)], {'dist': 1})

    # Until we have recieved all 4 items.
    items = {}
    while len(items) < 4:
      item = next(self._client.sample('dist'))[0]
      items[item.info.key] = item.info.probability

    # remove 2 items
    items_to_keep = [*items.keys()][:2]
    items_to_remove = [*items.keys()][2:]
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      for key in items_to_remove:
        update_op = client.update_priorities(
          table=tf.constant('dist'),
          keys=tf.constant([], dtype=tf.uint64),
          priorities=tf.constant([], dtype=tf.float64),
          keys_to_delete=tf.constant([key], dtype=tf.uint64))
        self.assertIsNone(session.run(update_op))

    # 2 remaining items must persist
    final_items = {}
    for _ in range(1000):
      item = next(self._client.sample('dist'))[0]
      self.assertTrue(item.info.key in items_to_keep)
      final_items[item.info.key] = item.info.probability
    self.assertEqual(len(final_items), 2)


class InsertOpTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._server = make_server()
    cls._client = reverb_client.Client(f'localhost:{cls._server.port}')

  def tearDown(self):
    super().tearDown()
    self._client.reset('dist')
    self._client.reset('dist2')

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls._server.stop()

  def setUp(self):
    super().setUp()
    self.data = [tf.constant([1, 2, 3], dtype=tf.int8)]

  def test_checks_that_table_has_rank_1(self):
    client = tf_client.TFClient(self._client.server_address)
    priorities = tf.constant([1.0], dtype=tf.float64)

    # Works for rank 1.
    client.insert(self.data, tf.constant(['dist']), priorities)

    # Does not work for rank > 1.
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant([['dist']]), priorities)

    # Does not work for rank < 1.
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant('dist'), priorities)

  def test_checks_dtype_of_table_argument(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant([1]),
                    tf.constant([1.0], dtype=tf.float64))

  def test_checks_that_priorities_argument_has_rank_1(self):
    client = tf_client.TFClient(self._client.server_address)
    data = [tf.constant([1, 2])]
    tables = tf.constant(['dist'])

    # Works for rank 1.
    client.insert(data, tables, tf.constant([1.0], dtype=tf.float64))

    # Does not work for rank > 1.
    with self.assertRaises(ValueError):
      client.insert(data, tables, tf.constant([[1.0]], dtype=tf.float64))

    # Does not work for rank < 1.
    with self.assertRaises(ValueError):
      client.insert(data, tables, tf.constant(1.0, dtype=tf.float64))

  def test_checks_that_priorities_argument_has_dtype_float64(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant(['dist']),
                    tf.constant([1.0], dtype=tf.float32))

  def test_checks_that_tables_and_priorities_arguments_have_same_shape(self):
    client = tf_client.TFClient(self._client.server_address)
    with self.assertRaises(ValueError):
      client.insert(self.data, tf.constant(['dist', 'dist2']),
                    tf.constant([1.0], dtype=tf.float64))

  def test_single_table_insert(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      insert_op = client.insert(
          data=[tf.constant([1, 2, 3], dtype=tf.int8)],
          tables=tf.constant(['dist']),
          priorities=tf.constant([1.0], dtype=tf.float64))
      sample_op = client.sample('dist', [tf.int8])

      # Check that insert op succeeds.
      self.assertIsNone(session.run(insert_op))

      # Check that the sampled data matches the inserted.
      sample = session.run(sample_op)
      self.assertLen(sample.data, 1)
      np.testing.assert_equal(
          np.array([1, 2, 3], dtype=np.int8), sample.data[0])

  def test_multi_table_insert(self):
    with self.session() as session:
      client = tf_client.TFClient(self._client.server_address)
      insert_op = client.insert(
          data=[tf.constant([1, 2, 3], dtype=tf.int8)],
          tables=tf.constant(['dist', 'dist2']),
          priorities=tf.constant([1.0, 2.0], dtype=tf.float64))

      sample_ops = [
          client.sample('dist', [tf.int8]),
          client.sample('dist2', [tf.int8])
      ]

      # Check that insert op succeeds.
      self.assertIsNone(session.run(insert_op))

      # Check that the sampled data matches the inserted in all tables.
      for sample_op in sample_ops:
        sample = session.run(sample_op)
        self.assertLen(sample.data, 1)
        np.testing.assert_equal(
            np.array([1, 2, 3], dtype=np.int8), sample.data[0])


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.test.main()
