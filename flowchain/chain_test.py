import pytest
import numpy as np
import tensorflow as tf

from flowchain import enable_tensor_chaining


enable_tensor_chaining()


def test_variable_basic_chaining():
    x = tf.Variable(initial_value=[[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    ans1 = tf.reduce_mean(x)
    ans2 = x.reduce_mean()
    assert np.allclose(ans1.numpy(), ans2.numpy())


def test_variable_multiple_chaining():
    x = tf.Variable(initial_value=[[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    ans1 = tf.reduce_min(tf.nn.sigmoid(x), axis=-1)
    ans2 = x.sigmoid().reduce_min(-1)
    assert np.allclose(ans1.numpy(), ans2.numpy())


def test_tensor_basic_chaining():
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    ans1 = tf.reduce_min(x)
    ans2 = x.reduce_min()
    assert np.allclose(ans1.numpy(), ans2.numpy())


def test_tensor_multiple_chaining():
    x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    ans1 = tf.reduce_min(tf.nn.sigmoid(x), axis=-1)
    ans2 = x.sigmoid().reduce_min(-1)
    assert np.allclose(ans1.numpy(), ans2.numpy())


if __name__ == '__main__':
    pytest.main()
