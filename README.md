# Flowchain - Method Chaining for TensorFlow

Extensive and simple tensor method chaining for TensorFlow

This package requires Python >= 3.7 and uses some Python magics for extensive injection. Therefore, there could be some unexpected behaviour among method chaining. If you have experienced such behaviours, please feel free to open new issue in this repo!

Currently, I am still finding a way to support code completion for IDEs like PyCharm, Visual Studio Code, etc.

Install via:
```
pip install flowchain
```

Add only 2 lines of code at the top of your code!
```python
from flowchain import enable_tensor_chaining

enable_tensor_chaining()  # this does everything for you.
```

This package makes following approach possible:
```python
# before
x = tf.abs(lhs - rhs)
x = tf.reduce_sum(x, 1)
x = tf.argmin(x, output_type=tf.int32)

# after
x = (lhs - rhs).abs().reduce_sum(1).argmin(output_type=tf.int32)
```

