import unittest
import numpy as np
import tensorflow as tf

from model_compression_toolkit.core.keras.tf_tensor_numpy import to_tf_tensor, tf_tensor_to_numpy


class TestToTfTensor(unittest.TestCase):

    def test_numpy_array_conversion(self):
        numpy_array = np.array([1, 2, 3])
        tf_tensor = to_tf_tensor(numpy_array)
        self.assertTrue(isinstance(tf_tensor, tf.Tensor))
        self.assertTrue(tf.reduce_all(tf_tensor == numpy_array))


    def test_tensor_conversion(self):
        tf_tensor = tf.constant([1, 2, 3])
        converted_tensor = to_tf_tensor(tf_tensor)
        self.assertTrue(tf.reduce_all(tf_tensor == converted_tensor))

    def test_list_conversion(self):
        input_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        converted_list = to_tf_tensor(input_list)
        for idx, item in enumerate(converted_list):
            self.assertTrue(isinstance(item, tf.Tensor))
            self.assertTrue(tf.reduce_all(input_list[idx] == item))

    def test_tuple_conversion(self):
        input_tuple = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        converted_tuple = to_tf_tensor(input_tuple)
        for idx, item in enumerate(converted_tuple):
            self.assertTrue(isinstance(item, tf.Tensor))
            self.assertTrue(tf.reduce_all(input_tuple[idx] == item))

    def test_unsupported_type(self):
        unsupported_type = "string"
        with self.assertRaises(Exception) as e:
            to_tf_tensor(unsupported_type)
        self.assertEqual("Unsupported type for conversion to TF tensor: <class 'str'>.", str(e.exception))



class TestTFTensorToNumpy(unittest.TestCase):

    def test_np_array(self):
        arr = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(tf_tensor_to_numpy(arr), arr))

    def test_list_of_tensors(self):
        tensor_list = [tf.constant([1, 1]), tf.constant([2, 2]), tf.constant([3, 3])]
        numpy_list = tf_tensor_to_numpy(tensor_list)
        for i, tensor in enumerate(tensor_list):
            self.assertTrue(np.array_equal(numpy_list[i], tensor.numpy()))

    def test_tuple_of_tensors(self):
        tensor_tuple = (tf.constant([1, 1]), tf.constant([2, 2]), tf.constant([3, 3]))
        numpy_tuple = tf_tensor_to_numpy(tensor_tuple)
        for i, tensor in enumerate(tensor_tuple):
            self.assertTrue(np.array_equal(numpy_tuple[i], tensor.numpy()))

    def test_single_tensor(self):
        tensor = tf.constant([42, 42])
        numpy_array = tf_tensor_to_numpy(tensor, is_single_tensor=True)
        self.assertTrue(np.array_equal(numpy_array, tensor.numpy()))

    def test_single_tensor_list(self):
        tensor = [tf.constant([42, 42])]
        numpy_array = tf_tensor_to_numpy(tensor, is_single_tensor=True)
        self.assertTrue(np.array_equal(numpy_array[0], tensor[0].numpy()))

    def test_single_tensor_tuple(self):
        tensor = (tf.constant([42, 42]),)
        numpy_array = tf_tensor_to_numpy(tensor, is_single_tensor=True)
        self.assertTrue(np.array_equal(numpy_array[0], tensor[0].numpy()))

    def test_scalar_float(self):
        scalar_float = 3.14
        numpy_array = tf_tensor_to_numpy(scalar_float)
        self.assertTrue(np.array_equal(numpy_array, np.array([scalar_float])))

    def test_unsupported_type(self):
        unsupported_type = 'string'
        with self.assertRaises(Exception) as e:
            tf_tensor_to_numpy(unsupported_type)
        self.assertEqual("Unsupported type for conversion to Numpy array: <class 'str'>.", str(e.exception))

if __name__ == '__main__':
    unittest.main()
