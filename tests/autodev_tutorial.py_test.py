import unittest

class TestMysteriousFunction(unittest.TestCase):
    def test_already_sorted(self):
        input_list = [1, 2, 3, 4, 5]
        expected_output = [1, 2, 3, 4, 5]
        self.assertEqual(mysterious_function(input_list), expected_output)

    def test_reverse_sorted(self):
        input_list = [5, 4, 3, 2, 1]
        expected_output = [1, 2, 3, 4, 5]
        self.assertEqual(mysterious_function(input_list), expected_output)

    def test_unsorted(self):
        input_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        expected_output = [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
        self.assertEqual(mysterious_function(input_list), expected_output)

    def test_empty_list(self):
        input_list = []
        expected_output = []
        self.assertEqual(mysterious_function(input_list), expected_output)

    def test_single_element(self):
        input_list = [42]
        expected_output = [42]
        self.assertEqual(mysterious_function(input_list), expected_output)

    def test_duplicates(self):
        input_list = [5, 5, 5, 5, 5]
        expected_output = [5, 5, 5, 5, 5]
        self.assertEqual(mysterious_function(input_list), expected_output)

if __name__ == '__main__':
    unittest.main()