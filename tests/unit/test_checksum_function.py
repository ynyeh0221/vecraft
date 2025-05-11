import json
import unittest

from src.vecraft_db.core.data_model.checksum_util import _prepare_field_bytes, _concat_bytes, get_checksum_func


class TestChecksumFunctions(unittest.TestCase):
    """Test cases for checksum functions."""

    def test_predefined_checksums(self):
        """Test that predefined checksum algorithms work correctly."""
        test_data = b"test data"

        # Test CRC32
        crc32_func = get_checksum_func('crc32')
        self.assertEqual(crc32_func(test_data), 'd308aeb2')

        # Test MD5
        md5_func = get_checksum_func('md5')
        self.assertEqual(md5_func(test_data), 'eb733a00c0c9d336e65691a37ab54293')

        # Test SHA1
        sha1_func = get_checksum_func('sha1')
        self.assertEqual(sha1_func(test_data), 'f48dd853820860816c75d54d0f584dc863327a7c')

        # Test SHA256
        sha256_func = get_checksum_func('sha256')
        self.assertEqual(sha256_func(test_data), '916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9')

    def test_case_insensitive_algorithm(self):
        """Test that algorithm names are case-insensitive."""
        test_data = b"test"

        # Test different casings
        md5_lower = get_checksum_func('md5')
        md5_upper = get_checksum_func('MD5')
        md5_mixed = get_checksum_func('Md5')

        expected = '098f6bcd4621d373cade4e832627b4f6'
        self.assertEqual(md5_lower(test_data), expected)
        self.assertEqual(md5_upper(test_data), expected)
        self.assertEqual(md5_mixed(test_data), expected)

    def test_custom_checksum_function(self):
        """Test that custom checksum functions are accepted."""

        def custom_checksum(data: bytes) -> str:
            return f"custom_{len(data)}"

        func = get_checksum_func(custom_checksum)
        self.assertEqual(func(b"test"), "custom_4")
        self.assertEqual(func(b"longer test"), "custom_11")

    def test_hashlib_fallback(self):
        """Test fallback to hashlib for algorithms not in predefined list."""
        test_data = b"test"

        # SHA224 is not in predefined list but is available in hashlib
        sha224_func = get_checksum_func('sha224')
        expected = '90a3ed9e32b2aaf4c61c410eb925426119e1a9dc53d4286ade99a809'
        self.assertEqual(sha224_func(test_data), expected)

    def test_unsupported_algorithm(self):
        """Test that unsupported algorithms raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            get_checksum_func('nonexistent_algorithm')
        self.assertIn("Unsupported checksum algorithm", str(ctx.exception))


class TestPrepareFieldBytes(unittest.TestCase):
    """Test cases for _prepare_field_bytes function."""

    def test_json_serializable_values(self):
        """Test conversion of JSON-serializable values to bytes."""
        # Test string
        self.assertEqual(_prepare_field_bytes("test"), b'"test"')

        # Test number
        self.assertEqual(_prepare_field_bytes(123), b'123')

        # Test float
        self.assertEqual(_prepare_field_bytes(123.45), b'123.45')

        # Test boolean
        self.assertEqual(_prepare_field_bytes(True), b'true')

        # Test None
        self.assertEqual(_prepare_field_bytes(None), b'null')

        # Test list
        self.assertEqual(_prepare_field_bytes([1, 2, 3]), b'[1, 2, 3]')

        # Test dict (sorted keys)
        self.assertEqual(_prepare_field_bytes({"b": 2, "a": 1}), b'{"a": 1, "b": 2}')

    def test_dict_ordering(self):
        """Test that dictionaries are serialized with sorted keys."""
        dict1 = {"c": 3, "a": 1, "b": 2}
        dict2 = {"b": 2, "c": 3, "a": 1}

        # Both should produce the same result with sorted keys
        expected = b'{"a": 1, "b": 2, "c": 3}'
        self.assertEqual(_prepare_field_bytes(dict1), expected)
        self.assertEqual(_prepare_field_bytes(dict2), expected)

    def test_nested_structures(self):
        """Test nested dictionaries and lists."""
        nested = {
            "list": [1, 2, {"nested": True}],
            "dict": {"a": 1, "b": 2}
        }
        result = _prepare_field_bytes(nested)
        # Load back to check structure is preserved
        loaded = json.loads(result.decode('utf-8'))
        self.assertEqual(loaded["list"], [1, 2, {"nested": True}])
        self.assertEqual(loaded["dict"], {"a": 1, "b": 2})

    def test_non_json_serializable(self):
        """Test that non-JSON serializable values fall back to repr()."""

        class CustomClass:
            def __repr__(self):
                return "CustomClass()"

        obj = CustomClass()
        result = _prepare_field_bytes(obj)
        self.assertEqual(result, b'CustomClass()')

        # Test with a set (not JSON serializable)
        s = {1, 2, 3}
        result = _prepare_field_bytes(s)
        self.assertTrue(result.startswith(b'{') and result.endswith(b'}'))
        self.assertIn(b'1', result)
        self.assertIn(b'2', result)
        self.assertIn(b'3', result)


class TestConcatBytes(unittest.TestCase):
    """Test cases for _concat_bytes function."""

    def test_empty_list(self):
        """Test concatenation of empty list."""
        self.assertEqual(_concat_bytes([]), b"")

    def test_single_component(self):
        """Test concatenation of single component."""
        self.assertEqual(_concat_bytes([b"test"]), b"test")

    def test_multiple_components(self):
        """Test concatenation of multiple components."""
        components = [b"hello", b" ", b"world"]
        self.assertEqual(_concat_bytes(components), b"hello world")

    def test_empty_components(self):
        """Test concatenation with empty components."""
        components = [b"start", b"", b"middle", b"", b"end"]
        self.assertEqual(_concat_bytes(components), b"startmiddleend")

    def test_order_preservation(self):
        """Test that component order is preserved."""
        components = [b"1", b"2", b"3", b"4"]
        self.assertEqual(_concat_bytes(components), b"1234")

        # Different order should give different result
        components_reordered = [b"4", b"3", b"2", b"1"]
        self.assertEqual(_concat_bytes(components_reordered), b"4321")


class TestIntegration(unittest.TestCase):
    """Integration tests using multiple functions together."""

    def test_checksum_with_prepared_bytes(self):
        """Test using checksum functions with _prepare_field_bytes."""
        data = {"key": "value", "number": 123}
        prepared = _prepare_field_bytes(data)

        # Test with different algorithms
        md5_func = get_checksum_func('md5')
        sha256_func = get_checksum_func('sha256')

        md5_result = md5_func(prepared)
        sha256_result = sha256_func(prepared)

        # Results should be deterministic
        self.assertEqual(len(md5_result), 32)  # MD5 hex length
        self.assertEqual(len(sha256_result), 64)  # SHA256 hex length

        # Same data should produce same checksums
        prepared2 = _prepare_field_bytes(data)
        self.assertEqual(md5_func(prepared2), md5_result)
        self.assertEqual(sha256_func(prepared2), sha256_result)

    def test_concat_and_checksum(self):
        """Test concatenating multiple prepared fields and checksumming."""
        field1 = _prepare_field_bytes("field1")
        field2 = _prepare_field_bytes(123)
        field3 = _prepare_field_bytes({"key": "value"})

        concatenated = _concat_bytes([field1, field2, field3])

        md5_func = get_checksum_func('md5')
        checksum = md5_func(concatenated)

        # Should be deterministic
        concatenated2 = _concat_bytes([field1, field2, field3])
        checksum2 = md5_func(concatenated2)
        self.assertEqual(checksum, checksum2)

        # Different order should give different checksum
        concatenated3 = _concat_bytes([field3, field1, field2])
        checksum3 = md5_func(concatenated3)
        self.assertNotEqual(checksum, checksum3)


if __name__ == '__main__':
    unittest.main()