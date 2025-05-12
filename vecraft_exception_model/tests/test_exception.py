import unittest

from vecraft_exception_model.exception import VectorDimensionMismatchException, InvalidDataException, TimeoutException, \
    ChecksumValidationFailureError, RecordNotFoundError, NullOrZeroVectorException, UnsupportedMetricException, \
    CollectionNotExistedException, CollectionAlreadyExistedException, StorageFailureException, \
    VectorIndexBuildingException, MetadataIndexBuildingException, DocumentIndexBuildingException, \
    TsnePlotGeneratingFailureException, WriteConflictException, ReadWriteConflictException


class BaseExceptionTest(unittest.TestCase):
    """Base test class for common exception testing behavior"""

    exception_class = None  # Will be set in subclasses

    def setUp(self):
        # Skip tests in this base class
        if self.__class__ == BaseExceptionTest:
            self.skipTest("Base class")

    def test_inheritance(self):
        """Test that the exception inherits from Exception"""
        self.assertTrue(issubclass(self.exception_class, Exception))

    def test_basic_instantiation(self):
        """Test that exception can be instantiated with just a message"""
        message = "Test error message"
        exc = self.exception_class(message)
        self.assertEqual(exc.message, message)
        self.assertEqual(str(exc), message)

    def test_raise_and_catch(self):
        """Test that exception can be raised and caught"""
        message = "Test error message"
        try:
            raise self.exception_class(message)
        except self.exception_class as e:
            self.assertEqual(e.message, message)

class TestRecordNotFoundError(BaseExceptionTest):
    exception_class = RecordNotFoundError

    def test_with_record_id_and_collection(self):
        """Test with record_id and collection parameters"""
        message = "Record not found"
        record_id = "12345"
        collection = "test_collection"

        exc = RecordNotFoundError(message, record_id, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.message, message)
        self.assertEqual(str(exc), f"{message} (record_id={record_id}, collection={collection})")

    def test_with_record_id_only(self):
        """Test with only record_id parameter"""
        message = "Record not found"
        record_id = "12345"

        exc = RecordNotFoundError(message, record_id)
        self.assertEqual(exc.record_id, record_id)
        self.assertIsNone(exc.collection)
        self.assertEqual(str(exc), message)

    def test_with_collection_only(self):
        """Test with only collection parameter"""
        message = "Record not found"
        collection = "test_collection"

        exc = RecordNotFoundError(message, None, collection)
        self.assertIsNone(exc.record_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(str(exc), message)


class TestChecksumValidationFailureError(BaseExceptionTest):
    exception_class = ChecksumValidationFailureError

    def test_with_record_id_and_collection(self):
        message = "Checksum validation failed"
        record_id = "12345"
        collection = "test_collection"

        exc = ChecksumValidationFailureError(message, record_id, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(str(exc), f"{message} (record_id={record_id}, collection={collection})")


class TestTimeoutException(BaseExceptionTest):
    exception_class = TimeoutException

    def test_with_all_params(self):
        message = "Operation timed out"
        operation = "search"
        timeout_seconds = 30

        exc = TimeoutException(message, operation, timeout_seconds)
        self.assertEqual(exc.operation, operation)
        self.assertEqual(exc.timeout_seconds, timeout_seconds)
        self.assertEqual(str(exc), f"{message} (operation={operation}, timeout={timeout_seconds}s)")

    def test_with_operation_only(self):
        message = "Operation timed out"
        operation = "search"

        exc = TimeoutException(message, operation)
        self.assertEqual(exc.operation, operation)
        self.assertIsNone(exc.timeout_seconds)
        self.assertEqual(str(exc), f"{message} (operation={operation})")


class TestInvalidDataException(BaseExceptionTest):
    exception_class = InvalidDataException

    def test_with_all_params(self):
        message = "Data validation failed"
        collection = "test_collection"
        record_id = "12345"
        cause = ValueError("Invalid format")

        exc = InvalidDataException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")


class TestVectorDimensionMismatchException(BaseExceptionTest):
    exception_class = VectorDimensionMismatchException

    def test_with_all_params(self):
        message = "Vector dimension mismatch"
        provided_dim = 128
        expected_dim = 256
        collection = "test_collection"

        exc = VectorDimensionMismatchException(message, provided_dim, expected_dim, collection)
        self.assertEqual(exc.provided_dim, provided_dim)
        self.assertEqual(exc.expected_dim, expected_dim)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(
            str(exc),
            f"{message} (provided_dim={provided_dim}, expected_dim={expected_dim}, collection={collection})"
        )


# New test classes for remaining exceptions:

class TestNullOrZeroVectorException(BaseExceptionTest):
    exception_class = NullOrZeroVectorException

    def test_with_all_params(self):
        message = "Null or zero vector detected"
        vector_id = "vec123"
        collection = "test_collection"

        exc = NullOrZeroVectorException(message, vector_id, collection)
        self.assertEqual(exc.vector_id, vector_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(str(exc), f"{message} (vector_id={vector_id}, collection={collection})")

    def test_with_vector_id_only(self):
        message = "Null or zero vector detected"
        vector_id = "vec123"

        exc = NullOrZeroVectorException(message, vector_id)
        self.assertEqual(exc.vector_id, vector_id)
        self.assertIsNone(exc.collection)
        self.assertEqual(str(exc), f"{message} (vector_id={vector_id})")


class TestUnsupportedMetricException(BaseExceptionTest):
    exception_class = UnsupportedMetricException

    def test_with_all_params(self):
        message = "Unsupported distance metric"
        metric = "mahalanobis"
        supported_metrics = ["cosine", "euclidean", "dot"]

        exc = UnsupportedMetricException(message, metric, supported_metrics)
        self.assertEqual(exc.metric, metric)
        self.assertEqual(exc.supported_metrics, supported_metrics)
        self.assertEqual(str(exc), f"{message} (metric={metric}, supported_metrics={supported_metrics})")

    def test_with_metric_only(self):
        message = "Unsupported distance metric"
        metric = "mahalanobis"

        exc = UnsupportedMetricException(message, metric)
        self.assertEqual(exc.metric, metric)
        self.assertIsNone(exc.supported_metrics)
        self.assertEqual(str(exc), f"{message} (metric={metric})")


class TestCollectionNotExistedException(BaseExceptionTest):
    exception_class = CollectionNotExistedException

    def test_with_collection_name(self):
        message = "Collection does not exist"
        collection_name = "nonexistent_collection"

        exc = CollectionNotExistedException(message, collection_name)
        self.assertEqual(exc.collection_name, collection_name)
        self.assertEqual(str(exc), f"{message} (collection_name={collection_name})")


class TestCollectionAlreadyExistedException(BaseExceptionTest):
    exception_class = CollectionAlreadyExistedException

    def test_with_collection_name(self):
        message = "Collection already exists"
        collection_name = "existing_collection"

        exc = CollectionAlreadyExistedException(message, collection_name)
        self.assertEqual(exc.collection_name, collection_name)
        self.assertEqual(str(exc), f"{message} (collection_name={collection_name})")


class TestStorageFailureException(BaseExceptionTest):
    exception_class = StorageFailureException

    def test_with_all_params(self):
        message = "Storage operation failed"
        collection = "test_collection"
        record_id = "12345"
        cause = IOError("Disk full")

        exc = StorageFailureException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_only(self):
        message = "Storage operation failed"
        collection = "test_collection"

        exc = StorageFailureException(message, collection)
        self.assertEqual(exc.collection, collection)
        self.assertIsNone(exc.record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection})")


class TestVectorIndexBuildingException(BaseExceptionTest):
    exception_class = VectorIndexBuildingException

    def test_with_all_params(self):
        message = "Vector index building failed"
        collection = "test_collection"
        record_id = "12345"
        cause = RuntimeError("Out of memory")

        exc = VectorIndexBuildingException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_only(self):
        message = "Vector index building failed"
        collection = "test_collection"

        exc = VectorIndexBuildingException(message, collection)
        self.assertEqual(exc.collection, collection)
        self.assertIsNone(exc.record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection})")


class TestMetadataIndexBuildingException(BaseExceptionTest):
    exception_class = MetadataIndexBuildingException

    def test_with_all_params(self):
        message = "Metadata index building failed"
        collection = "test_collection"
        record_id = "12345"
        cause = RuntimeError("Invalid metadata format")

        exc = MetadataIndexBuildingException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_only(self):
        message = "Metadata index building failed"
        collection = "test_collection"

        exc = MetadataIndexBuildingException(message, collection)
        self.assertEqual(exc.collection, collection)
        self.assertIsNone(exc.record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection})")


class TestDocumentIndexBuildingException(BaseExceptionTest):
    exception_class = DocumentIndexBuildingException

    def test_with_all_params(self):
        message = "Document index building failed"
        collection = "test_collection"
        record_id = "12345"
        cause = RuntimeError("Invalid document format")

        exc = DocumentIndexBuildingException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_only(self):
        message = "Document index building failed"
        collection = "test_collection"

        exc = DocumentIndexBuildingException(message, collection)
        self.assertEqual(exc.collection, collection)
        self.assertIsNone(exc.record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection})")


class TestTsnePlotGeneratingFailureException(BaseExceptionTest):
    exception_class = TsnePlotGeneratingFailureException

    def test_with_all_params(self):
        message = "t-SNE plot generation failed"
        collection_name = "test_collection"
        cause = RuntimeError("Insufficient data points")

        exc = TsnePlotGeneratingFailureException(message, collection_name, cause)
        self.assertEqual(exc.collection_name, collection_name)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection_name={collection_name}, cause={cause})")

    def test_with_collection_name_only(self):
        message = "t-SNE plot generation failed"
        collection_name = "test_collection"

        exc = TsnePlotGeneratingFailureException(message, collection_name)
        self.assertEqual(exc.collection_name, collection_name)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection_name={collection_name})")


class TestWriteConflictException(BaseExceptionTest):
    exception_class = WriteConflictException

    def test_with_all_params(self):
        message = "Write-write conflict detected"
        collection = "test_collection"
        record_id = "12345"
        cause = RuntimeError("Concurrent modification")

        exc = WriteConflictException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_and_record_id(self):
        message = "Write-write conflict detected"
        collection = "test_collection"
        record_id = "12345"

        exc = WriteConflictException(message, collection, record_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id})")


class TestReadWriteConflictException(BaseExceptionTest):
    exception_class = ReadWriteConflictException

    def test_with_all_params(self):
        message = "Read-write conflict detected"
        collection = "test_collection"
        record_id = "12345"
        cause = RuntimeError("Snapshot isolation violation")

        exc = ReadWriteConflictException(message, collection, record_id, cause)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertEqual(exc.cause, cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id}, cause={cause})")

    def test_with_collection_and_record_id(self):
        message = "Read-write conflict detected"
        collection = "test_collection"
        record_id = "12345"

        exc = ReadWriteConflictException(message, collection, record_id)
        self.assertEqual(exc.collection, collection)
        self.assertEqual(exc.record_id, record_id)
        self.assertIsNone(exc.cause)
        self.assertEqual(str(exc), f"{message} (collection={collection}, record id={record_id})")


if __name__ == '__main__':
    unittest.main()