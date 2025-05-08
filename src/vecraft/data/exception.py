class RecordNotFoundError(Exception):
    """
    Exception raised when a requested record cannot be found in the database.

    This exception is raised by the VectorDB.get() method when attempting to 
    retrieve a record that doesn't exist or has been deleted.

    Attributes:
        record_id -- ID of the record that was not found
        collection -- name of the collection that was searched
        message -- explanation of the error
    """

    def __init__(self, message, record_id=None, collection=None):
        self.record_id = record_id
        self.collection = collection
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.record_id is not None and self.collection is not None:
            return f"{self.message} (record_id={self.record_id}, collection={self.collection})"
        return self.message


class ChecksumValidationFailureError(Exception):
    """
    Exception raised when a record fails checksum validation.
    """

    def __init__(self, message, record_id=None, collection=None):
        self.record_id = record_id
        self.collection = collection
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.record_id is not None and self.collection is not None:
            return f"{self.message} (record_id={self.record_id}, collection={self.collection})"
        return self.message


class TimeoutException(Exception):
    """
    Exception raised when an operation (e.g. index build, search) exceeded
    configured time budget.
    """

    def __init__(self, message, operation=None, timeout_seconds=None):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.operation is not None:
            details.append(f"operation={self.operation}")
        if self.timeout_seconds is not None:
            details.append(f"timeout={self.timeout_seconds}s")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class InvalidDataException(Exception):
    """
    Exception raised when a DataPacket cannot be validated.
    """
    def __init__(self, message, collection=None, record_id=None, cause: Exception = None):
        self.collection = collection
        self.record_id = record_id
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.record_id is not None:
            details.append(f"record id={self.record_id}")
        if self.cause is not None:
            details.append(f"cause={self.cause}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class VectorDimensionMismatchException(Exception):
    """
    Exception raised when the length of the provided vector doesn't match
    the index's configured dimension.
    """

    def __init__(self, message, provided_dim=None, expected_dim=None, collection=None):
        self.provided_dim = provided_dim
        self.expected_dim = expected_dim
        self.collection = collection
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.provided_dim is not None:
            details.append(f"provided_dim={self.provided_dim}")
        if self.expected_dim is not None:
            details.append(f"expected_dim={self.expected_dim}")
        if self.collection is not None:
            details.append(f"collection={self.collection}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class NullOrZeroVectorException(Exception):
    """
    Exception raised when user tries to insert or search with an empty
    (all-zeros) or null vector.
    """

    def __init__(self, message, vector_id=None, collection=None):
        self.vector_id = vector_id
        self.collection = collection
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.vector_id is not None:
            details.append(f"vector_id={self.vector_id}")
        if self.collection is not None:
            details.append(f"collection={self.collection}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class UnsupportedMetricException(Exception):
    """
    Exception raised when requested distance metric (e.g. Mahalanobis)
    isn't supported by this index.
    """

    def __init__(self, message, metric=None, supported_metrics=None):
        self.metric = metric
        self.supported_metrics = supported_metrics
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.metric is not None:
            details.append(f"metric={self.metric}")
        if self.supported_metrics is not None:
            details.append(f"supported_metrics={self.supported_metrics}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class CollectionNotExistedException(Exception):
    """
    Exception raised when target collection isn't available, so the operation can't proceed.
    """

    def __init__(self, message, collection_name=None):
        self.collection_name = collection_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection_name is not None:
            details.append(f"collection_name={self.collection_name}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class CollectionAlreadyExistedException(Exception):
    """
    Exception raised when trying to create a collection which has same name as existing one.
    """

    def __init__(self, message, collection_name=None):
        self.collection_name = collection_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection_name is not None:
            details.append(f"collection_name={self.collection_name}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class StorageFailureException(Exception):
    """
    Exception raised when disk full, write I/O error, or underlying storage
    layer fault during persistence.
    """
    def __init__(self, message, collection=None, record_id=None, cause: Exception = None):
        self.collection = collection
        self.record_id = record_id
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.record_id is not None:
            details.append(f"record id={self.record_id}")
        if self.cause is not None:
            details.append(f"cause={self.cause}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class VectorIndexBuildingException(Exception):
    """
    Exception raised when there's an internal error while constructing
    or optimizing the ANN vector index.
    """
    def __init__(self, message, collection=None, record_id=None, cause:Exception=None):
        self.collection = collection
        self.record_id = record_id
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.record_id is not None:
            details.append(f"record id={self.record_id}")
        if self.cause is not None:
            details.append(f"cause={self.cause}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message

class MetadataIndexBuildingException(Exception):
    """
    Exception raised when there's an internal error while constructing
    or optimizing the user metadata index.
    """
    def __init__(self, message, collection=None, record_id=None, cause:Exception=None):
        self.collection = collection
        self.record_id = record_id
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.record_id is not None:
            details.append(f"record id={self.record_id}")
        if self.cause is not None:
            details.append(f"cause={self.cause}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message

class DocumentIndexBuildingException(Exception):
    """
    Exception raised when there's an internal error while constructing
    or optimizing the user document index.
    """
    def __init__(self, message, collection=None, record_id=None, cause: Exception = None):
        self.collection = collection
        self.record_id = record_id
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.record_id is not None:
            details.append(f"record id={self.record_id}")
        if self.cause is not None:
            details.append(f"cause={self.cause}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message