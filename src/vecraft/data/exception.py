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


# Key Exception Types to Surface to Users

class AuthenticationException(Exception):
    """
    Exception raised when credentials are missing, invalid, or lack necessary permissions.
    """

    def __init__(self, message, user_id=None):
        self.user_id = user_id
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.user_id is not None:
            return f"{self.message} (user_id={self.user_id})"
        return self.message


class ConnectionException(Exception):
    """
    Exception raised when there's a network-level failure (e.g. DNS lookup failure,
    TCP timeout) between client and server.
    """

    def __init__(self, message, host=None, port=None):
        self.host = host
        self.port = port
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.host is not None and self.port is not None:
            return f"{self.message} (host={self.host}, port={self.port})"
        elif self.host is not None:
            return f"{self.message} (host={self.host})"
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


class InvalidQueryException(Exception):
    """
    Exception raised when the user supplies a malformed query
    (bad JSON, unsupported filter syntax, etc.).
    """

    def __init__(self, message, query=None):
        self.query = query
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if self.query is not None:
            return f"{self.message} (query={self.query})"
        return self.message


class IndexNotFoundException(Exception):
    """
    Exception raised when referring to a non-existent namespace, collection
    or index name.
    """

    def __init__(self, message, namespace=None, collection=None, index_name=None):
        self.namespace = namespace
        self.collection = collection
        self.index_name = index_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.namespace is not None:
            details.append(f"namespace={self.namespace}")
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.index_name is not None:
            details.append(f"index_name={self.index_name}")

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


class QuotaExceededException(Exception):
    """
    Exception raised when user has hit subscription or tenant limits
    (max vectors, storage bytes, QPS).
    """

    def __init__(self, message, limit_type=None, current_usage=None, max_allowed=None):
        self.limit_type = limit_type
        self.current_usage = current_usage
        self.max_allowed = max_allowed
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.limit_type is not None:
            details.append(f"limit_type={self.limit_type}")
        if self.current_usage is not None:
            details.append(f"current_usage={self.current_usage}")
        if self.max_allowed is not None:
            details.append(f"max_allowed={self.max_allowed}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class CollectionUnavailableException(Exception):
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


class StorageFailureException(Exception):
    """
    Exception raised when disk full, write I/O error, or underlying storage
    layer fault during persistence.
    """

    def __init__(self, message, storage_path=None, error_code=None):
        self.storage_path = storage_path
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.storage_path is not None:
            details.append(f"storage_path={self.storage_path}")
        if self.error_code is not None:
            details.append(f"error_code={self.error_code}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class IndexBuildingException(Exception):
    """
    Exception raised when there's an internal error while constructing
    or optimizing the ANN index.
    """

    def __init__(self, message, index_name=None, collection=None, phase=None):
        self.index_name = index_name
        self.collection = collection
        self.phase = phase
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.index_name is not None:
            details.append(f"index_name={self.index_name}")
        if self.collection is not None:
            details.append(f"collection={self.collection}")
        if self.phase is not None:
            details.append(f"phase={self.phase}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class PermissionDeniedException(Exception):
    """
    Exception raised when even if authenticated, the user isn't authorized
    to perform this operation on the resource.
    """

    def __init__(self, message, user_id=None, resource=None, operation=None):
        self.user_id = user_id
        self.resource = resource
        self.operation = operation
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.user_id is not None:
            details.append(f"user_id={self.user_id}")
        if self.resource is not None:
            details.append(f"resource={self.resource}")
        if self.operation is not None:
            details.append(f"operation={self.operation}")

        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class PartialFailureException(Exception):
    """
    Exception raised in bulk operations, some items succeeded while
    others failed—contains per-item sub-errors.
    """

    def __init__(self, message, success_count=None, failure_count=None, errors=None):
        self.success_count = success_count
        self.failure_count = failure_count
        self.errors = errors if errors is not None else []
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = []
        if self.success_count is not None:
            details.append(f"success_count={self.success_count}")
        if self.failure_count is not None:
            details.append(f"failure_count={self.failure_count}")

        result = self.message
        if details:
            result += f" ({', '.join(details)})"

        if self.errors:
            result += "\nErrors:"
            for i, error in enumerate(self.errors):
                result += f"\n  {i + 1}. {str(error)}"

        return result