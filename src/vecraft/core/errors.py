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