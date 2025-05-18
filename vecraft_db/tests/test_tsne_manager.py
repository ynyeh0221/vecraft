import logging
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vecraft_db.engine.tsne_manager import TSNEManager
from vecraft_exception_model.exception import TsnePlotGeneratingFailureException


class TestTSNEManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.tsne_manager = TSNEManager(logger=self.logger)

        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.temp_dir, "test_tsne.png")

        # Mock version
        self.version = Mock()
        self.version.storage = Mock()
        self.version.storage.get_all_record_locations.return_value = {"id1": "loc1", "id2": "loc2"}

        # Mock records
        self.record1 = Mock()
        self.record1.vector = np.array([1.0, 2.0, 3.0])

        self.record2 = Mock()
        self.record2.vector = np.array([4.0, 5.0, 6.0])

        # Mock get_record_func
        self.get_record_func = Mock()
        self.get_record_func.side_effect = lambda v, rid: {
            "id1": self.record1,
            "id2": self.record2
        }.get(rid)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    @patch.object(TSNEManager, '_generate_tsne')
    def test_generate_tsne_plot_with_provided_record_ids(self, mock_generate_tsne):
        """Test generate_tsne_plot with explicitly provided record IDs."""
        # Set up the mock
        mock_generate_tsne.return_value = self.outfile

        # Call the method
        result = self.tsne_manager.generate_tsne_plot(
            name="test_collection",
            version=self.version,
            get_record_func=self.get_record_func,
            record_ids=["id1", "id2"],
            outfile=self.outfile
        )

        # Assertions
        self.assertEqual(result, self.outfile)
        self.get_record_func.assert_any_call(self.version, "id1")
        self.get_record_func.assert_any_call(self.version, "id2")

        # Check the arguments passed to _generate_tsne
        mock_generate_tsne.assert_called_once()
        _, kwargs = mock_generate_tsne.call_args

        # Verify vectors are stacked correctly
        np.testing.assert_array_equal(
            kwargs['vectors'],
            np.vstack([self.record1.vector, self.record2.vector])
        )

        # Verify labels
        self.assertEqual(kwargs['labels'], ["id1", "id2"])

        # Verify other parameters
        self.assertEqual(kwargs['outfile'], self.outfile)
        self.assertEqual(kwargs['perplexity'], 30)  # default value
        self.assertEqual(kwargs['random_state'], 42)  # default value

    @patch.object(TSNEManager, '_generate_tsne')
    def test_generate_tsne_plot_with_all_records(self, mock_generate_tsne):
        """Test generate_tsne_plot with all records from the collection."""
        # Set up the mock
        mock_generate_tsne.return_value = self.outfile

        # Call the method with record_ids=None
        result = self.tsne_manager.generate_tsne_plot(
            name="test_collection",
            version=self.version,
            get_record_func=self.get_record_func,
            record_ids=None,
            outfile=self.outfile
        )

        # Assertions
        self.assertEqual(result, self.outfile)
        self.version.storage.get_all_record_locations.assert_called_once()
        self.get_record_func.assert_any_call(self.version, "id1")
        self.get_record_func.assert_any_call(self.version, "id2")

        # Check the arguments passed to _generate_tsne
        mock_generate_tsne.assert_called_once()

    @patch.object(TSNEManager, '_generate_tsne')
    def test_generate_tsne_plot_with_custom_parameters(self, mock_generate_tsne):
        """Test generate_tsne_plot with custom perplexity and random_state."""
        # Set up the mock
        mock_generate_tsne.return_value = self.outfile

        # Call the method with custom parameters
        result = self.tsne_manager.generate_tsne_plot(
            name="test_collection",
            version=self.version,
            get_record_func=self.get_record_func,
            record_ids=["id1", "id2"],
            perplexity=50,
            random_state=123,
            outfile=self.outfile
        )

        # Assertions
        self.assertEqual(result, self.outfile)

        # Check the arguments passed to _generate_tsne
        mock_generate_tsne.assert_called_once()
        _, kwargs = mock_generate_tsne.call_args

        # Verify custom parameters
        self.assertEqual(kwargs['perplexity'], 50)
        self.assertEqual(kwargs['random_state'], 123)

    def test_generate_tsne_plot_with_no_vectors(self):
        """Test generate_tsne_plot when no vectors are available."""
        # Mock get_record_func to return None for all IDs
        empty_get_record_func = Mock(return_value=None)

        # Call the method and expect a TsnePlotGeneratingFailureException with NullOrZeroVectorException as cause
        with self.assertRaises(TsnePlotGeneratingFailureException) as context:
            self.tsne_manager.generate_tsne_plot(
                name="test_collection",
                version=self.version,
                get_record_func=empty_get_record_func,
                record_ids=["id1", "id2"],
                outfile=self.outfile
            )

        # Verify the exception contains information about the original NullOrZeroVectorException
        self.assertIn("No vectors available for t-SNE visualization", str(context.exception))

    @patch.object(TSNEManager, '_generate_tsne')
    def test_generate_tsne_plot_with_exception_in_generate_tsne(self, mock_generate_tsne):
        """Test generate_tsne_plot when _generate_tsne raises an exception."""
        # Set up the mock to raise an exception
        mock_generate_tsne.side_effect = ValueError("Test error")

        # Call the method and expect an exception
        with self.assertRaises(TsnePlotGeneratingFailureException):
            self.tsne_manager.generate_tsne_plot(
                name="test_collection",
                version=self.version,
                get_record_func=self.get_record_func,
                record_ids=["id1", "id2"],
                outfile=self.outfile
            )

    @patch('vecraft_db.engine.tsne_manager.TSNE')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    def test_generate_tsne_method(self, mock_close, mock_subplots, mock_tsne_class):
        """Test the _generate_tsne static method with a valid perplexity value."""
        # Set up mock TSNE instance
        mock_tsne_instance = Mock()
        mock_tsne_class.return_value = mock_tsne_instance
        mock_tsne_instance.fit_transform.return_value = np.array([[1, 2], [3, 4]])

        # Set up mock figure and axes
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Call the method
        vectors = np.array([[1, 2, 3], [4, 5, 6]])
        labels = ["id1", "id2"]

        result = self.tsne_manager._generate_tsne(
            vectors=vectors,
            labels=labels,
            outfile=self.outfile,
            perplexity=1,  # Low perplexity for small sample
            random_state=42
        )

        # Assertions
        self.assertEqual(result, self.outfile)
        mock_tsne_class.assert_called_once_with(
            n_components=2, perplexity=1, init='random', random_state=42
        )
        mock_tsne_instance.fit_transform.assert_called_once_with(vectors)
        mock_fig.savefig.assert_called_once_with(self.outfile, dpi=300)
        mock_close.assert_called_once()

    def test_generate_tsne_with_invalid_vectors(self):
        """Test _generate_tsne with invalid vectors (not 2D)."""
        # Call the method with 1D vectors and expect an exception
        with self.assertRaises(ValueError):
            self.tsne_manager._generate_tsne(
                vectors=np.array([1, 2, 3]),  # 1D array
                outfile=self.outfile
            )


if __name__ == '__main__':
    unittest.main()