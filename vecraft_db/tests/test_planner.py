import unittest

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_db.query.plan_nodes import InsertNode, DeleteNode, GetNode, SearchNode, TSNENode, ShutdownNode
from vecraft_db.query.planner import Planner


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = Planner()
        self.collection = "test_collection"

        # Create a real DataPacket for testing
        self.vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.data_packet = DataPacket.create_record(
            record_id="test_record",
            vector=self.vector,
            original_data={"text": "test document"},
            metadata={"tag": "test"}
        )

        # Create a real QueryPacket for testing
        self.query_vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.query_packet = QueryPacket(
            query_vector=self.query_vector,
            k=10,
            where={}
        )

        self.record_id = "test_id"

    def test_plan_insert(self):
        # Test that plan_insert returns an InsertNode with the right parameters
        result = self.planner.plan_insert(self.collection, self.data_packet)

        self.assertIsInstance(result, InsertNode)
        self.assertEqual(result.collection, self.collection)
        self.assertEqual(result.data_packet, self.data_packet)

    def test_plan_delete(self):
        # Test that plan_delete returns a DeleteNode with the right parameters
        result = self.planner.plan_delete(self.collection, self.data_packet)

        self.assertIsInstance(result, DeleteNode)
        self.assertEqual(result.collection, self.collection)
        self.assertEqual(result.data_packet, self.data_packet)

    def test_plan_get(self):
        # Test that plan_get returns a GetNode with the right parameters
        result = self.planner.plan_get(self.collection, self.record_id)

        self.assertIsInstance(result, GetNode)
        self.assertEqual(result.collection, self.collection)
        self.assertEqual(result.record_id, self.record_id)

    def test_plan_search_basic(self):
        # Test search planning with no where clause
        self.query_packet.where = {}

        result = self.planner.plan_search(self.collection, self.query_packet)

        self.assertIsInstance(result, SearchNode)
        self.assertEqual(result.collection, self.collection)
        # QueryPacket is modified and returned, so we can't directly compare objects
        self.assertEqual(result.query_packet.k, self.query_packet.k)
        self.assertTrue(np.array_equal(result.query_packet.query_vector, self.query_packet.query_vector))
        self.assertEqual(result.query_packet.where, {})

    def test_plan_search_with_list_condition(self):
        # Test search planning with a list condition that should be converted to $in
        # Create a new QueryPacket for this test to avoid side effects
        query_packet = QueryPacket(
            query_vector=self.query_vector,
            k=10,
            where={"field1": ["value1", "value2"]}
        )

        result = self.planner.plan_search(self.collection, query_packet)

        self.assertIsInstance(result, SearchNode)
        self.assertEqual(result.collection, self.collection)
        # Check that the list was converted to an $in condition
        self.assertEqual(result.query_packet.where, {"field1": {"$in": ["value1", "value2"]}})

    def test_plan_search_with_mixed_conditions(self):
        # Test search planning with both list and non-list conditions
        # Create a new QueryPacket for this test to avoid side effects
        query_packet = QueryPacket(
            query_vector=self.query_vector,
            k=10,
            where={
                "field1": ["value1", "value2"],
                "field2": {"$gt": 10}
            }
        )

        result = self.planner.plan_search(self.collection, query_packet)

        self.assertIsInstance(result, SearchNode)
        self.assertEqual(result.collection, self.collection)
        # Verify that only the list was converted to an $in condition
        expected_where = {
            "field1": {"$in": ["value1", "value2"]},
            "field2": {"$gt": 10}
        }
        self.assertEqual(result.query_packet.where, expected_where)

    def test_plan_tsne_plot_default_params(self):
        # Test TSNE planning with default parameters
        result = self.planner.plan_tsne_plot(self.collection)

        self.assertIsInstance(result, TSNENode)
        self.assertEqual(result.collection, self.collection)
        self.assertIsNone(result.record_ids)
        self.assertEqual(result.perplexity, 30)
        self.assertEqual(result.random_state, 42)
        self.assertEqual(result.outfile, "tsne.png")

    def test_plan_tsne_plot_custom_params(self):
        # Test TSNE planning with custom parameters
        record_ids = ["id1", "id2", "id3"]
        perplexity = 40
        random_state = 123
        outfile = "custom_tsne.png"

        result = self.planner.plan_tsne_plot(
            self.collection,
            record_ids=record_ids,
            perplexity=perplexity,
            random_state=random_state,
            outfile=outfile
        )

        self.assertIsInstance(result, TSNENode)
        self.assertEqual(result.collection, self.collection)
        self.assertEqual(result.record_ids, record_ids)
        self.assertEqual(result.perplexity, perplexity)
        self.assertEqual(result.random_state, random_state)
        self.assertEqual(result.outfile, outfile)

    def test_plan_shutdown(self):
        # Test that plan_shutdown returns a ShutdownNode
        result = self.planner.plan_shutdown()

        self.assertIsInstance(result, ShutdownNode)


if __name__ == '__main__':
    unittest.main()