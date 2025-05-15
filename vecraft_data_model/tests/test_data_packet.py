import unittest

import numpy as np

from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_data_model.search_data_packet import SearchDataPacket


class TestPackets(unittest.TestCase):

    def test_data_packet_dict(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        data_packet_bytes = data_packet.to_dict()
        reconstructed_packet = DataPacket.from_dict(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_data_packet_dict_data_only(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32)
        )
        data_packet_bytes = data_packet.to_dict()
        reconstructed_packet = DataPacket.from_dict(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_data_packet_dict_metadata_only(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        data_packet_bytes = data_packet.to_dict()
        reconstructed_packet = DataPacket.from_dict(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_data_packet_bytes(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        data_packet_bytes = data_packet.to_bytes()
        reconstructed_packet = DataPacket.from_bytes(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_data_packet_bytes_data_only(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32)
        )
        data_packet_bytes = data_packet.to_bytes()
        reconstructed_packet = DataPacket.from_bytes(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_data_packet_bytes_metadata_only(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        data_packet_bytes = data_packet.to_bytes()
        reconstructed_packet = DataPacket.from_bytes(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_query_packet(self):
        query_packet = QueryPacket(
            query_vector=np.array([1, 1, 1], dtype=np.float32),
            k=2
        )
        query_packet_bytes = query_packet.to_dict()
        reconstructed_packet = QueryPacket.from_dict(query_packet_bytes)

        self.assertEqual(query_packet, reconstructed_packet)

    def test_search_data_packet_dict(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        search_data_packet = SearchDataPacket(data_packet=data_packet, distance=0.1)
        search_data_packet_bytes = search_data_packet.to_dict()
        reconstructed_packet = SearchDataPacket.from_dict(search_data_packet_bytes)

        self.assertEqual(search_data_packet, reconstructed_packet)

if __name__ == '__main__':
    unittest.main()