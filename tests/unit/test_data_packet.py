import unittest

import numpy as np

from src.vecraft.data.data_packet import DataPacket
from src.vecraft.data.query_packet import QueryPacket


class TestPackets(unittest.TestCase):

    def test_data_packet(self):
        data_packet = DataPacket.create_record(
            record_id="packet_1",
            original_data={"x": 1, "y": "abc"},
            vector=np.array([1, 2, 3], dtype=np.float32),
            metadata={"test_key": "test_value"}
        )
        data_packet_bytes = data_packet.to_dict()
        reconstructed_packet = DataPacket.from_dict(data_packet_bytes)

        self.assertEqual(data_packet, reconstructed_packet)

    def test_query_packet(self):
        query_packet = QueryPacket(
            query_vector=np.array([1, 1, 1], dtype=np.float32),
            k=2
        )
        query_packet_bytes = query_packet.to_dict()
        reconstructed_packet = QueryPacket.from_dict(query_packet_bytes)

        self.assertEqual(query_packet, reconstructed_packet)

if __name__ == '__main__':
    unittest.main()