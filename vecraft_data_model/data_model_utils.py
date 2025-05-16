from vecraft_data_model.data_models import DataPacketModel, NumpyArray, QueryPacketModel
from vecraft_data_model.data_packet import DataPacket
from vecraft_data_model.query_packet import QueryPacket
from vecraft_exception_model.exception import InvalidDataException


class DataModelUtils:
    @staticmethod
    def convert_to_data_packet(model: DataPacketModel) -> DataPacket:
        """Convert API model to DataPacket with mandatory checksum validation"""
        # Create the packet (this will compute checksum automatically)
        if model.type == "RECORD":
            packet_data = {
                'record_id': model.record_id,
                'original_data': model.original_data,
                'metadata': model.metadata,
                'checksum_algorithm': model.checksum_algorithm,
            }

            # Convert vector if present
            packet_data['vector'] = model.vector.to_numpy()
            packet = DataPacket.create_record(**packet_data)
        elif model.type == "TOMBSTONE":
            packet_data = {
                'record_id': model.record_id,
                'checksum_algorithm': model.checksum_algorithm,
            }
            packet = DataPacket.create_tombstone(**packet_data)
        elif model.type == "NONEXISTENT":
            packet_data = {
                'record_id': model.record_id,
                'checksum_algorithm': model.checksum_algorithm,
            }
            packet = DataPacket.create_nonexistent(**packet_data)
        else:
            raise InvalidDataException(f"Unknown data packet type: {model.type}")

        # TODO Complete checksum validation # NOSONAR
        packet.validate_checksum()

        return packet


    @staticmethod
    def convert_from_data_packet(packet: DataPacket) -> DataPacketModel:
        """Convert DataPacket to an API model with checksum validation"""
        packet.validate_checksum()

        packet_dict = packet.to_dict()

        # Convert numpy array to NumpyArray model
        if packet.vector is not None:
            packet_dict['vector'] = NumpyArray.from_numpy(packet.vector)

        # TODO Complete checksum validation # NOSONAR
        packet.validate_checksum()

        return DataPacketModel(**packet_dict)


    @staticmethod
    def convert_to_query_packet(model: QueryPacketModel) -> QueryPacket:
        """Convert API model to QueryPacket with mandatory checksum validation"""
        query_packet = QueryPacket(
            query_vector=model.query_vector.to_numpy(),
            k=model.k,
            where=model.where,
            where_document=model.where_document,
            checksum_algorithm=model.checksum_algorithm
        )

        # TODO Complete checksum validation # NOSONAR
        query_packet.validate_checksum()

        return query_packet

    @staticmethod
    def convert_from_query_packet(packet: QueryPacket) -> QueryPacketModel:
        """Convert QueryPacket to an API model with checksum validation"""
        packet.validate_checksum()

        packet_dict = packet.to_dict()

        # Convert numpy array to NumpyArray model for query_vector
        if packet.query_vector is not None:
            packet_dict['query_vector'] = NumpyArray.from_numpy(packet.query_vector)

        # Include the computed checksum
        packet_dict['checksum'] = packet.checksum

        return QueryPacketModel(**packet_dict)