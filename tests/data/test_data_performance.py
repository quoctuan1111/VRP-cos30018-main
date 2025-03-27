import pytest
from src.data.data_processor import DataProcessor

class TestDataProcessor:
    @pytest.fixture
    def processor(self):
        processor = DataProcessor()
        processor.load_data('distance.csv', 'order_large.csv')
        return processor

    def test_data_loading(self, processor):
        assert processor.distance_data is not None
        assert processor.order_data is not None
        assert len(processor.cities) > 0

    def test_distance_matrix(self, processor):
        assert processor.distance_matrix is not None
        assert processor.distance_matrix.shape[0] == processor.distance_matrix.shape[1]

    def test_truck_specifications(self, processor):
        assert processor.get_truck_capacity('9.6') == 2000
        assert processor.get_route_cost(100, '9.6') == 100