class Location:
    def __init__(self, city_name: str, lat: float = 0.0, lon: float = 0.0):
        self.city_name = city_name
        self.lat = lat
        self.lon = lon

    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return self.city_name == other.city_name

    def __str__(self):
        return f"{self.city_name} ({self.lat}, {self.lon})"