from math import asin, cos, radians, sin, sqrt

R = 6371008.8


def dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * R * asin(min(1.0, sqrt(a)))


def decode_polyline(s: str, precision: int = 5) -> list:
    """Encoded polyline (as in Wialon order paths and OSRM) to [[lat, lon], ...]."""
    coords = []
    i = lat = lon = 0
    factor = 10**precision
    try:
        while i < len(s):
            for axis in (0, 1):
                shift = result = 0
                while True:
                    b = ord(s[i]) - 63
                    i += 1
                    result |= (b & 0x1F) << shift
                    shift += 5
                    if b < 0x20:
                        break
                delta = ~(result >> 1) if result & 1 else result >> 1
                if axis == 0:
                    lat += delta
                else:
                    lon += delta
            coords.append([lat / factor, lon / factor])
    except IndexError:
        pass
    return coords


def thin(points: list, limit: int) -> list:
    """At most `limit` points, keeping the first and last."""
    if len(points) <= limit:
        return points
    step = len(points) / (limit - 1)
    return [points[min(len(points) - 1, round(k * step))] for k in range(limit - 1)] + [points[-1]]


def valid_coord(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return abs(lat) <= 90 and abs(lon) <= 180 and not (lat == 0 and lon == 0)
