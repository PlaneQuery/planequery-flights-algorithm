import math
import subprocess


def haversine(lat1, lon1, lat2, lon2):
    """Return the great-circle distance between two coordinates in km."""
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_deg(angle: float) -> float:
    return angle % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Return the smallest absolute difference between two headings."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the initial bearing from point 1 to point 2."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    )

    return normalize_deg(math.degrees(math.atan2(x, y)))


def current_commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
