"""
CATEGORY 9(b): FUZZY INFERENCE SYSTEM
Use case: Smart manufacturing machine-temperature control.

Inputs:
    Machine temperature: 0 to 100 degrees
    Vibration level: 0 to 10

Output:
    Fan speed: 0 to 100 percent

The system uses triangular membership functions and a simple
Mamdani-style min-max inference process with centroid defuzzification.
No external libraries are required.

Run:
    python fuzzy_inference_system.py
"""


def triangular_membership(x, a, b, c):
    """Return membership degree for a triangular fuzzy set."""
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def temperature_memberships(temp):
    return {
        "low": triangular_membership(temp, 0, 0, 50),
        "medium": triangular_membership(temp, 25, 50, 75),
        "high": triangular_membership(temp, 50, 100, 100),
    }


def vibration_memberships(vibration):
    return {
        "low": triangular_membership(vibration, 0, 0, 5),
        "high": triangular_membership(vibration, 2, 10, 10),
    }


def fan_membership(speed, label):
    sets = {
        "slow": (0, 0, 50),
        "medium": (25, 50, 75),
        "fast": (50, 100, 100),
    }
    return triangular_membership(speed, *sets[label])


def infer_fan_speed(temp, vibration):
    """
    Rules:
      1. IF temperature is low AND vibration is low, fan is slow.
      2. IF temperature is medium, fan is medium.
      3. IF temperature is high, fan is fast.
      4. IF vibration is high, fan is fast.

    AND is implemented using min; rule aggregation uses max.
    """
    if not (0 <= temp <= 100):
        raise ValueError("Temperature must be between 0 and 100.")
    if not (0 <= vibration <= 10):
        raise ValueError("Vibration must be between 0 and 10.")

    t = temperature_memberships(temp)
    v = vibration_memberships(vibration)

    rules = [
        (min(t["low"], v["low"]), "slow"),
        (t["medium"], "medium"),
        (t["high"], "fast"),
        (v["high"], "fast"),
    ]

    # Aggregate the clipped output membership functions.
    output_points = []
    step = 1
    for speed in range(0, 101, step):
        membership = 0.0
        for strength, output_label in rules:
            clipped = min(strength, fan_membership(speed, output_label))
            membership = max(membership, clipped)
        output_points.append((speed, membership))

    # Centroid defuzzification.
    numerator = sum(x * degree for x, degree in output_points)
    denominator = sum(degree for _, degree in output_points)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def run_test_case(case_name, temperature, vibration):
    fan_speed = infer_fan_speed(temperature, vibration)
    print(f"\n{case_name}")
    print(f"Machine temperature: {temperature} C")
    print(f"Vibration level: {vibration}/10")
    print(f"Recommended fan speed: {fan_speed:.2f}%")

    if fan_speed < 35:
        print("Fan status: SLOW")
    elif fan_speed < 70:
        print("Fan status: MEDIUM")
    else:
        print("Fan status: FAST")


if __name__ == "__main__":
    # Test case 1: Cool machine and low vibration
    run_test_case("TEST CASE 1", temperature=20, vibration=1)

    # Test case 2: Moderate temperature and vibration
    run_test_case("TEST CASE 2", temperature=55, vibration=3)

    # Test case 3: High temperature and high vibration
    run_test_case("TEST CASE 3", temperature=90, vibration=8)
