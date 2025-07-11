# thresholds.py
THRESHOLDS = {
    "LPG(ppm)":      (200, 1000),
    "Propane(ppm)":  (200, 1000),
    "Methane(ppm)":  (300, 1200),
    "Smoke(ppm)":    (100, 300),
    "Ammonia(ppm)":  (25,  200),
    "Benzene(ppm)":  (5,   50),
}

def gas_status(gas_name, value):
    low, high = THRESHOLDS[gas_name]
    if value >= high:
        return "High"
    if value >= low:
        return "Moderate"
    return "Low"