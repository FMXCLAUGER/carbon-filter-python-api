"""Physical constants and default values"""

# Gas constant
R = 8.314  # J/(mol·K)

# Standard conditions
T_STANDARD = 273.15  # K (0°C)
P_STANDARD = 101325  # Pa

# Air properties at 25°C
AIR_DENSITY_25C = 1.184  # kg/m³
AIR_VISCOSITY_25C = 1.849e-5  # Pa·s

# Molecular diffusivity in air (typical VOC)
D_AIR_TYPICAL = 8e-6  # m²/s

# Adsorption heat (typical for VOCs)
DELTA_H_ADS_TYPICAL = -40000  # J/mol (exothermic)

# Specific heat
CP_AIR = 1005  # J/(kg·K)
CP_CARBON = 840  # J/(kg·K)

# Water vapor pressure at different temperatures (Pa)
WATER_VAPOR_PRESSURE = {
    0: 611,
    10: 1228,
    20: 2338,
    25: 3169,
    30: 4243,
    40: 7381,
    50: 12344,
    60: 19932,
}

# Typical velocity range
VELOCITY_MIN = 0.05  # m/s
VELOCITY_MAX = 0.6  # m/s
VELOCITY_TYPICAL = 0.25  # m/s

# Typical EBCT range
EBCT_MIN = 0.5  # s
EBCT_MAX = 20  # s
EBCT_TYPICAL = 5  # s

# Temperature limits for safety
TEMP_WARNING = 60  # °C - Warning threshold
TEMP_CRITICAL = 100  # °C - Critical threshold (hotspot risk)

# Humidity threshold for water competition
HUMIDITY_THRESHOLD = 40  # % RH - Below this, humidity effect is minimal
