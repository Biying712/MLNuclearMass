def micro_u_to_kev(mass_micro_u):
    # Conversion constants
    amu_to_kg = 1.66053906660e-27  # kg/amu
    speed_of_light = 2.99792458e8   # m/s
    joules_to_ev = 1.60218e-19      # J/eV

    # Convert micro-u to kg
    mass_kg = mass_micro_u * amu_to_kg * 1e-6

    # Calculate energy in Joules using E = mc^2
    energy_joules = mass_kg * speed_of_light**2

    # Convert energy to eV
    energy_ev = energy_joules / joules_to_ev

    # Convert energy to keV
    energy_kev = energy_ev / 1e3

    return energy_kev

# Example: Convert 1 micro-u to keV
mass_micro_u = 1  # 1 micro-u
energy_kev = micro_u_to_kev(mass_micro_u)
print(f"{mass_micro_u} micro-u is approximately {energy_kev:.2f} keV")
