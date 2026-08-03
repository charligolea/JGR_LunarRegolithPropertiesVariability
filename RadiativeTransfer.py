import numpy as np
from Regolith1D_JGR import pore_filling_fraction, porosity, get_Cr

c = 3e8

def compute_TB(T, w):
        
    TB = np.sum(w * T)

    return TB

def compute_TB_new(T, w, eps):

    # --- Brightness temperature ---
    T_CMB = 2.725 # K
    R0 = ((np.sqrt(eps[0]) - 1) / (np.sqrt(eps[0]) + 1))**2
    # TB = (1 - R0) * np.sum(w * T) + R0 * T_CMB
    TB = np.sum(w * T) + R0 * T_CMB

    return TB


import numpy as np

def compute_TB_polarized(T, w, eps, theta_deg = 0):
    """
    Compute separate H and V brightness temperatures including the CMB
    for an off-nadir observation angle.

    Parameters
    ----------
    T : array (N)
        Temperature profile
    w : array (N)
        Normalized internal weighting function (sums to 1.0)
    eps : float
        Dielectric constant
    theta_deg : float
        Observation incidence angle in degrees

    Returns
    -------
    TB_H : float
        Horizontal polarization brightness temperature
    TB_V : float
        Vertical polarization brightness temperature
    """
    # 1. Cosmic Microwave Background constant
    T_CMB = 2.725  

    # Convert observation angle to radians
    theta = np.radians(theta_deg)

    # 2. Compute the off-nadir Fresnel power reflectivity coefficients
    # This square root term dictates the refraction into the soil
    sqrt_term = np.sqrt(eps[0] - np.sin(theta)**2)
    cos_theta = np.cos(theta)

    # Horizontal power reflectivity
    R_H = np.abs((cos_theta - sqrt_term) / (cos_theta + sqrt_term))**2

    # Vertical power reflectivity
    R_V = np.abs((eps[0] * cos_theta - sqrt_term) / (eps[0] * cos_theta + sqrt_term))**2

    # 3. Calculate internal core regolith emission seen by the sensor
    T_internal_sum = np.sum(w * T)
    
    # Scale by the respective H and V emissivity factors
    T_emission_H = (1.0 - R_H) * T_internal_sum
    T_emission_V = (1.0 - R_V) * T_internal_sum

    # 4. Calculate the polarized CMB reflection
    T_reflected_CMB_H = R_H * T_CMB
    T_reflected_CMB_V = R_V * T_CMB

    # 5. Final combined brightness temperatures
    TB_H = T_emission_H + T_reflected_CMB_H
    TB_V = T_emission_V + T_reflected_CMB_V

    return TB_H, TB_V


def dielectric_constant_regolith(rho):
    return 1.92 ** (rho/1000)
    

def dielectric_constant_LLL(z, rho):
    
    """
    L - L - L model
    """

    eps_air = 1.0
    eps_ice = 3.09
    
    # Extracción de propiedades físicas del medio
    phi = porosity(rho)
    Cr = get_Cr(rhos=rho, z=z)
    PFF = pore_filling_fraction(rho, Cr, phi)
    eps_dry = dielectric_constant_regolith(rho)
    
    eps_eff = (eps_dry**(1/3) + phi*PFF*(eps_ice**(1/3)-eps_air**(1/3)))**3
    return eps_eff
    
    
def dielectric_constant_CS(z, rho):
    
    """
    Core-shell
    """
    
    # Maxwell-Garnett formula
    eps_air = 1.0
    eps_ice = 3.09
    
    # Step 1: From simplified MG formula (see dielectric_constant_GM)
    # We have a model for eps_dry. We want to know the permittivity of the solid regolith matrix.
    # We invert the case where eps_MG = eps_dry, eps_h = eps_regolith, eps_i = 1. we want to obtain eps_h
    # In this scenario f = phi
    phi = porosity(rho)
    Cr = get_Cr(rhos = rho, z=z)
    PFF = pore_filling_fraction(rho, Cr, phi)
    eps_dry = dielectric_constant_regolith(rho)
    
    A = 2*(1-phi)
    B = (1+2*phi) - eps_dry * (2+phi) 
    C = -eps_dry * (1-phi)
    
    eps_rego = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    
    # Step 2: Apply generalized formula for vacuum inclusions with spherical icy shell
    f_ice = phi * PFF
    f_air = phi * (1 - PFF)
    f = (f_ice + f_air)
    
    C1i = (eps_ice - eps_rego)
    C2i = (eps_ice + 2*eps_rego)
    C1a = (eps_air - eps_rego)
    C2a = (eps_air + 2*eps_rego)
    
    C3 = 3*eps_ice / (eps_air + 2*eps_ice)
    
    eps_eff = f * (f_ice*C1i + f_air * C3 * C1a) / (f_ice*C2i + f_air * C3 * C2a)
            
    return eps_eff

def dielectric_constant_GM(z, rho):
    """
    Maxwell-Garnett formula:
    
    eps_MG - eps_h / (eps_MG + 2 eps_h) = f (eps_i-eps_h)/(eps_i + 2 eps_h)
    
    or 
    
    eps_MG = eps_h * (1 + 2f * (eps_i-eps_h)/(eps_i+2eps_h)) / (1 - f * (eps_i-eps_h)/(eps_i+2eps_h))
    
    eps_MG: effective permittivity
    eps_h: permittivity of host medium
    eps_i: permittivity of inclusions (here vacuum = 1)
    f: volume fraction (the proportion of the total mixture made up of the particles)
    """

    eps_air = 1.0
    eps_ice = 3.09
    
    # Extracción de propiedades físicas del medio
    phi = porosity(rho)
    Cr = get_Cr(rhos=rho, z=z)
    PFF = pore_filling_fraction(rho, Cr, phi)
    eps_dry = dielectric_constant_regolith(rho)
    
    # Step 1: We have a model for eps_dry. We want to know the permittivity of the solid regolith matrix.
    # We invert the case where eps_MG = eps_dry, eps_h = eps_regolith, eps_i = 1. we want to obtain eps_h
    # In this scenario f = phi
    A = 2 * (1 - phi)
    B = (1 + 2 * phi) - eps_dry * (2 + phi) 
    C = -eps_dry * (1 - phi)
    
    eps_rego = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
    
    # Step 2: Calculate the permittivity of a pore, now filled with vacuum + ice
    # (1 - PFF) is the volume fraction here
    num_pore = eps_air + 2*eps_ice + 2*(1 -PFF)*(eps_air - eps_ice)
    den_pore = eps_air + 2*eps_ice - (1 - PFF)*(eps_air - eps_ice)
    eps_pore_eff = eps_ice * (num_pore / den_pore)
    
    # Step 3: Global permittivity
    num_eff = eps_pore_eff + 2*eps_rego + 2*phi*(eps_pore_eff - eps_rego)
    den_eff = eps_pore_eff + 2*eps_rego - phi*(eps_pore_eff - eps_rego)
    eps_eff = eps_rego * (num_eff / den_eff)
    
    return eps_eff

def dielectric_constant_regolith_icy_new(z, rho):
    
    """
    Volumetric Mixing Model. Developed by Carlos G. de Olea B. 2026
    """

    eps_rego = dielectric_constant_regolith(rho)

    eps_vac = 1.0
    eps_ice = 3.09

    phi = porosity(rho)
    Cr = get_Cr(rhos=rho, z=z)
    PFF = pore_filling_fraction(rho, Cr, phi)

    # effective pore permittivity
    eps_pore = (1 - PFF)*eps_vac + PFF*eps_ice

    # perturbation relative to vacuum pores
    delta_eps = eps_pore - eps_vac

    # scale by porosity (only pores can change)
    eps_eff = eps_rego + phi * delta_eps

    return eps_eff


def compute_Fresnel_reflection(N, eps):
    
    R = np.zeros(N)
    for i in range(N-1):
        R[i] = (np.sqrt(eps[i+1]) - np.sqrt(eps[i])) / (np.sqrt(eps[i+1]) + np.sqrt(eps[i]))
    R[-1] = 0
    
    return R
    
    
def compute_weighting_function(z, rho, freq_GHz, eps, loss_tangent_type = "Carrier_1991", S_TiO2 = 14):

    """
    Compute brightness temperature from 1D thermal model.

    Parameters
    ----------
    T : array (N)
        Temperature profile
    z : array (N)
        Depth grid (non-uniform allowed)
    rho : array (N)
        Density profile
    freq_GHz : float
        Frequency in GHz

    Returns
    -------
    TB : float
    w : array (N)
        Weighting function
    """
    
    N = len(z)

    # --- Compute dz (non-uniform grid) ---
    dz = np.zeros_like(z)
    dz[1:-1] = (z[2:] - z[:-2]) / 2
    dz[0] = z[1] - z[0]
    dz[-1] = z[-1] - z[-2]

    # --- Constants ---
    c = 3e8
    f = freq_GHz * 1e9

    # --- Dielectric constant ---
    # eps = dielectric_constant_regolith(rho)

    # --- Loss tangent ---
    tan_delta = get_loss_tangent(rho, freq_GHz, S_TiO2 = S_TiO2, loss_tangent_type = loss_tangent_type)

    # --- Absorption coefficient ---
    k = compute_absorption_coefficient(f, eps, tan_delta)

    # --- Reflection coefficients ---
    R = compute_Fresnel_reflection(N, eps)

    # --- Transmission per layer ---
    tau = np.exp(-k * dz)

    # --- Weighting function ---
    w = np.zeros(N)
    T_cum = 1.0

    for i in range(N):
        w[i] = (1 - tau[i]) * (1 + abs(R[i])**2 * tau[i]) * T_cum

        if i < N-1:
            T_cum *= (1 - abs(R[i])**2) * tau[i]
            
    return w


def compute_absorption_coefficient(freq, eps, tan_delta):
    """
    Absorption coefficient from complex dielectric constant.
    k here represents attenuation of coherent wave.
    """
    return 2 * (2 * np.pi * freq / c) * np.sqrt(eps / 2 * (np.sqrt(1 + tan_delta**2) - 1))


def get_loss_tangent(rho, freq_GHz, S_TiO2 = 5, loss_tangent_type = "Carrier_1991"):
    if loss_tangent_type == "Feng_2021_highland":
        return loss_tangent_feng_2021_highland(rho, freq_GHz)
    elif loss_tangent_type == "Feng_2021_mare":
        return loss_tangent_feng_2021_mare(rho, freq_GHz, S_TiO2)
    elif loss_tangent_type == "Carrier_1991":
        return loss_tangent_carrier_1991(rho, S_TiO2 = S_TiO2)
    elif loss_tangent_type == "Siegler_2020":
        return loss_tangent_siegler_2020(rho, S_TiO2 = S_TiO2)
    elif loss_tangent_type == "Feng_2020":
        return loss_tangent_feng_2020(rho, freq_GHz)
    elif loss_tangent_type == "Carlos":
        return loss_tangent_carlos(rho, freq_GHz, S_TiO2 = S_TiO2)

def loss_tangent_carrier_1991(rho, S_TiO2=5):
    return 10**(0.312*(rho/1000) + 0.038*S_TiO2 - 3.26)

def loss_tangent_feng_2020(rho, freq_GHz):
    return 10**(0.312*(rho/1000) + 0.0043*freq_GHz - 2.64)

def loss_tangent_siegler_2020(rho, S_TiO2=5):
    return 10**((0.0272+0.2967)*(rho/1000)+ 0.0027*S_TiO2 - 3.058)

def loss_tangent_feng_2021_highland(rho, freq_GHz):
    return 10**(0.312*(rho/1000) + freq_GHz**0.069 - 3.79)

def loss_tangent_feng_2021_mare(rho, freq_GHz, S_TiO2):
    return 10**(0.312*(rho/1000) + S_TiO2 * (freq_GHz**(-0.0025)-0.958) - 2.65)

def loss_tangent_carlos(rho, freq_GHz, S_TiO2=5):
    a = 0.036684
    b = 0.019186
    c = 4.044818
    return 10**(0.312*(rho/1000) + b*S_TiO2 + freq_GHz**a - c)