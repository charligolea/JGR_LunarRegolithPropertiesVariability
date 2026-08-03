import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class Grid1D:
    """
    Create a 1D primal grid over [0, L] with exponential stretching.
    """

    def __init__(self, N, z_max = 1.0, Hx = 0.2, dz0 = 0.002, dz_max = 0.1):
        self.N = N
        self.z_max = z_max
        self.Hx = Hx # scale height (m)
        self.dz0 = dz0 # initial spacing (m)
        self.dz_max = dz_max # max spacing (m)

    def create_grid_exp1(self):
        # Initialize arrays
        self.z = np.ones(self.N+1)
        self.dz = np.ones(self.N)

        # Build grid iteratively
        for i in range(self.N):
            self.dz[i] = min(self.dz0 * np.exp(self.z[i] / self.Hx), self.dz_max)
            self.z[i+1] = self.z[i] + self.dz[i]
        self.dz = self.z[1:] - self.z[:-1]
    
        return self.z, self.dz
    
    def create_grid_exp2(self, alpha = 2):
        # Initialize arrays
        self.z = np.ones(self.N+1)
        self.dz = np.ones(self.N)
        eps = np.ones(self.N+1)

        # Build grid iteratively
        for i in range(self.N+1):
            eps[i] = i / (self.N)
            self.z[i] = self.z_max * (np.exp(alpha*eps[i])-1) / (np.exp(alpha)-1)
                    
        for i in range(self.N):
            self.dz[i] = self.z[i+1] - self.z[i]
    
        return self.z, self.dz
    
    def create_grid_Hayne(self, m=10, n=20):
        """
        Create vertical grid following Hayne et al. scheme based on thermal skin depth.

        Parameters
        ----------
        zs : float
            Thermal skin depth.
        m : float
            Controls top layer thickness (Δz0 = zs / m).
        n : float
            Controls geometric growth rate.
        depth_factor : float
            Total depth as multiple of skin depth (~10 recommended).

        Returns
        -------
        z : ndarray
            Grid node depths (size N+1)
        dz : ndarray
            Layer thicknesses (size N)
        """

        # Per = 29.53*24*3600
        # kappa =  Ks / (rho_s*cp_s)
        # zs = np.sqrt(kappa * Per /np.pi)
        
        zs = 0.05
        
        # Total domain depth
        depth_factor = self.z_max / zs
        z_max = depth_factor * zs

        # Initialize arrays
        z = np.ones(self.N + 1)
        dz = np.ones(self.N)

        # --- First layer thickness
        dz[0] = zs / m

        # --- Build geometrically increasing layers
        for i in range(1, self.N):
            dz[i] = dz[i-1] * (1 + 1/n)

        # --- Normalize to match desired total depth
        """total_depth = np.sum(dz)
        dz *= z_max / total_depth"""

        # --- Construct depth grid
        z[1:] = np.cumsum(dz)

        return z, dz

class Regolith1D:
    """
    """

    def __init__(self, N=100, lmbda=0, z = [], dz = [], 
                 t_max = 86400, n_steps = 86400, 
                 H_par = 0.068, rho_s = 1100, rho_d = 1800,
                 epsilon = 0.95, s = 0.0, theta = 0.0, wt = 0, z0_ice = 0, zend_ice = 0,
                 albedo_model = "Hayne_2017", cond_mode = "VM",
                 prob_type = "Moon"):
                
        self.lam = lmbda
        self.lambda_rad = np.radians(self.lam)
        
        self.N = N
        self.dz = dz
        self.z = z
        
        self.t_max = t_max
        self.n_steps = n_steps
        self.dt = t_max / (n_steps)
        
        self.time = np.linspace(0, t_max, n_steps+1)
        self.conductivity_mode = cond_mode
        self.H_par = H_par
        self.rho_s = rho_s
        self.rho_d = rho_d
        self.Kd = 3.4e-3     # W/m/K (deep)
        self.Ks = 7.4e-4
        self.Chi_phonon = 2.7
        
        # CONSTANTS
        self.epsilon = epsilon       # emissivity OF LUNAR SURFACE
        self.S0 = 1361  # Solar irradiation (W/m²)
        self.sigma = 5.670374419e-8 # W/m²K4
        self.Rau = 1.017
        
        # Water ice initialization
        self.wt = wt # Water content percentage
        self.z0_ice = z0_ice # Start of ice layer
        self.zend_ice = zend_ice # End of ice layer

        # Geometry (example values)
        self.theta = np.radians(theta)      # slope angle
        self.s = s              # slope aspect
        # self.declination = 0
        
        #Solution
        
        if prob_type == "Moon":
            
            self.J0 = - 0.018              # geothermal flux (W/m²)
            
            if albedo_model == "Yu_2025_Highland":
                self.albedo = albedo_yu_2025_highland
            elif albedo_model == "Yu_2025_Mare":
                self.albedo = albedo_yu_2025_mare
            elif albedo_model == "Vasavada_2012_MAre":
                self.albedo = albedo_vasavada_mare_2012
            elif albedo_model == "Vasavada_2012_Highland":
                self.albedo = albedo_vasavada_highland_2012
            elif albedo_model == "Feng_2020_Mare":
                self.albedo = albedo_feng_2020_mare
            elif albedo_model == "Feng_2020_Highland":
                self.albedo = albedo_feng_2020_highland
            else:
                self.albedo = albedo_hayne_2017
            
            self.surface_BC = self.get_surface_BC_moon
            self.deep_BC = self.get_deep_BC_neumann
            
            self.rhos = rho_effective_carlos_new2(self.z, H = self.H_par, wt_max = self.wt, z_start = self.z0_ice, z_end = self.zend_ice)
            self.porosity = porosity(self.rhos)
            self.Cr = get_Cr(self.rhos, self.z, H = self.H_par, rho_s = self.rho_s, rho_d = self.rho_d)
            self.PFF = pore_filling_fraction(self.z, self.Cr, self.porosity, H = self.H_par, rho_s = self.rho_s, rho_d = self.rho_d)
                
            self.properties = self.get_regolith_properties_moon
            
            self.diviner_T = lunar_temperature_diviner(n_steps = 86400, time = np.linspace(0, 86400, 86400+1), lambda_rad = self.lambda_rad)
            self.T = initial_profile_3(self, albedo = self.albedo) 
            self.T_iter = self.T.copy()
            
            
        elif prob_type == "analytical_constant_dirichlet":
            self.J0 = 0
            self.Ts = 0
            self.cpi = 0.008
            self.Ki = 1e-3
            self.rhos = np.ones(self.N+1) * 1300
            self.properties = self.get_regolith_properties_constant
            
            self.surface_BC = self.get_surface_BC_dirichlet
            self.deep_BC = self.get_deep_BC_dirichlet
            # self.deep_BC_factor = 0
            # self.surface_BC_factor = 0
            
            self.T = np.sin(np.pi * self.z / self.z[-1])
            self.T[0], self.T[-1] = 0, 0
            
        elif prob_type == "analytical_constant_neumann":
            self.J0 = 0
            self.cpi = 0.008
            self.Ki = 1e-3
            self.rhos = np.ones(self.N+1) * 1300
            self.properties = self.get_regolith_properties_constant
            
            self.surface_BC = self.get_surface_BC_neumann
            self.deep_BC = self.get_deep_BC_neumann
            # self.deep_BC_factor = 1
            # self.surface_BC_factor = 0
            
            L = self.z[-1]
            f = np.sin(np.pi * self.z / L)

            A0 = np.trapz(f, self.z) / L
            T = A0 * np.ones_like(self.z)

            for n in range(1, N):  # N modos
                An = 2/L * np.trapz(f * np.cos(n*np.pi*self.z/L), self.z)
                T += An * np.cos(n*np.pi*self.z/L)

            self.T = T
                    
        
    def get_regolith_properties_moon(self):
        cps = cp_effective_carlos(self.T_iter, rhos = self.rhos, phi = self.porosity, PFF = self.PFF)
        
        if self.conductivity_mode == "VM": # Volumetric Mixing
            # Ks = k_eff(self.T_iter, self.rhos, phi = self.porosity, PFF = self.PFF, rho_s = self.rho_s, rho_d = self.rho_d, Kd = self.Kd, Ks = self.Ks, Chi = self.Chi_phonon)
            Ks = k_eff(self.T_iter, self.z, phi = self.porosity, PFF = self.PFF, H = self.H_par, Kd = self.Kd, Ks = self.Ks, Chi = self.Chi_phonon)
        elif self.conductivity_mode == "GB": # Grain Boundary
            # Ks = k_eff_Mellon_1997_new(self.T_iter, self.rhos, phi = self.porosity, PFF = self.PFF, rho_s = self.rho_s, rho_d = self.rho_d, Kd = self.Kd, Ks = self.Ks, Chi = self.Chi_phonon)
            Ks = k_eff_Mellon_1997_new(self.T_iter, self.z, phi = self.porosity, PFF = self.PFF, H = self.H_par, Kd = self.Kd, Ks = self.Ks, Chi = self.Chi_phonon)
        # Ks = k_eff_Mellon_1997(self.T_iter, self.rhos, model = self.model, phi = self.porosity, PFF = self.PFF)
        
        return cps, Ks
    
    def get_regolith_properties_constant(self):
        Ks = np.ones(self.N+1) * self.Ki
        cps = np.ones(self.N+1) * self.cpi
        return cps, Ks
    
    
    def get_surface_BC_moon(self, t_step = 0, **kwargs):
                    
        rho0 = self.rhos[0]
        
        Kco = self.Ks
        Bo  = Kco * self.Chi_phonon / (350**3)
        
        T0 = self.T_iter[0]  # initial guess
        T1 = self.T_iter[1]
        T2 = self.T_iter[2]
        dz0 = self.dz[0]
        
        ci = cos_iota(t_step, self.lambda_rad, self.theta, self.s)
        ch = np.cos(hour_angle((t_step / 3600) % 24))
        
        S_rad = (self.S0/(self.Rau**2)) * (1 - self.albedo(np.arccos(ci))) * ci if ci > 0 else 0.0
        # S_rad = (self.S0/(self.Rau**2)) * (1 - self.albedo(np.arccos(ci))) * ci if ch > 0 else 0.0

        # if not (6 <= (t_step / 3600) % 24 <= 18) and S_rad > 0:
            # print(f"[WARNING] Solar input at night! t = {(t_step / 3600) % 24:.2f} h, S_rad = {S_rad:.3e}")
        
        max_Newton = 50
        
        for _ in range(max_Newton):  # Newton iterations

            # Conductivity
            K0 = Kco + Bo/2 * (T0**3+T1**3)
            
            # Gradient
            dTdz = (-3*T0 + 4*T1 - T2) / (2 * dz0)
            
            # Function
            f = self.epsilon * self.sigma * T0**4 - K0 * dTdz - S_rad # - self.J0  J0 is negative in our coordinate system
            
            # Derivatives
            dK_dT0 = 3 * Bo * T0**2
            ddTdz_dT0 = -3 / (2 * dz0)
            
            f_prime = (4 * self.epsilon * self.sigma * T0**3 - dK_dT0 * dTdz - K0 * ddTdz_dT0)
            
            # Newton update
            deltaT = -f / f_prime
            T0 += deltaT
            
            # Convergence check
            if abs(deltaT) < 1e-3:  # ~1 mK tolerance
                break
                
        return T0
    
    def get_surface_BC_dirichlet(self, **kwargs):
        return self.Ts
    
    def get_surface_BC_neumann(self, Ksur = 0, q = 0, **kwargs):
        dz_0 = self.dz[0]   
        # return q * dz_0 / Ksur
        Tsur = (4*self.T_iter[1]-self.T_iter[2])/3 + (2*dz_0/(3*Ksur)) * q
        return Tsur
    
    def get_deep_BC_neumann(self, Kdeep = 0, q = 0, **kwargs):
        dz_N = self.dz[self.N-1]    
        # return q * dz_0 / Ksur
        Tdeep = (4*self.T_iter[self.N-1]-self.T_iter[self.N-2])/3 - (2*dz_N/(3*Kdeep)) * q
        # Tdeep = self.T_iter[self.N-1] - (2*dz_N/(3*Kdeep)) * q
        return Tdeep
    
    
    def get_deep_BC_dirichlet(self, **kwargs):
        return self.Ts
     
    def calculate_temperature_profile_explicit(self):
        T_history = np.ones((self.n_steps+1, self.N+1))
        T_history[0, :] = self.T.copy()

        tol = 1e-4
        max_iter = 50
        
        # T_div = self.lunar_temperature_diviner()

        for t_idx, t_step  in enumerate(self.time[1:]):
            self.T_iter = self.T.copy()
            

            # ---------------------------------------------------------
            # BUCLE DE ITERACIÓN NO LINEAL (Punto Fijo / Picard)
            # ---------------------------------------------------------
            for _ in range(max_iter):
                T_new = self.T_iter.copy()
                
                # 1. ACTUALIZAR PARÁMETROS CON LA ESTIMACIÓN ACTUAL T_iter
                cps, Ks = self.properties()
                
                # 2. SURFACE BOUNDARY CONDITION
                # self.T_iter = T_new.copy()
                T_new[0] = self.surface_BC(t_step=t_step, q = self.J0, Ksur = (Ks[0]+Ks[1])/2)

                # 3. INTERNAL NODES
                for i in range(1, self.N):
                    dz_i = self.dz[i-1]
                    dz_ip1 = self.dz[i]
                    
                    # Coeficientes actualizados con T_iter
                    a = (Ks[i-1] + Ks[i]) / dz_i * self.dt / (self.rhos[i] * cps[i] * (dz_i + dz_ip1))
                    b = (Ks[i+1] + Ks[i]) / dz_ip1 * self.dt / (self.rhos[i] * cps[i] * (dz_i + dz_ip1))
                    d = 1 - (a + b)
                    
                    # T_new[i] = a * T_old[i-1] + d * T_old[i] + b * T_old[i+1]
                    T_new[i] = a * self.T_iter[i-1] + d * self.T_iter[i] + b * self.T_iter[i+1]
                
                # 4. NODO INFERIOR (i = N) - Flujo de calor geotérmico J0
                T_new[self.N] = self.deep_BC(q = self.J0, Kdeep = (Ks[self.N]+Ks[self.N-1])/2)             

                # 5. CONTROL DE CONVERGENCIAlam
                error = np.max(np.abs(T_new - self.T_iter))
                if error < tol:
                    # print("Convergence")
                    break

                self.T_iter = T_new.copy()

            # Finalizado el bucle de iteración, guardamos el paso de tiempo
            self.T = T_new.copy()
            T_history[t_idx+1, :] = self.T
            
            progress = t_idx / self.n_steps
            percent = progress * 100

            if percent % 25 == 0:
                print(f"{percent}% time steps computed")
            
        return T_history


## ======================================================
## 1. REGOLITH PROPERTIES
## ======================================================
    
## ================================   
#### 1.1 POROSITY INFORMATION  ####
## ================================ 

def porosity(rhos, rho_basalt = 3000):
    return 1 - rhos/rho_basalt

def get_Cr(rhos, z, H=0.068, rho_s = 1100, rho_d = 1800):
    z = np.asarray(z)
    H = np.asarray(H)
    rho_s = np.asarray(rho_s)
    rho_d = np.asarray(rho_d)
    Cr = rho_Hayne_2017(z, H=H, rho_s = rho_s, rho_d = rho_d) / rhos
    return Cr
    
def pore_filling_fraction(z, Cr, phi, rho_ice=917.0, H=0.068, rho_s = 1100, rho_d = 1800):
    z = np.asarray(z)
    H = np.asarray(H)
    rho_s = np.asarray(rho_s)
    rho_d = np.asarray(rho_d)
    
    rho_dry = rho_Hayne_2017(z, H=H, rho_s = rho_s, rho_d = rho_d)
    F = (rho_dry / (phi * rho_ice)) * (1.0 / Cr - 1.0)
    return np.clip(F, 0.0, 1.0)



## ================================
#### 1.2. DENSITY              ####
## ================================

def rho(z):
    # z must be in m
    # if z <= 0.02:
        # return 1363.2
    # else:
    return 1920 * ((z*100) + 12.2) / ((z*100) + 18)

def rho_Hayne_2017(z, H = 0.068, rho_s = 1100, rho_d = 1800):
    return rho_d - (rho_d - rho_s) * np.exp(-z/H)
    
def rho_effective_carlos(z, H = 0.068, wt = 0, rho_s = 1100, rho_d = 1800):
    """
    - wt: % water ice massic content
    """
    rho_dry = np.array([rho_Hayne_2017(zi, H=H, rho_s = rho_s, rho_d = rho_d) for zi in z])  
    return rho_dry / (1-wt/100)

def rho_effective_carlos_new(z, H = 0.068, z0 = 0, wt_max=0, dz=0.1, rho_s = 1100, rho_d = 1800):
    """
    z      : depth array
    H      : scale factor
    z0     : depth where ice starts
    wt_max : max mass fraction (%) at depth
    dz     : transition smoothness (smaller = sharper)
    """
    rho_dry = np.array([rho_Hayne_2017(zi, H=H, rho_s = rho_s, rho_d = rho_d) for zi in z])
    
    # Smooth transition
    wt = wt_max / (1 + np.exp(-(z - z0)/dz))
    
    return rho_dry / (1 - wt/100)

def _as_array(x, shape):
    """Broadcast scalar or array to target shape."""
    x = np.asarray(x)
    if x.ndim == 0:
        return np.full(shape, x, dtype=float)
    return x

def rho_effective_carlos_new2(z, H=0.068, z_start=0, z_end=None, wt_max=0, dz=0.01, rho_s=1100, rho_d=1800):
    """
    Works with scalars OR arrays for all parameters.
    
    z        : depth (scalar or array)
    H        : scale factor
    z_start  : depth where ice starts
    z_end    : depth where ice ends
    wt_max   : max mass fraction (%)
    dz       : transition smoothness
    """

    # Handle z_end default safely
    if z_end is None:
        z_end = 10 * z
    z_end = np.asarray(z_end)

    # --- DRY DENSITY (vectorized) ---
    rho_dry = rho_d - (rho_d - rho_s) * np.exp(-z / H)

    # --- ICE WINDOW (fully vectorized) ---
    on  = 1 / (1 + np.exp(-(z - z_start)/dz))
    off = 1 / (1 + np.exp(-(z - z_end)/dz))

    wt = wt_max * (on - off)

    # --- APPLY ONLY WHERE wt_max != 0 ---
    rho = np.where(
        wt_max != 0,
        rho_dry / (1 - wt/100),
        rho_dry
    )

    return rho


## ================================
#### 1.3. HEAT CAPACITY        ####
## ================================

### DRY MODELS ###
def cp_dry_Hayne_2017(T):
    return (-3.6125
            + 2.7431*T
            + 2.3616e-3*T**2
            - 1.2340e-5*T**3
            + 8.9093e-9*T**4)


### ICY MODELS ###
def cp_ice_Lide_2003(T):
    return (7.49*T + 90)

def cp_ice_Shulman_2004(T):
    return  1000*(7.73e-3*T*(1-np.exp(-1.263e-3*T**2))) * ( 1 + np.exp(-3*np.sqrt(T)) * 8.47e-3*T**6 + 2.0825e-7*T**4 * np.exp(-4.97e-2*T))


### EFFECTIVE HEAT CAPACITY ###
def cp_effective_carlos(T, rhos = None, phi = None, PFF = None, rho_ice = 917):
    # Based on Siegler (2011)
    cp_dry = cp_dry_Hayne_2017(T)
    cp_ice = cp_ice_Shulman_2004(T)
    return (PFF * phi * cp_ice * rho_ice) / rhos + cp_dry
    


## ================================
#### 1.4. CONDUCTIVITY         ####
## ================================

### ICY MODELS ###

# def k_eff(T, rhos = None, phi = None, PFF = None, rho_s = 1100, rho_d = 1800, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
def k_eff(T, z, phi = None, PFF = None, H = 0.068, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
    # VOLUMETRIC MIXING
    # k_dry = np.array([K_Hayne_2017(T[i], rhos[i], rho_s = rho_s, rho_d = rho_d, Kd = Kd, Ks = Ks, Chi = Chi) for i in range(len(T))])
    k_dry = K_Hayne_2017(T, z, H = H, Kd = Kd, Ks = Ks, Chi = Chi)
    k_ice = k_ice_Hobbs_1974(T)
    return k_dry + k_ice * PFF * phi


# def k_eff_Mellon_1997_new(T, rhos = None , phi = None, PFF = None, rho_s = 1100, rho_d = 1800, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
def k_eff_Mellon_1997_new(T, z, phi = None, PFF = None, H = 0.068, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7, k_solid = 3.0):
    """
    Calculates the effective thermal conductivity of icy regolith 
    using the sintering model by Mellon et al. (1997).
    
    Parameters:
    T       : Array or float of Temperature [K]
    rhos    : Regolith density (required for your K_Hayne_2017 function)
    phi     : Initial porosity (eps_0) [Fraction between 0 and 1]
    PFF     : Pore filling fraction of ice (F) [Fraction between 0 and 1]
    k_solid : Thermal conductivity of the solid grain [W/m-K]. Default is 1.2 (Glass).
    """

    # eps = PFF * phi
        
    # 1. Get k_dry for each point using your existing function
    # k_dry = np.array([K_Hayne_2017(T[i], rhos[i], rho_s = rho_s, rho_d = rho_d, Kd = Kd, Ks = Ks, Chi = Chi) for i in range(len(T))])
    k_dry = K_Hayne_2017(T, z, H = H, Kd = Kd, Ks = Ks, Chi = Chi)
    # print(k_dry.shape)
    
    # 2. Invert Equation 23 to find the empty pore conduction (kpo)
    kpo = (k_dry * phi * k_solid) / (k_solid - k_dry * (1 - phi))
    
    # 3. Calculate pure ice thermal conductivity at temperature T
    k_ice = k_ice_Hobbs_1974(T)
    # print(k_ice.shape)
    
    # 4. Fractional area factor added by the ice bridges (fA = sqrt(F))
    fA = np.sqrt(PFF)
    
    # 5. Calculate combined conduction across the modified pore (Equation 22)
    kp = (1 - fA) * kpo + k_ice * fA
    
    # 6. Calculate the final bulk effective thermal conductivity (Equation 23)
    k_final = (k_solid * kp) / ((1 - phi) * kp + phi * k_solid)
    
    return k_final

    
def k_ice_Hobbs_1974(T):
    return 488.19 / T + 0.4685


### DRY MODELS ###

# def K_phonon_cond(rho, rho_d = 1800, rho_s = 1100, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
def K_phonon_cond(z, H, Kd, Ks, Chi):

    # Ensure numpy arrays (prevents list / dtype issues)
    # rho = np.asarray(rho, dtype=float)

    # Conductivity at given density
    # Kc = Kd - (Kd - Ks) * (rho_d - rho) / (rho_d - rho_s)
    Kc = Kd - (Kd - Ks) * np.exp(-z/H)
    
    B_ph = Kc * Chi / (350**3)
    
    return Kc, B_ph
    
    
# def K_Hayne_2017(Temp, rho_Hayne, rho_s = 1100, rho_d = 1800, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
def K_Hayne_2017(Temp, z, H = 0.068, Kd = 3.4e-3, Ks = 7.4e-4, Chi = 2.7):
    """
    Thermal conductivity model from Hayne et al. (2017)

    Parameters
    ----------
    T : array_like or float
        Temperature [K]
    rho_Hayne : array_like or float
        Density [kg/m^3]

    Returns
    -------
    K : ndarray or float
        Thermal conductivity [W/m/K]
    """

    # Ensure numpy arrays (prevents list / dtype issues)
    Temp = np.asarray(Temp, dtype=float)
    
    # Kc, B_ph = K_phonon_cond(rho_Hayne, rho_s = rho_s, rho_d = rho_d, Kd = Kd, Ks = Ks, Chi = Chi)
    Kc, B_ph = K_phonon_cond(z, H, Kd, Ks, Chi)

    return Kc + B_ph*Temp**3


## ======================================================
## 2. SOLAR GEOMETRY
## ======================================================
    
## ================================   
#### 2.1 COORDINATES           ####
## ================================ 

def hour_angle(tau):
    return (tau/12 - 1) *np.pi

def func_zeta(lam,h, dec):
    # return np.cos(lam) * np.cos(h)
    return np.sin(lam)*np.sin(dec) + np.cos(lam)*np.cos(dec)*np.cos(h)

def solar_declination(t):
    """
    t: simulation time in seconds (1 lunar day = 86400)
    """

    T_lunar_day = 86400
    T_year = 12.37 * T_lunar_day  # ~1 year in your units

    eps = np.deg2rad(1.54)

    return eps * np.sin(2 * np.pi * t / T_year)

def cos_iota(t, lam, theta, s):
    # Local time τ in hours (convert from seconds)
    tau = (t / 3600) % 24
    
    # Hour angle
    h = hour_angle(tau)
    
    # Zenith angle
    dec  = solar_declination(t)
    cos_zeta = func_zeta(lam,h, dec)
    zeta = np.arccos(cos_zeta)
    
    # Azimuth
    
    if lam == 0 and np.isclose(h, 0):
        a = 0
    else:
        cos_a = -np.sin(lam)*np.cos(h)/np.sin(zeta)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        # sin_a = np.sin(h) / np.sin(zeta)
        # a = np.arctan2(sin_a, cos_a)
        a = np.arccos(cos_a)
    
    # Incidence angle
    cos_i = np.cos(theta)*np.cos(zeta) + np.sin(theta)*np.sin(zeta)*np.cos(s - a)
    
    # cos⁺
    return cos_i


## ================================   
#### 2.2 ALBEDO MODELS         ####
## ================================ 

def albedo_hayne_2017(iota):
    A0 = 0.12  # Hayne et al. 2017
    a, b = 0.06, 0.25
    return A0 + a*(iota/(np.pi/4))**3 + b*(iota/(np.pi/2))**8

def albedo_yu_2025_highland(iota):
    A0 = 0.09  # Yu 2025 (highland)
    a, b = 0.095, 0.06
    return A0 + a*(iota/(np.pi/4))**3 + b*(iota/(np.pi/2))**8

def albedo_yu_2025_mare(iota):
    A0 = 0.05  # Yu 2025 (mare)
    a, b = 0.095, 0.06
    return A0 + a*(iota/(np.pi/4))**3 + b*(iota/(np.pi/2))**8

def albedo_vasavada_mare_2012(iota):
    A0 = 0.07  # Yu 2025 (mare)
    a, b = 0.14, 0.045
    return A0 + a*(iota/(np.pi/4))**3 + b*(iota/(np.pi/2))**8

def albedo_vasavada_highland_2012(iota):
    A0 = 0.16  # Yu 2025 (mare)
    a, b = 0.14, 0.045
    return A0 + a*(iota/(np.pi/4))**3 + b*(iota/(np.pi/2))**8

def albedo_feng_2020_mare(iota):
    A0 = 0.07
    return A0 + (0.6126 -1.597*np.cos(iota) + 2.15*np.cos(iota)**2 - 1.636*np.cos(iota)**3 + 0.4704*np.cos(iota)**4)

def albedo_feng_2020_highland(iota):
    A0 = 0.12
    return A0 + (0.6126 -1.597*np.cos(iota) + 2.15*np.cos(iota)**2 - 1.636*np.cos(iota)**3 + 0.4704*np.cos(iota)**4)


## ======================================================
## 3. INITIAL TEMPERATURE PROFILES
## ======================================================

def initial_profile(z, T_surface, T_deep=250, delta=0.05):
    return T_deep + (T_surface - T_deep) * np.exp(-z / delta)

def initial_profile_2(z, T_surface, H=0.068):
    T0 = T_surface
    TN = T0 / np.sqrt(2)
    return TN - (TN - T0) * np.exp(-z / H)

def initial_profile_3(Rego = Regolith1D, H=0.068, albedo = albedo_hayne_2017):
    # T0 = T_surface
    # Equilibirum temperature at noon
    ci = cos_iota(86400/2, Rego.lambda_rad, Rego.theta, Rego.s) 
    ch = np.cos(hour_angle(tau = 6))
    S_rad = Rego.S0 * (1 - albedo(np.arccos(ci))) * ci if ch >= 0 else 0.0
    T0_noon = (S_rad / (Rego.epsilon * Rego.sigma))**(1/4)
    
    T0 = lunar_temperature_diviner(Rego.n_steps, Rego.time, Rego.lambda_rad)[0]
    
    TN = T0_noon / np.sqrt(2)
    # T0 = 100
    
    return TN - (TN - T0) * np.exp(-Rego.z / H)
    # return TN - (TN - T0_noon) * np.exp(-Rego.z / H)


def lunar_temperature_diviner(n_steps, time, lambda_rad):
    """
    Returns Diviner-based lunar surface temperature over one day
    for a given latitude.

    Parameters:
        latitude_deg (float): latitude in degrees (-90 to 90)
        n_steps (int): number of time steps
        t_max (float): duration of one day in seconds (default 86400)

    Returns:
        T (array): temperature in Kelvin
    """        
    
    # Nightside polynomial coefficients
    a = np.array([444.738, -448.937, 239.668, -63.8844, 8.34064, -0.423502])
    
    # Output temperature array
    Temp = np.ones(n_steps+1)

    for i, t in enumerate(time):
        
        tau = t/86400
        
        # Solar zenith angle (simplified)
        # h = hour_angle(24*tau)
        # cos_z = func_zeta(lambda_rad, h)
        
        cos_z = np.cos(lambda_rad)* np.cos(2*np.pi*(tau-0.5))
        
        if cos_z > 0:
            # -----------------------------
            # DAY SIDE
            # -----------------------------
            Temp[i] = 262 * cos_z**0.5 + 130
        
        else:
            # -----------------------------
            # NIGHT SIDE
            # -----------------------------
            
            # Shift longitude so midnight = π
            u = 2 * np.pi * tau + np.pi
            u = u % (2 * np.pi)
            
            # Polynomial evaluation
            T_night = sum(a[k] * u**k for k in range(6))
            
            # Latitude correction
            T_lat = 35 * (np.cos(lambda_rad) - 1)
            
            Temp[i] = T_night + T_lat
    
    Temp[-1] = Temp[-2]; Temp[0] = Temp[1]

    return Temp