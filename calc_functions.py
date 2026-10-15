# importing libraries
import numpy as np
import scipy.integrate as integrate

def calculate_cylinder_volume(theta_deg, bore, stroke, cr, l_rod):
    """Calculates instantaneous cylinder volume and its derivative."""
    r_crank = stroke / 2.0
    v_d = (np.pi * bore**2 / 4.0) * stroke
    v_c = v_d / (cr - 1.0)
    
    theta_rad = np.radians(theta_deg)
    
    term1 = l_rod + r_crank
    term2 = r_crank * np.cos(theta_rad)
    term3 = np.sqrt(l_rod**2 - (r_crank * np.sin(theta_rad))**2)
    
    # Instantaneous Volume (m^3)
    V = v_c + (np.pi * bore**2 / 4.0) * (term1 - term2 - term3)
    
    # Derivative of volume w.r.t crank angle (m^3/rad)
    dV_dtheta_rad = (np.pi * bore**2 / 4.0) * r_crank * np.sin(theta_rad) * \
                (1 + (r_crank * np.cos(theta_rad)) / np.sqrt(l_rod**2 - (r_crank * np.sin(theta_rad))**2))
    
    return V, dV_dtheta_rad, v_d


def calculate_pressure_metrics(theta_deg, p_bar):
    """Extracts peak pressure and maximum pressure rise rate."""
    p_max = np.max(p_bar)
    # Gradient in bar/deg for standard PPRR reporting
    dp_dtheta_deg = np.gradient(p_bar, theta_deg) 
    pprr = np.max(dp_dtheta_deg)
    
    return p_max, pprr


def calculate_imep(p_bar, V, v_d):
    """Integrates P*dV over the cycle to find Indicated Mean Effective Pressure."""
    p_pa = p_bar * 1e5
    work_indicated = np.trapezoid(p_pa, V)
    imep_bar = (work_indicated / v_d) / 1e5
    
    return imep_bar


def calculate_apparent_heat_release(theta_deg, p_bar, V, dV_dtheta_rad, gamma=1.33):
    """Calculates First Law net heat release rate and normalized cumulative heat."""
    p_pa = p_bar * 1e5
    theta_rad = np.radians(theta_deg)
    dp_dtheta_rad = np.gradient(p_pa, theta_rad)
    
    # dQ/dtheta (Joules/rad)
    dQ_dtheta = (gamma / (gamma - 1)) * p_pa * dV_dtheta_rad + \
                (1 / (gamma - 1)) * V * dp_dtheta_rad
                
    # filtering to combustion window (-40 to +60 ATDC) to eliminate gas-exchange noise
    comb_mask = (theta_deg >= -40) & (theta_deg <= 60)
    theta_comb = theta_deg[comb_mask]
    dQ_comb = dQ_dtheta[comb_mask]
    
    # calculating cumulative heat release
    Q_cum = integrate.cumulative_trapezoid(dQ_comb, theta_comb, initial=0)
    Q_cum_norm = Q_cum / np.max(Q_cum)
    
    return dQ_dtheta, Q_cum_norm, theta_comb


def calculate_combustion_phasing(Q_cum_norm, theta_comb, ignition_timing):
    """Interpolates cumulative heat arrays to find critical CA markers."""
    ca10 = np.interp(0.10, Q_cum_norm, theta_comb)
    ca50 = np.interp(0.50, Q_cum_norm, theta_comb)
    ca90 = np.interp(0.90, Q_cum_norm, theta_comb)
    
    ca_0_10 = ca10 - ignition_timing
    ca_10_90 = ca90 - ca10
    
    return ca50, ca_0_10, ca_10_90


def calculate_trapped_mass(m_dot_ng_kgh, m_dot_m_kgh, lambda_val=1.3, rpm=1600, n_cyl=6):
    """
    Calculates trapped mass per cylinder per cycle (kg) deterministically.
    Uses stoichiometric AF ratios: NG = 17.4, Methanol = 6.5.
    """
    # calculating air mass flow rate (kg/h)
    m_dot_air_kgh = lambda_val * (m_dot_ng_kgh * 17.4 + m_dot_m_kgh * 6.5)
    
    # calculating Total mass flow rate (kg/h)
    m_dot_total_kgh = m_dot_air_kgh + m_dot_ng_kgh + m_dot_m_kgh
    
    # calculating total system cycles per hour (4-stroke engine: 1 cycle per 2 revs)
    cycles_per_hour = (rpm / 2.0) * 60
    total_system_cycles = cycles_per_hour * n_cyl
    
    # calculating mass per cycle per cylinder (kg)
    m_trapped_kg = m_dot_total_kgh / total_system_cycles
    
    return m_trapped_kg


def calculate_t_max(p_bar, V_m3, m_trapped_kg, R=287.0):
    """
    Sweeps the Ideal Gas Law across the 7200-row arrays to find peak temperature.
    """
    p_pa = p_bar * 1e5
    # T = PV / mR
    T_array = (p_pa * V_m3) / (m_trapped_kg * R)
    
    return np.max(T_array)


def calculate_ringing_intensity(pprr_bar_deg, p_max_bar, t_max_k, gamma=1.33, R=287.0):
    """
    Calculates knock intensity using Eng's correlation.
    Returns RI in MW/m^2.
    """
    # Paper Eq (8): RI ≈ (1 / 2γ) * [0.05 * (dP/dθ_max)] / P_max * sqrt(γ * R * T_max)
    ri = (1.0 / (2.0 * gamma)) * ((0.05 * pprr_bar_deg) / p_max_bar) * np.sqrt(gamma * R * t_max_k)
    
    return ri