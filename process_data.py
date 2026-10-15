# importing libraries
import pandas as pd
import numpy as np
from calc_functions import (
    calculate_cylinder_volume,
    calculate_pressure_metrics,
    calculate_imep,
    calculate_apparent_heat_release,
    calculate_combustion_phasing,
    calculate_trapped_mass,
    calculate_t_max,
    calculate_ringing_intensity
)

# specifying the engine parameters
BORE = 0.126
STROKE = 0.130
CR = 11.7
L_ROD = 0.220
RPM = 1600     
LAMBDA = 1.3   
N_CYL = 6      

# loading the experimental data
file_path = 'data/Data-for-Experimental investigation on the performance and combustion characteristics.xlsx'
df_perf = pd.read_excel(file_path, sheet_name='Performance Data', skiprows=1)
df_comb = pd.read_excel(file_path, sheet_name='Combustion Data', skiprows=1)

# cleaning the performance data (forward filling the merged cells)
df_perf['BMEP'] = df_perf['BMEP'].ffill()
df_perf['θig '] = df_perf['θig '].ffill()
df_perf = df_perf.dropna(subset=['No.']).reset_index(drop=True)

extracted_features = []

# processing all 15 steady-state operating points
for test_idx in range(len(df_perf)):
    
    # Dynamic operational inputs from the Performance sheet
    bmep_target = df_perf.loc[test_idx, 'BMEP']
    msr_target = df_perf.loc[test_idx, 'MSR']
    ignition_timing = df_perf.loc[test_idx, 'θig ']
    m_dot_ng = df_perf.loc[test_idx, 'mNG']
    m_dot_m = df_perf.loc[test_idx, 'mM']
    
    # mapping the test index to the correct columns
    if 0 <= test_idx <= 4:
        theta_col, p_col = 0, 1 + test_idx
    elif 5 <= test_idx <= 9:
        theta_col, p_col = 7, 8 + (test_idx - 5)
    elif 10 <= test_idx <= 14:
        theta_col, p_col = 14, 15 + (test_idx - 10)
    else:
        continue
        
    # extracting and cleaning the 7200-row arrays for the specific test coordinate
    theta_array = pd.to_numeric(df_comb.iloc[:, theta_col], errors='coerce').dropna().values
    p_array = pd.to_numeric(df_comb.iloc[:, p_col], errors='coerce').dropna().values
    
    # ensuring arrays match in length before vector calculations
    min_len = min(len(theta_array), len(p_array))
    theta_array = theta_array[:min_len]
    p_array = p_array[:min_len]

    # calculating the thermodynamics properties
    
    # calculating the cylinder volume
    V, dV_dtheta_rad, v_d = calculate_cylinder_volume(theta_array, BORE, STROKE, CR, L_ROD)
    
    # calculating pressure metrics
    p_max, pprr = calculate_pressure_metrics(theta_array, p_array)
    imep = calculate_imep(p_array, V, v_d)
    
    # calculating the heat release & combustion phasing
    dQ, Q_norm, theta_comb = calculate_apparent_heat_release(theta_array, p_array, V, dV_dtheta_rad)
    ca50, ca_0_10, ca_10_90 = calculate_combustion_phasing(Q_norm, theta_comb, ignition_timing)
    
    # calculating the knock limits
    m_trapped = calculate_trapped_mass(m_dot_ng, m_dot_m, LAMBDA, RPM, N_CYL)
    t_max = calculate_t_max(p_array, V, m_trapped)
    ri = calculate_ringing_intensity(pprr, p_max, t_max)
    
    # compiling the mapped outputs
    extracted_features.append({
        'Test_No': int(test_idx + 1),
        'BMEP_MPa': bmep_target,
        'MSR_%': msr_target,
        'IMEP_bar': imep,
        'P_max_bar': p_max,
        'PPRR_bar_deg': pprr,
        'CA50_ATDC': ca50,
        'CA_0_10_deg': ca_0_10,
        'CA_10_90_deg': ca_10_90,
        'T_max_K': t_max,
        'Ringing_Intensity': ri
    })

# generating final output dataframe
df_results = pd.DataFrame(extracted_features)

print(df_results.to_string(index=False))

df_results.to_csv('data/processed_data.csv', index=False)
print("\nM successfully saved to 'processed_data.csv'.")