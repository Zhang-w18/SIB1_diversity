"""
Created on Sat Dec  7 15:04:44 2019

### Author: Mao YAN 00285287, 
### This file includes the following functions before channel coeffcients generation:
### 1. channel_initialize
### 2. channel_settings_initialize
### 3. channel_cdl_initialize
### 4. rotate_angles_max_direction
### 5. scaling_angles
### 6. calculate_delay_spread
### 7. calculate_angular_spread
"""


import numpy as np
import scipy.io as scio
from channel_functions import calculate_antenna_pattern, get_dft_codebook
from channel_functions import channel_virtualization, channel_time_interpolation
from channel_functions import channel_response_generate

"""
### function for channel initialization, including antenna settings, offline channel generation 
### cdl_type        CDL channel model: CDLA, CDLB, CDLC, CDLD, CDLE
### Kf              desired K factor, default []                          unit: dB. 
### delay_spread    desired delay spread                                  unit: second
### fc_GHz          carrier frequency                                     unit: GHz
### fs              sampling rate                                         unit: Hz
### num_tti         simulated tti's for offline channel generation        unit: dB
### tti_length      tti duration                                          unit: second
### beamforming     beamforming for the generated channel: 'yes', 'no' 
### chan_method     channel generation method: 'offline', 'online'
"""

def channel_initialize(cdl_type, delay_spread, Kf, tx_antenna, rx_antenna, ue_speed, fc_GHz, fs, num_tti, tti_length, angle_gap, power_gap, beamforming='no', chan_method='online'):
    # Desired angular spread angular_spread = {asd, zsd, asa, zsa}, where asd and zsd are for BS, asa and zsa are for the UE. 
    # angular_spread is a cell with 4 elements. If one element is [], the default value given in the CDL table is used. 
    asd, zsd, asa, zsa = [], [], [], [] # set the desired angular spread here. asd = 5; zsd = 1; asa = 30; zsa = 5;    asd = 25; zsd = 1; asa = 60; zsa = 5; 
    # asd, zsd, asa, zsa = [], 1, [], 5 # set the desired angular spread here. asd = 5; zsd = 1; asa = 30; zsa = 5;    asd = 25; zsd = 1; asa = 60; zsa = 5; 
    # asd, zsd, asa, zsa = 25, 1, 60, 5 # set the desired angular spread here. asd = 5; zsd = 1; asa = 30; zsa = 5;    asd = 25; zsd = 1; asa = 60; zsa = 5; 
    angular_spread = (asd, zsd, asa, zsa)

    # Desired angle mean angle_mean = [aod_mu, zod_mu, aoa_mu, zoa_mu]. This parameter 
    # angle_mean is a cell with 4 element. If the element is [], the default value
    # given in the CDL table is used. 
    zod_mu, zoa_mu = [], [] # set the desired angle mean here
    # # make the AOD and AOA of the strongest cluster uniform random dropped within [-30,30] degree.
    # aod_mu = (-16.62-0.3)*asd/37.4+(np.random.rand()-0.5)*60  # model_mean = -16.62, model spread is 37.4
    # aoa_mu = (77.67-170.4)*asa/71.5+(np.random.rand()-0.5)*60 # model_mean = 77.67, model spread is 71.5
    aod_mu, aoa_mu = [], [] #[], []
    angle_mean = (aod_mu, zod_mu, aoa_mu, zoa_mu)  # desired angle mean, unit: degree
    # # angle_mean = (12.0781198399811, zod_mu, -89.3151215701514, zoa_mu)  # desired angle mean, unit: degree

    UE_direction = np.array([90, 0]) # [zenith angle, horizontal angle] in degree 
    ## UE_direction = np.array([np.random.rand()*180, (np.random.rand()-0.5)*360])  # random UE velocity direction, unit: degree

    # K-factor Kf is only used in CDL_D and CDL_E. If Kf = [], the default value is used.
    cdl = channel_cdl_initialize(cdl_type, delay_spread, angular_spread, angle_mean, Kf)
       
    # Note that for the XPO antennas, the antennas with the second polarization count on vertical dimension (z-axis).
    tx_panel_Mg = tx_antenna[0]         # number of panels in z-axis demension, M in [38.901, Figure 7.3-1], including polarization 
    tx_panel_Ng = tx_antenna[1]         # number of panels in y-axis demension, N in [38.901, Figure 7.3-1]
    tx_panel_M = tx_antenna[2]          # number of elements in z-axis demension, M in [38.901, Figure 7.3-1], including polarization 
    tx_panel_N = tx_antenna[3]          # number of elements in y-axis demension, N in [38.901, Figure 7.3-1]
    tx_panel_P = tx_antenna[4]          # number of polarization, P in [38.901, Figure 7.3-1] 

    tx_panel_dV = 0.5       # element space in the vertical direction, unit: lambda 
    tx_panel_dH = 0.5       # element space in the horizontal direction, unit: lambda 
    tx_panel_dgV = 4        # panel space in the vertical direction, unit: lambda 
    tx_panel_dgH = 2        # panel space in the horizontal direction, unit: lambda 

    rx_panel_Mg = rx_antenna[0]         # number of panels in z-axis demension, Mg in [38.901, Figure 7.3-1] 
    rx_panel_Ng = rx_antenna[1]         # number of panels in y-axis demension, Ng in [38.901, Figure 7.3-1]
    rx_panel_M = rx_antenna[2]          # number of elements in z-axis demension, Mg in [38.901, Figure 7.3-1] 
    rx_panel_N = rx_antenna[3]          # number of elements in y-axis demension, Ng in [38.901, Figure 7.3-1]
    rx_panel_P = rx_antenna[4]          # number of polarization, P in [38.901, Figure 7.3-1] 

    rx_panel_dV = 0.5       # element space in the vertical direction, unit: lambda 
    rx_panel_dH = 0.5       # element space in the horizontal direction, unit: lambda 
    rx_panel_dgV = 4        # panel space in the vertical direction, unit: lambda 
    rx_panel_dgH = 2        # panel space in the horizontal direction, unit: lambda 

    tx_panel_MNP = [tx_panel_Mg, tx_panel_Ng, tx_panel_M, tx_panel_N, tx_panel_P]
    tx_panel_dVdH = [tx_panel_dgV, tx_panel_dgH, tx_panel_dV, tx_panel_dH]

    rx_panel_MNP = [rx_panel_Mg, rx_panel_Ng, rx_panel_M, rx_panel_N, rx_panel_P]
    rx_panel_dVdH = [rx_panel_dgV, rx_panel_dgH, rx_panel_dV, rx_panel_dH]

    #add angle, delay and power gap of the two trp. when only one TRP is deployed, the these gap equals to 0
    # cdl["aoa"]=(cdl["aoa"]+angle_gap)%360
    # cdl["aod"]=(cdl["aod"]-angle_gap)%360
    cdl["aod"]=(cdl["aod"]+angle_gap)%360
    cdl["power"]=cdl["power"]-power_gap
    
    channel = channel_settings_initialize(cdl, tx_panel_MNP, tx_panel_dVdH, rx_panel_MNP, rx_panel_dVdH, ue_speed, UE_direction, fc_GHz)
    #channel["pwr"]=channel["pwr"]*(10**(-power_gap/10)) 
    
    channel['beamforming'] = beamforming
    channel['tti_length'] = tti_length
    channel['fs'] = fs
    channel['cdl_type'] = cdl_type
    # DFT based beamforming is applied, one codebook for all  
    if beamforming.lower() == 'yes': 
        tx_dimension = channel['tx_panel_Mg'] * channel['tx_panel_Ng'] * channel['tx_panel_P']
        rx_dimension = channel['rx_panel_Mg'] * channel['rx_panel_Ng'] * channel['rx_panel_P']
        channel['tx_dimension'] = tx_dimension
        channel['rx_dimension'] = rx_dimension

        ## get the PDP of channel response
        this_chan_rx_tx, delays = channel_response_generate(channel, 0)

        ### DFT codebook initilization 
        tx_dft_codebook = get_dft_codebook(channel['tx_panel_M'], channel['tx_panel_N'], channel['tx_panel_P'], 0) # row as codebook
        rx_dft_codebook = get_dft_codebook(channel['rx_panel_M'], channel['rx_panel_N'], channel['rx_panel_P'], 0) # row as codebook

        ### beamforming vector search 
        ## find the best beam pair at the first TTI for each panel
        num_tx_beam = channel['tx_panel_M'] * channel['tx_panel_P'] * channel['tx_panel_N']
        num_rx_beam = channel['rx_panel_M'] * channel['rx_panel_P'] * channel['rx_panel_N']
        rsrp_of_beam_pair = np.zeros((num_rx_beam, num_tx_beam))

        ## take the first tx panel and first rx panel for finding the best DFT vector pair 
        chan_panel_rx1 = this_chan_rx_tx[:, range(num_rx_beam)]
        chan_panel_rx1_tx1 = chan_panel_rx1[:, :, range(num_tx_beam)]
        # loop for all the beam pairs, calculate the RSRP of each beam pair.
        for rx_beam_idx in range(num_rx_beam):
            for tx_beam_idx in range(num_tx_beam):
                # the following is equivalent to rx_codebook * chan * tx_codebook.transpose() 
                tmp_chan_coff = channel_virtualization(chan_panel_rx1_tx1, rx_dft_codebook[rx_beam_idx], tx_dft_codebook[tx_beam_idx])
                rsrp_of_beam_pair[rx_beam_idx][tx_beam_idx] = np.linalg.norm(tmp_chan_coff)**2 # norm已经是求二阶范数了，为什么还要平方？

        # the best beam pair is choosen as the beam pair with the maximum RSRP. 
        # best_beam_pair = 0 # rsrp_of_beam_pair.argmax()
        best_beam_pair = int(np.argmax(rsrp_of_beam_pair))
        best_rx_beam = best_beam_pair // num_tx_beam
        best_tx_beam = best_beam_pair % num_tx_beam

        # Each polarization has a beam. So XPO have 2 beams. 
        if channel['tx_panel_P'] == 2:
            best_tx_beam = np.mod(best_tx_beam, num_tx_beam//2) + np.array([0, 1]) * num_tx_beam//2

        if channel['rx_panel_P'] == 2:
            best_rx_beam = np.mod(best_rx_beam, num_rx_beam//2) + np.array([0, 1]) * num_rx_beam//2
        
        # tx_precoding_vector = tx_dft_codebook[best_tx_beam, ] 
        # rx_precoding_vector = rx_dft_codebook[best_rx_beam, ]
        tx_precoding_vector = np.kron(np.eye((channel['tx_panel_Mg']*channel['tx_panel_Ng'])), tx_dft_codebook[best_tx_beam, ])
        rx_precoding_vector = np.kron(np.eye((channel['rx_panel_Mg']*channel['rx_panel_Ng'])), rx_dft_codebook[best_rx_beam, ])

        channel['tx_precoding_vector'], channel['rx_precoding_vector'] = tx_precoding_vector, rx_precoding_vector
###############--------------------------改这里
        ## online channel response generation 
        if chan_method.lower() == 'online':#and num_tti>0:
            return channel, tx_precoding_vector, rx_precoding_vector
        # offline channel response generation 
        else: # if chan_method = 'offline'
            ## generate channel coefficients 
            chan_resp_time_rx_tx = dict()
            chan_bf_rx_tx = np.zeros((num_tti, rx_dimension, tx_dimension, channel['num_cluster']), dtype='complex')
            sum_power = 0

            for time_idx in range(num_tti):
                initial_time = time_idx * tti_length
                ## get the PDP of channel response
                this_chan_rx_tx, delays = channel_response_generate(channel, initial_time)
                chan_bf_rx_tx[time_idx] = channel_virtualization(this_chan_rx_tx, rx_precoding_vector, tx_precoding_vector)

                num_ext = 4
                num_taps = (np.ceil(delays.max() * fs) + 2 * num_ext + 1).astype('int')
                chan_resp_time_rx_tx[format(time_idx, '05d')] = np.zeros((rx_dimension, tx_dimension, num_taps), dtype='complex')
                for rx_idx in range(rx_dimension):
                    for tx_idx in range(tx_dimension): 
                        chan_resp_time_rx_tx[format(time_idx, '05d')][rx_idx][tx_idx] = channel_time_interpolation(delays, chan_bf_rx_tx[time_idx][rx_idx][tx_idx], 1/fs, num_ext)
                                
                sum_power += np.linalg.norm(chan_resp_time_rx_tx[format(time_idx, '05d')])**2

            scio.savemat("channel_coefficients[time](rx, tx, delay)_" + format(fc_GHz, '02d') + "GHz(" + cdl_type + ").mat",
                {'read_me': 'dim of channel: numtti * rx * tx * path', 'chan_coff_aft_bf': chan_bf_rx_tx,
                'delays': delays, 'rx_dimension': rx_dimension, 'tx_dimension': tx_dimension, 'fs': fs, 
                'beamforming': beamforming, 'tti_length': tti_length, 'cdl_type': cdl_type, 
                'norm_factor': np.sqrt(rx_dimension / (sum_power / num_tti))})
            
            return channel, tx_precoding_vector, rx_precoding_vector
    # no beamforming is applied
    else: # if beamforming.lower() == 'no'
        tx_dimension = channel['tx_panel_Mg'] * channel['tx_panel_Ng'] * channel['tx_panel_M'] * channel['tx_panel_N'] * channel['tx_panel_P']
        rx_dimension = channel['rx_panel_Mg'] * channel['rx_panel_Ng'] * channel['rx_panel_M'] * channel['rx_panel_N'] * channel['rx_panel_P']
        channel['tx_dimension'] = tx_dimension
        channel['rx_dimension'] = rx_dimension

        if chan_method.lower() == 'online':
            return channel, 1, 1
        else: # if chan_method = 'offline'
            chan_resp_time_rx_tx = dict()
            chan_bf_rx_tx = np.zeros((num_tti, rx_dimension, tx_dimension, channel['num_cluster']), dtype='complex')
            sum_power = 0

            for time_idx in range(num_tti):
                initial_time = time_idx * tti_length
                ## get the PDP of channel response
                this_chan_rx_tx, delays = channel_response_generate(channel, initial_time)
                chan_bf_rx_tx[time_idx] = np.moveaxis(this_chan_rx_tx, 0, -1)

                num_ext = 4
                num_taps = (np.ceil(delays.max() * fs) + 2 * num_ext + 1).astype('int')
                chan_resp_time_rx_tx[format(time_idx, '05d')] = np.zeros((rx_dimension, tx_dimension, num_taps), dtype='complex')
                for rx_idx in range(rx_dimension):
                    for tx_idx in range(tx_dimension): 
                        chan_resp_time_rx_tx[format(time_idx, '05d')][rx_idx][tx_idx] = channel_time_interpolation(delays, chan_bf_rx_tx[time_idx][rx_idx][tx_idx], 1/fs, num_ext)

                sum_power += np.linalg.norm(chan_resp_time_rx_tx[format(time_idx, '05d')])**2

            scio.savemat("channel_coefficients[time](rx, tx, delay)_" + format(fc_GHz, '02d') + "GHz(" + cdl_type + ").mat",
                {'read_me': 'dim of channel: numtti * rx * tx * path', 'chan_coff_aft_bf': chan_bf_rx_tx,
                'delays': delays, 'rx_dimension': rx_dimension, 'tx_dimension': tx_dimension, 'fs': fs, 
                'beamforming': beamforming, 'tti_length': tti_length, 'cdl_type': cdl_type, 
                'norm_factor': np.sqrt(rx_dimension / (sum_power / num_tti))})
            
            return channel, 1, 1

"""
## parameter settings for channel model: including antenna settings 
"""
def channel_settings_initialize(cdl, tx_panel_MNP, tx_panel_dVdH, rx_panel_MNP, rx_panel_dVdH, ue_speed, UE_direction, fc_GHz):
    if fc_GHz < 7: # frequency range 1 settings
        # if the antenna is omni-directional, set sla_v = 0 and Am = 0.
        bs_tht_3db = 65  # BS antenna vertical 3dB beamwidth, degree
        bs_phi_3db = 65  # BS antenna horizontal 3dB beamwidth
        bs_sla_v = 30     # BS antenna vertical, unit: dB
        bs_Am = 30        # BS antenna horizontal, unit: dB

        ue_tht_3db = 125  # UE antenna vertical 3dB beamwidth
        ue_phi_3db = 125  # UE antenna horizontal 3dB beamwidth
        ue_sla_v = 22.5     # UE antenna vertical, unit: dB
        ue_Am = 22.5        # UE antenna horizontal, unit: dB
    else: # [38.901, Table 7.3-1]
        # if the antenna is omni-directional, set sla_v = 0 and Am = 0.
        bs_tht_3db = 65  # BS antenna vertical 3dB beamwidth, degree
        bs_phi_3db = 65  # BS antenna horizontal 3dB beamwidth
        bs_sla_v = 30    # BS antenna vertical, unit: dB
        bs_Am = 30       # BS antenna horizontal, unit: dB

        ue_tht_3db = 125  # UE antenna vertical 3dB beamwidth
        ue_phi_3db = 125  # UE antenna horizontal 3dB beamwidth
        ue_sla_v = 22.5   # UE antenna vertical, unit: dB
        ue_Am = 22.5      # UE antenna horizontal, unit: dB

    delays = cdl['delays']       # unit: second
    power = cdl['power']         # unit: dB
    XPR = cdl['XPR']             # cross polarization power ratio: unit: dB
    is_nlos = cdl['is_nlos'] # LOS state

    # if the chanel is LOS, set the angles and the power of the LOS path
    if is_nlos == 0:
        los_aoa = cdl['los_aoa']
        los_aod = cdl['los_aod']
        los_zoa = cdl['los_zoa']
        los_zod = cdl['los_zod']
        los_power = cdl['los_power']

    num_cluster = len(delays)  # number of cluster (path)
    num_subpath = 20           # number of subcluster (subpath or ray)

    xpr_mat = 10**(XPR/10) * np.ones([num_cluster, num_subpath])   # XPR, dB to linnear. Note that xpr for CDL is a constant.

    ## get the angles of each subcluster from CDL
    aoa = cdl['aoa']
    aod = cdl['aod']
    zoa = cdl['zoa']
    zod = cdl['zod']

    ## calculate random phase
    phase_vv = np.exp(1j*np.random.rand(num_cluster, num_subpath)*2*np.pi)
    phase_hh = np.exp(1j*np.random.rand(num_cluster, num_subpath)*2*np.pi)
    phase_vh = np.exp(1j*np.random.rand(num_cluster, num_subpath)*2*np.pi)
    phase_hv = np.exp(1j*np.random.rand(num_cluster, num_subpath)*2*np.pi)
    phase_vv_los = np.exp(1j * np.random.rand() * 2 * np.pi)

    ### channel initilization 
    channel = dict()
    # channel['time'] = initial_time

    # tx antenna settings 
    channel['tx_panel_Mg'] = tx_panel_MNP[0]        # number of panels in z-axis demension, M in [38.901, Figure 7.3-1] 
    channel['tx_panel_Ng'] = tx_panel_MNP[1]        # number of panels in y-axis demension, N in [38.901, Figure 7.3-1]
    channel['tx_panel_M'] = tx_panel_MNP[2]         # number of elements in z-axis demension, M in [38.901, Figure 7.3-1]
    channel['tx_panel_N'] = tx_panel_MNP[3]         # number of elements in y-axis demension, N in [38.901, Figure 7.3-1]
    channel['tx_panel_P'] = tx_panel_MNP[4]         # number of polarizations, P in [38.901, Figure 7.3-1]
    if channel['tx_panel_P'] == 2: 
        channel['tx_slant'] = [-45, 45]    # XPO, degree
    elif channel['tx_panel_P'] == 1: 
        channel['tx_slant'] = np.array([90])           # # ULA, degree

    channel['tx_panel_dgV'] = tx_panel_dVdH[0]      # panel space in the vertical direction, unit: lambda 
    channel['tx_panel_dgH'] = tx_panel_dVdH[1]      # panel space in the horizontal direction, unit: lambda 
    channel['tx_panel_dV'] = tx_panel_dVdH[2]       # element space in the vertical direction, unit: lambda 
    channel['tx_panel_dH'] = tx_panel_dVdH[3]       # element space in the horizontal direction, unit: lambda 

    # rx antenna settings 
    channel['rx_panel_Mg'] = rx_panel_MNP[0]        # number of panels in z-axis demension, M in [38.901, Figure 7.3-1] 
    channel['rx_panel_Ng'] = rx_panel_MNP[1]        # number of panels in y-axis demension, N in [38.901, Figure 7.3-1]
    channel['rx_panel_M'] = rx_panel_MNP[2]         # number of elements in z-axis demension, M in [38.901, Figure 7.3-1]
    channel['rx_panel_N'] = rx_panel_MNP[3]         # number of elements in y-axis demension, N in [38.901, Figure 7.3-1]
    channel['rx_panel_P'] = rx_panel_MNP[4]         # number of polarizations, P in [38.901, Figure 7.3-1]
    if channel['rx_panel_P'] == 2:
        channel['rx_slant'] = [-45, 45]    # XPO, degree
        # channel['rx_slant'] = [90, 0]    # XPO, degree
    elif channel['rx_panel_P'] == 1: 
        channel['rx_slant'] = np.array([90])           # ULA, degree

    channel['rx_panel_dgV'] = rx_panel_dVdH[0]      # panel space in the vertical direction, unit: lambda 
    channel['rx_panel_dgH'] = rx_panel_dVdH[1]      # panel space in the horizontal  direction, unit: lambda 
    channel['rx_panel_dV'] = rx_panel_dVdH[2]       # element space in the vertical direction, unit: lambda 
    channel['rx_panel_dH'] = rx_panel_dVdH[3]       # element space in the horizontal  direction, unit: lambda 

    channel['fc_GHz'] = fc_GHz
    channel['num_cluster'] = num_cluster
    channel['pwr'] = 10**(power/10)

    channel['delays'] = delays
    channel['is_nlos'] = is_nlos    
    channel['v'] = ue_speed
    channel['v_dir'] = UE_direction
    channel['xpr_mat'] = xpr_mat
    channel['phase_vv'] = phase_vv
    channel['phase_hh'] = phase_hh
    channel['phase_vh'] = phase_vh
    channel['phase_hv'] = phase_hv
    channel['phase_vv_los'] = phase_vv_los

    channel['aod'] = aod
    channel['zod'] = zod
    channel['aoa'] = aoa
    channel['zoa'] = zoa
    
    ### beam shape
    channel['tx_ant_gain'] = np.zeros(zod.shape)
    channel['rx_ant_gain'] = np.zeros(zod.shape)
    for idx0 in range(zod.shape[0]):
        for idx1 in range(zod.shape[1]):
            channel['tx_ant_gain'][idx0][idx1] = calculate_antenna_pattern(zod[idx0][idx1], aod[idx0][idx1], bs_tht_3db, bs_phi_3db, bs_sla_v, bs_Am)
            channel['rx_ant_gain'][idx0][idx1] = calculate_antenna_pattern(zoa[idx0][idx1], aoa[idx0][idx1], ue_tht_3db, ue_phi_3db, ue_sla_v, ue_Am)

    if is_nlos == 0:
        channel['los_aod'] = los_aod
        channel['los_zod'] = los_zod
        channel['los_aoa'] = los_aoa
        channel['los_zoa'] = los_zoa
        channel['los_power'] = 10**(los_power/10)

        channel['tx_los_ant_gain'] = calculate_antenna_pattern(los_zod, los_aod, bs_tht_3db, bs_phi_3db, bs_sla_v, bs_Am)
        channel['rx_los_ant_gain'] = calculate_antenna_pattern(los_zoa, los_aoa, ue_tht_3db, ue_phi_3db, ue_sla_v, ue_Am)
    
    return channel

"""
Created on Sat Dec  7 15:04:44 2019

### Author: Mao YAN 00285287, [Originally by Yanshen DU, d00354520), 
### This function produce the parameters of CDL channel model.
### History: 
###  1. 2016-09-28  Created this code based on TR 38.900 v1.0.0   
###  2. 2017-02-13  Modified based on TR 38.900 v14.2.0
%%%%%%%%%%%%%%%%%%%###  input %%%%%%%%%%%%%%%%%%%%%%%%%%%
### cdl_type          choose the CDL channel type                           
### delay_spread      desired delay spread                                  unit: second
### angular_spread    desired angular spread, {asd, zsd, asa, zsa}          unit: degree
### angle_mean        desired angle mean, [aod_mu, zod_mu, aoa_mu, zoa_mu)  unit: degree
### Kf                desired K factor                                      unit: dB
%%%%%%%%%%%%%%%%%%%###  output %%%%%%%%%%%%%%%%%%%%%%%%%%%
### CDL                structure contains all the CDL parameters
"""

## load cluster delay line model parameters [TS 38.901, 7.7.1]
def channel_cdl_initialize(cdl_type, delay_spread, angular_spread, angle_mean, Kf):
    cdl = dict()

    # get CDL paramters from CDL table
    if cdl_type.upper() == 'CDLA': 
        cdl['is_nlos'] = 1  # NLOS
        cdl['delays'] = np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
            0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119,
            3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
        cdl['power'] = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9,
            6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
            -18.3, -18.9, -16.6, -19.9, -29.7])

        cdl['norm_aod'] = np.array([-178.1, -4.2, -4.2, -4.2, 90.2, 90.2, 90.2, 121.5,
            -81.7, 158.4, -83, 134.8, -153, -172, -129.9, -136,
            165.4, 148.4, 132.7, -118.6, -154.1, 126.5, -56.2])
        cdl['norm_aoa'] = np.array([51.3, -152.7, -152.7, -152.7, 76.6, 76.6, 76.6, -1.8,
            -41.9, 94.2, 51.9, -115.9, 26.6, 76.6, -7, -23, -47.2, 110.4,
            144.5, 155.3, 102, -151.8, 55.2])
        cdl['norm_zod'] = np.array([50.2, 93.2, 93.2, 93.2, 122, 122, 122, 150.2, 55.2,
            26.4, 126.4, 171.6, 151.4, 157.2, 47.2, 40.4, 43.3, 161.8,
            10.8, 16.7, 171.7, 22.7, 144.9])
        cdl['norm_zoa'] = np.array([125.4, 91.3, 91.3, 91.3, 94, 94, 94, 47.1, 56, 30.1,
            58.8, 26, 49.2, 143.1, 117.4, 122.7, 123.2, 32.6, 27.2, 15.2,
            146, 150.7, 156.1])
        
        cdl['los_aod'] = []
        cdl['los_aoa'] = []
        cdl['los_zod'] = []
        cdl['los_zoa'] = []
        cdl['los_power'] = []
        
        cdl['c_asd'] = 5
        cdl['c_asa'] = 11
        cdl['c_zsd'] = 3
        cdl['c_zsa'] = 3
        
        cdl['XPR'] = 10
    elif cdl_type.upper() == 'CDLB': 
        cdl['is_nlos'] = 1  # NLOS
        cdl['delays'] = np.array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752,
            0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474,
            1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834])
        
        cdl['power'] = np.array([0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3,
            -8.9, -9, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8,
            -11.4, -14.9, -9.2, -11.3])
            
        cdl['norm_aod'] = np.array([9.3, 9.3, 9.3, -34.1, -65.4, -11.4, -11.4, -11.4,
            -67.2, 52.5, -72, 74.3, -52.2, -50.5, 61.4, 30.6,
            -72.5, -90.6, -77.6, -82.6, -103.6, 75.6, -77.6])
        cdl['norm_aoa'] = np.array([-173.3, -173.3, -173.3, 125.5, -88.0, 155.1, 155.1,
            155.1, -89.8, 132.1, -83.6, 95.3, 103.7, -87.8, -92.5, -139.1,
            -90.6, 58.6, -79.0, 65.8, 52.7, 88.7, -60.4])
        cdl['norm_zod'] = np.array([105.8, 105.8, 105.8, 115.3, 119.3, 103.2, 103.2,
            103.2, 118.2, 102.0, 100.4, 98.3, 103.4, 102.5, 101.4, 103.0,
            100.0, 115.2, 100.5, 119.6, 118.7, 117.8, 115.7])
        cdl['norm_zoa'] = np.array([78.9, 78.9, 78.9, 63.3, 59.9, 67.5, 67.5, 67.5,
            82.6, 66.3, 61.6, 58.0, 78.2, 82.0, 62.4, 78.0, 60.9, 82.9,
            60.8, 57.3, 59.9, 60.1, 62.3])
        
        cdl['los_aod'] = []
        cdl['los_aoa'] = []
        cdl['los_zod'] = []
        cdl['los_zoa'] = []
        cdl['los_power'] = []
        
        cdl['c_asd'] = 10
        cdl['c_asa'] = 22
        cdl['c_zsd'] = 3
        cdl['c_zsa'] = 7
        
        cdl['XPR'] = 8
    elif cdl_type.upper() == 'CDLC': 
        cdl['is_nlos'] = 1  # NLOS
        cdl['delays'] = np.array([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
            0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
            4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
        cdl['power'] = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1,
            -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
            -17.1, -16, -15.7, -21.6, -22.8])
        
        cdl['norm_aod'] = np.array([-46.6, -22.8, -22.8, -22.8, -40.7, 0.3, 0.3, 0.3, 73.1,
            -64.5, 80.2, -97.1, -55.3, -64.3, -78.5, 102.7, 99.2, 88.8,
            -101.9, 92.2, 93.3, 106.6, 119.5, -123.8])
        cdl['norm_aoa'] = np.array([-101, 120, 120, 120, -127.5, 170.4, 170.4, 170.4, 55.4,
            66.5, -48.1, 46.9, 68.1, -68.7, 81.5, 30.7, -16.4, 3.8, -13.7,
            9.7, 5.6, 0.7, -21.9, 33.6])
        cdl['norm_zod'] = np.array([97.2, 98.6, 98.6, 98.6, 100.6, 99.2, 99.2, 99.2, 105.2,
            95.3, 106.1, 93.5, 103.7, 104.2, 93.0, 104.2, 94.9, 93.1,
            92.2, 106.7, 93.0, 92.9, 105.2, 107.8])
        cdl['norm_zoa'] = np.array([87.6, 72.1, 72.1, 72.1, 70.1, 75.3, 75.3, 75.3, 67.4,
            63.8, 71.4, 60.5, 90.6, 60.1, 61.0, 100.7, 62.3, 66.7, 52.9,
            61.8, 51.9, 61.7, 58, 57])
        
        cdl['los_aod'] = []
        cdl['los_aoa'] = []
        cdl['los_zod'] = []
        cdl['los_zoa'] = []
        cdl['los_power'] = []
        
        cdl['c_asd'] = 2
        cdl['c_asa'] = 15
        cdl['c_zsd'] = 3
        cdl['c_zsa'] = 7
        
        cdl['XPR'] = 7
    elif cdl_type.upper() == 'CDLD':
        cdl['is_nlos'] = 0  # LOS
        cdl['delays'] = np.array([0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775,
            4.042, 7.937, 9.424, 9.708, 12.525])
        cdl['power'] = np.array([-13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9,
            -27.8, -23.6, -24.8, -30.0, -27.7])
        cdl['norm_aod'] = np.array([0, 89.2, 89.2, 89.2, 13, 13, 13, 34.6, -64.5, -32.9,
            52.6, -132.1, 77.2])
        cdl['norm_aoa'] = np.array([-180, 89.2, 89.2, 89.2, 163, 163, 163, -137, 74.5,
            127.7, -119.6, -9.1, -83.8])
        cdl['norm_zod'] = np.array([98.5, 85.5, 85.5, 85.5, 97.5, 97.5, 97.5, 98.5,
            88.4, 91.3, 103.8, 80.3, 86.5])
        cdl['norm_zoa'] = np.array([81.5, 86.9, 86.9, 86.9, 79.4, 79.4, 79.4, 78.2,
            73.6, 78.3, 87, 70.6, 72.9])
        
        cdl['los_aod'] = 0
        cdl['los_aoa'] = -180
        cdl['los_zod'] = 98.5
        cdl['los_zoa'] = 81.5
        cdl['los_power'] = -0.2
        
        cdl['c_asd'] = 5
        cdl['c_asa'] = 8
        cdl['c_zsd'] = 3
        cdl['c_zsa'] = 3
        
        cdl['XPR'] = 11
        #       power_scale = 9;
    elif cdl_type.upper() == 'CDLE': 
        cdl['is_nlos'] = 0  # LOS
        cdl['delays'] = np.array([0.000, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092,
            1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6419])
        
        cdl['power'] = np.array([-22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6,
            -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2])

        cdl['norm_aod'] = np.array([0, 57.5, 57.5, 57.5, -20.1, 16.2, 9.3, 9.3, 9.3,
            19, 32.7, 0.5, 55.9, 57.6])
        cdl['norm_aoa'] = np.array([-180, 18.2, 18.2, 18.2, 101.8, 112.9, -155.5,
            -155.5, -155.5, -143.3, -94.7, 147, -36.2, -26])
        cdl['norm_zod'] = np.array([99.6, 104.2, 104.2, 104.2, 99.4, 100.8, 98.8,
            98.8, 98.8, 100.8, 96.4, 98.9, 95.6, 104.6])
        cdl['norm_zoa'] = np.array([80.4, 80.4, 80.4, 80.4, 80.8, 86.3, 82.7, 82.7,
            82.7, 82.9, 88, 81, 88.6, 78.3])
            
        cdl['los_aod'] = 0
        cdl['los_aoa'] = -180
        cdl['los_zod'] = 99.6
        cdl['los_zoa'] = 80.4
        cdl['los_power'] = -0.03
        
        cdl['c_asd'] = 5
        cdl['c_asa'] = 11
        cdl['c_zsd'] = 3
        cdl['c_zsa'] = 7
        
        cdl['XPR'] = 8
        #       power_scale = 9.3;

    if Kf != [] and cdl['is_nlos'] == 0:  # if K-factor is assigned in the LOS case
        Kf_mod = cdl['los_power'] - 10 * np.log10((np.sum(10**(cdl['power']/10))))
        cdl['power'] = cdl['power'] - Kf + Kf_mod
        ds = calculate_delay_spread(cdl['delays'], cdl['power'])
        cdl['delays'] = cdl['delays'] / ds

    if delay_spread:
        cdl['delays'] = cdl['delays'] * delay_spread


    # get ASA/ASD/ZSA/ZSD from the input angular spread
    asd, zsd, asa, zsa = angular_spread
    # asd = angular_spread[0]
    # zsd = angular_spread[1]
    # asa = angular_spread[2]
    # zsa = angular_spread[3]

    # get the AOA/AOD/ZOA/ZOD mean from the input angle mean
    aod_mean, zod_mean, aoa_mean, zoa_mean = angle_mean
    # aod_mean = angle_mean[0]
    # zod_mean = angle_mean[1]
    # aoa_mean = angle_mean[2]
    # zoa_mean = angle_mean[3]

    # angle offset of the subpath
    subpath_angle_offset = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492,
        0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844,
        -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

    num_cluster = len(cdl['delays'])   # number of path
    num_subpath = len(subpath_angle_offset)

    # calculate the angle of each subpath and do the random coupling
    cdl['aoa'], cdl['aod'] = np.array(np.zeros([num_cluster, num_subpath])), np.array(np.zeros([num_cluster, num_subpath]))
    cdl['zoa'], cdl['zod'] = np.array(np.zeros([num_cluster, num_subpath])), np.array(np.zeros([num_cluster, num_subpath]))

    for path_idx in range(num_cluster): 
        tmp_aoa = cdl['norm_aoa'][path_idx] + cdl['c_asa'] * subpath_angle_offset
        cdl['aoa'][path_idx] = np.random.permutation(tmp_aoa)

        tmp_aod = cdl['norm_aod'][path_idx] + cdl['c_asd'] * subpath_angle_offset
        cdl['aod'][path_idx] = np.random.permutation(tmp_aod)

        tmp_zoa = cdl['norm_zoa'][path_idx] + cdl['c_zsa'] * subpath_angle_offset
        cdl['zoa'][path_idx] = np.random.permutation(tmp_zoa)

        tmp_zod = cdl['norm_zod'][path_idx] + cdl['c_zsd'] * subpath_angle_offset
        cdl['zod'][path_idx] = np.random.permutation(tmp_zod)

    # rotate the angles to make their major angular directions match the input angular directions.
    cdl['aod'], cdl['los_aod'] = rotate_angles_max_direction(cdl['aod'], cdl['los_aod'], 30, cdl['power'], cdl['los_power'])
    cdl['zod'], cdl['los_zod'] = rotate_angles_max_direction(cdl['zod'], cdl['los_zod'], 90, cdl['power'], cdl['los_power'])
    cdl['aoa'], cdl['los_aoa'] = rotate_angles_max_direction(cdl['aoa'], cdl['los_aoa'], 0, cdl['power'], cdl['los_power'])
    cdl['zoa'], cdl['los_zoa'] = rotate_angles_max_direction(cdl['zoa'], cdl['los_zoa'], 90, cdl['power'], cdl['los_power'])

    # scaling the angles to make their angular spread and offset match the input angular spread and offset.
    cdl['aod'], cdl['los_aod'] = scaling_angles(cdl['aod'], cdl['los_aod'], asd, aod_mean, cdl['power'], cdl['los_power'])
    cdl['zod'], cdl['los_zod'] = scaling_angles(cdl['zod'], cdl['los_zod'], zsd, zod_mean, cdl['power'], cdl['los_power'])
    cdl['aoa'], cdl['los_aoa'] = scaling_angles(cdl['aoa'], cdl['los_aoa'], asa, aoa_mean, cdl['power'], cdl['los_power'])
    cdl['zoa'], cdl['los_zoa'] = scaling_angles(cdl['zoa'], cdl['los_zoa'], zsa, zoa_mean, cdl['power'], cdl['los_power'])

    return cdl
def circular_mean_deg(angle_deg, axis=-1):
    a = np.deg2rad(angle_deg)
    return (np.rad2deg(np.angle(np.mean(np.exp(1j * a), axis=axis))) + 360.0) % 360.0


def wrap_azimuth_deg(x):
    return (np.asarray(x, dtype=float) + 360.0) % 360.0


def rotate_angles_max_direction(norm_angle, los_angle, angle_desired, power, los_power):
    max_idx = power.argmax()

    # 原来是 np.sum(norm_angle, 1) / 20，负角/跨0度时会错
    mean_angle = circular_mean_deg(norm_angle, axis=1)

    if angle_desired == "" or angle_desired == []:
        angles = norm_angle
    else:
        if los_angle == "" or los_angle == []:
            angles = norm_angle - mean_angle[max_idx] + float(angle_desired)
            angles = wrap_azimuth_deg(angles)
            los_angle = []
        else:
            angles = norm_angle - float(los_angle) + float(angle_desired)
            angles = wrap_azimuth_deg(angles)
            los_angle = float(angle_desired) % 360.0

    return angles, los_angle
# def rotate_angles_max_direction(norm_angle, los_angle, angle_desired, power, los_power):
#     max_idx = power.argmax()
#     mean_angle = np.sum(norm_angle, 1) / 20
#
#     if angle_desired == "" or angle_desired == []:
#         angles = norm_angle
#     else:
#         if los_angle == "" or los_angle == []:
#             angles = norm_angle - mean_angle[max_idx] + angle_desired
#             los_angle = []
#         else:
#             angles = norm_angle - los_angle + angle_desired
#             los_angle = angle_desired
#
#     return angles, los_angle


## scaling the angles according to 7.7.5.1 of TR 38900 v14.2.0.
def scaling_angles(angle_mod, los_angle_mod, AS_desired, angle_mean_desired, power, los_power): 
    # compute the angular spread and the angle mean.
    AS_model, angle_mean_mod = calculate_angular_spread(angle_mod, los_angle_mod, power, los_power)

    # if desired angular spread is assigned, the angle will be scaled by the
    # desired angular spread.
    if AS_desired: 
        AS_ratio = AS_desired/AS_model
    else:
        AS_ratio = 1


    # if desired angle mean is not assigned, it will be setted as the default
    # angle mean computed according to the CDL table.
    if angle_mean_desired == "" or angle_mean_desired == []:
        angle_mean_desired = angle_mean_mod
    if los_angle_mod == "" or los_angle_mod == []:
        los_angle_mod = 0


    # Note that if both the AS_desired and the angle_mean_desired are not assigned,
    # the angles will not changed.
    angles = AS_ratio*(angle_mod - angle_mean_mod) + angle_mean_desired
    los_angle = AS_ratio*(los_angle_mod - angle_mean_mod) + angle_mean_desired

    return angles, los_angle

## calculate the delay spread.
def calculate_delay_spread(delay, power_db): 
    power = 10**(power_db/10)
    power = power / np.sum(power)
    mu_tau = np.sum(delay * power)
    mu_tau_2 = np.sum(delay**2 * power)
    ds = np.sqrt(mu_tau_2 - mu_tau**2)

    return ds

## calculate the angular spread according to the Annex A of TR 38900
## v14.2.0. Additioanly, the mean of the angle is also computed here.
def calculate_angular_spread(angle_mod, los_angle_mod, power_db, los_power_db): 
    # Note that angle_mod is a matrix, each row reprsent a path (cluster).
    num_ray = np.size(angle_mod, axis=1)
    power = 10**(power_db.flatten('F') / 10)
    ray_power = np.kron(power / num_ray, np.ones((num_ray, 1)))
    
    # transform the angle and power matrix into vector
    ray_angle = angle_mod.flatten('F')
    ray_power = ray_power.flatten('C')

    # if LOS, add the los angle and los power to the vectors
    if los_power_db:
        ray_angle = np.concatenate(([los_angle_mod], ray_angle))
        ray_power = np.concatenate(([10**(los_power_db / 10)], ray_power))

    ray_power = ray_power / np.sum(ray_power)

    AS = np.sqrt(-2 * np.log(np.abs(sum(np.exp(1j*ray_angle*np.pi/180)*ray_power))))
    angle_mean = np.angle(np.sum(np.exp(1j*ray_angle*np.pi/180) * ray_power), deg=True)
    # AS = np.sqrt(-2 * np.log(np.abs(sum(np.exp(1j*ray_angle*np.pi/180)*ray_power))))*180/np.pi
    # angle_mean = np.sum(ray_angle * ray_power)

    return AS, angle_mean

def get_ul_channel(channel_dl):
    
    channel_ul=channel_dl.copy()
    channel_ul["aoa"]=channel_dl["aod"]
    channel_ul["aod"]=channel_dl["aoa"]
    channel_ul["zoa"]=channel_dl["zod"]
    channel_ul["zod"]=channel_dl["zoa"]
    
    channel_ul["los_aoa"]=channel_dl["los_aod"]
    channel_ul["los_aod"]=channel_dl["los_aoa"]
    channel_ul["los_zoa"]=channel_dl["los_zod"]
    channel_ul["los_zod"]=channel_dl["los_zoa"]
    
    channel_ul["rx_ant_gain"]=channel_dl["tx_ant_gain"]
    channel_ul["tx_ant_gain"]=channel_dl["rx_ant_gain"]
    channel_ul["rx_dimension"]=channel_dl["tx_dimension"]
    channel_ul["tx_dimension"]=channel_dl["rx_dimension"]
    channel_ul["rx_los_ant_gain"]=channel_dl["tx_los_ant_gain"]
    channel_ul["tx_los_ant_gain"]=channel_dl["rx_los_ant_gain"]
    channel_ul["rx_panel_M"]=channel_dl["tx_panel_M"]
    channel_ul["tx_panel_M"]=channel_dl["rx_panel_M"]
    channel_ul["rx_panel_Mg"]=channel_dl["tx_panel_Mg"]
    channel_ul["tx_panel_Mg"]=channel_dl["rx_panel_Mg"]
    channel_ul["rx_panel_N"]=channel_dl["tx_panel_N"]
    channel_ul["tx_panel_N"]=channel_dl["rx_panel_N"]
    channel_ul["rx_panel_Ng"]=channel_dl["tx_panel_Ng"]
    channel_ul["tx_panel_Ng"]=channel_dl["rx_panel_Ng"]
    channel_ul["rx_panel_P"]=channel_dl["tx_panel_P"]
    channel_ul["tx_panel_P"]=channel_dl["rx_panel_P"]
    channel_ul["rx_panel_dH"]=channel_dl["tx_panel_dH"]
    channel_ul["tx_panel_dH"]=channel_dl["rx_panel_dH"]
    channel_ul["rx_panel_dV"]=channel_dl["tx_panel_dV"]
    channel_ul["tx_panel_dV"]=channel_dl["rx_panel_dV"]
    channel_ul["rx_panel_dgH"]=channel_dl["tx_panel_dgH"]
    channel_ul["tx_panel_dgH"]=channel_dl["rx_panel_dgH"]
    channel_ul["rx_panel_dgV"]=channel_dl["tx_panel_dgV"]
    channel_ul["tx_panel_dgV"]=channel_dl["rx_panel_dgV"]    
    channel_ul["rx_precoding_vector"]=channel_dl["tx_precoding_vector"]
    channel_ul["tx_precoding_vector"]=channel_dl["rx_precoding_vector"]    
    channel_ul["rx_slant"]=channel_dl["tx_slant"]
    channel_ul["tx_slant"]=channel_dl["rx_slant"]  
    
    return channel_ul