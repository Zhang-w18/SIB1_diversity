"""
Created on Sat Dec  7 15:04:44 2019

### Author: Mao YAN 00285287, [Originally by Yanshen DU, d00354520), revised by Mao YAN 00285287] 
### This file includes the following functions during channel coeffcients generation:
### 1. channel_fading_signal
### 2. channel_response_generate
### 3. calculate_antenna_pattern
### 4. get_dft_codebook
### 5. channel_virtualization
### 6. channel_time_interpolation
### 7. channel_frequency_response
### This function produce the parameters of CDL channel model.
"""

import numpy as np
import scipy as sp


### This function compute the channel response of the given input signal, both in time domain.
### Attention: the generated channel is not normlized for 'online'!!!!! 
def channel_fading_signal(input_signal, chan_method, channel, idx_tti, num_fft, system_par, RSRP_gap, IdxSC):
    if system_par["TransMode"].lower() == "TM1".lower():
        input_signal = np.kron(input_signal, [1, 1] / np.sqrt(2)) # yanmao 00285287: this line is used only for TM1
    
    if channel['cdl_type'] == 'awgn':
        signal_aft_channel = input_signal
        rx_dimension, tx_dimension = channel['rx_dimension'], channel['tx_dimension']
        return signal_aft_channel, 1, np.ones([rx_dimension, tx_dimension, num_fft], dtype='complex') 

    if chan_method == 'online': 
        #this_chan_rx_tx: cluster* rx_antenna_element * tx_antenna_element: the channel of each antenna element on each cluster
        this_chan_rx_tx, delays = channel_response_generate(channel, channel['tti_length'] * idx_tti)
        
        if channel['beamforming'] == 'yes': 
            #this_chan_aft_bf: rx*tx*cluster: convert antenna element channel to trx channel
            this_chan_aft_bf = channel_virtualization(this_chan_rx_tx, channel['rx_precoding_vector'], channel['tx_precoding_vector'])
        else:
            this_chan_aft_bf = np.moveaxis(this_chan_rx_tx, 0, -1)

        rx_dimension, tx_dimension = channel['rx_dimension'], channel['tx_dimension']
    elif chan_method == 'offline':
        chan_coff_aft_bf, delays = channel['chan_coff_aft_bf'], channel['delays']
        rx_dimension, tx_dimension = channel['rx_dimension'][0,0], channel['tx_dimension'][0,0]
        this_chan_aft_bf = chan_coff_aft_bf[idx_tti] * channel['norm_factor']

    fs = channel['fs']


    num_ext = 4
    #从第一个cluster对应的采样点前的第四个采样点开始，到最后一个cluster对应的采样点后的第4个采样点
    #cluster内又没有时延扩展，为什么需要扩展采样点呢？
    num_taps = (np.ceil(delays.max() * fs) + 2 * num_ext + 1).astype('int')
    
    if len(delays.shape) == 2: 
        num_taps = num_taps[0, 0]

    chan_rx_tx_time_ideal = np.zeros([rx_dimension, tx_dimension, num_taps], dtype='complex')
    signal_aft_channel = np.zeros((input_signal.shape[0], rx_dimension), dtype='complex')
    
    for rx_idx in range(rx_dimension):
        for tx_idx in range(tx_dimension):            
            #根据各个cluster对应的采样点，确定时域信道chan_rx_tx_time_ideal
            #chan_rx_tx_time_ideal：采样点*rx*rx，采样点数等于时延最大的cluster对应的采用点加上一定的扩展数
            #this_chan_aft_bf：rx*tx*cluster: 各个cluster对应的各rx-tx间的时域冲击响应
            chan_rx_tx_time_ideal[rx_idx, tx_idx, :] = np.transpose(channel_time_interpolation(delays, this_chan_aft_bf[rx_idx, tx_idx], 1/fs, num_ext))
            #用信道冲击卷积时域信号，得到过时域信道的时域信号
            tmp_signal = np.convolve(chan_rx_tx_time_ideal[rx_idx, tx_idx, :], input_signal[:, tx_idx]) #时域信号卷积时域信道
            #将各个tx的时域信号合并起来，得到各个rx收到的时域信号;长度30720的数据与长度45的信道卷积得到长度30764的输出信号，取前30720个值
            signal_aft_channel[:, rx_idx] += tmp_signal[0 : input_signal.shape[0]] 
            
    #chan_rx_tx_freq_ideal = np.fft.fftshift(np.fft.fft(chan_rx_tx_time_ideal, num_fft, axis = 2), axes = 2)
    # chan_rx_tx_freq_ideal = np.fft.fftshift(sp.fft(chan_rx_tx_time_ideal, num_fft, axis = 2), axes = 2)
    # chan_rx_tx_time_ideal_test1= sp.ifft(np.fft.ifftshift(chan_rx_tx_freq_ideal, axes = 2), axis=2)#*np.sqrt(chan_rx_tx_freq_ideal.shape[2])
    chan_rx_tx_freq_ideal = np.fft.fftshift(sp.fft.fft(chan_rx_tx_time_ideal, num_fft, axis = 2), axes = 2) #[WY]
    chan_rx_tx_time_ideal_test1= sp.fft.ifft(np.fft.ifftshift(chan_rx_tx_freq_ideal, axes = 2), axis=2)  #[WY]
    #normalization the channel across UE band    
    chan_rx_tx_freq_ideal_UEband = chan_rx_tx_freq_ideal[:,:,IdxSC.T[0]]
    channel_norm = np.average(chan_rx_tx_freq_ideal_UEband*chan_rx_tx_freq_ideal_UEband.conj(), axis=(0,1,2))
    signal_aft_channel=signal_aft_channel/np.sqrt(channel_norm)/np.sqrt(10**(RSRP_gap/10))
    chan_rx_tx_time_ideal=chan_rx_tx_time_ideal/np.sqrt(channel_norm)/np.sqrt(10**(RSRP_gap/10))
    chan_rx_tx_freq_ideal=chan_rx_tx_freq_ideal/np.sqrt(channel_norm)/np.sqrt(10**(RSRP_gap/10))

    return signal_aft_channel, chan_rx_tx_time_ideal, chan_rx_tx_freq_ideal



### This function generate the time domain channel response based on the
### input channel parameters and the antenna configuration.
def channel_response_generate(channel, initial_time):
    ## initialization
    num_cluster = channel['num_cluster']        # actual output cluster number
    pwr = channel['pwr']         # linear value
    delays = channel['delays']

    wave_length = 3e8/(channel['fc_GHz']*1e9) # wave length
    ## parameters of antenna elements
    tx_panel_Mg = channel['tx_panel_Mg']                # number of panels in z-axis demension, Mg in [38.901, Figure 7.3-1], including polarization
    tx_panel_Ng = channel['tx_panel_Ng']                # number of panels in y-axis demension, Ng in [38.901, Figure 7.3-1]
    tx_panel_M = channel['tx_panel_M']                  # number of elements in z-axis demension, M in [38.901, Figure 7.3-1]
    tx_panel_N = channel['tx_panel_N']                  # number of elements in y-axis demension, N in [38.901, Figure 7.3-1]
    tx_panel_P = channel['tx_panel_P']                  # number of polarization, P in [38.901, Figure 7.3-1] 

    dgH_tx = channel['tx_panel_dgH'] * wave_length      # space between tx panels in y-axis demension, horizontal
    dgV_tx = channel['tx_panel_dgV'] * wave_length      # space between tx panels in z-axis demension, vertical
    dy_tx = channel['tx_panel_dH'] * wave_length        # space between tx elements in y-axis demension, horizontal
    dz_tx = channel['tx_panel_dV'] * wave_length        # space between tx elements in z-axis demension, vertical

    rx_panel_Mg = channel['rx_panel_Mg']                # number of panels in z-axis demension, Mg in [38.901, Figure 7.3-1], including polarization
    rx_panel_Ng = channel['rx_panel_Ng']                # number of panels in y-axis demension, Ng in [38.901, Figure 7.3-1]
    rx_panel_M = channel['rx_panel_M']                  # number of elements in z-axis demension, M in [38.901, Figure 7.3-1]
    rx_panel_N = channel['rx_panel_N']                  # number of elements in y-axis demension, N in [38.901, Figure 7.3-1]
    rx_panel_P = channel['rx_panel_P']                  # number of polarization, P in [38.901, Figure 7.3-1] 
 
    dgH_rx = channel['rx_panel_dgH'] * wave_length      # space between rx panels in y-axis demension, horizontal
    dgV_rx = channel['rx_panel_dgV'] * wave_length      # space between rx panels in z-axis demension, vertical
    dy_rx = channel['rx_panel_dH'] * wave_length        # space between rx elements in y-axis demension, horizontal
    dz_rx = channel['rx_panel_dV'] * wave_length        # space between rx elements in z-axis demension, vertical

    ## velocity and subcluster angles
    aoas = channel['aoa']
    aods = channel['aod']
    zoas = channel['zoa']
    zods = channel['zod']
    # pow = zeros(num_cluster,1)
    v_dir = channel['v_dir']
    v_bar = channel['v']*np.array([sind(v_dir[0])*cosd(v_dir[1]), sind(v_dir[0])*sind(v_dir[1]), cosd(v_dir[0])])

    #### calculate the actual location of each antenna element
    Ntx = tx_panel_N * tx_panel_M * tx_panel_P * tx_panel_Mg * tx_panel_Ng      # number of all the tx elements of all panels
    Nrx = rx_panel_N * rx_panel_M * rx_panel_P * rx_panel_Mg * rx_panel_Ng      # number of all the rx elements of all panels
    Ntx_panel = tx_panel_N * tx_panel_M * tx_panel_P                            # number of the tx elements per panel 
    Nrx_panel = rx_panel_N * rx_panel_M * rx_panel_P                            # number of the rx elements per panel 

    ## mapping order for the antenna elements: 
    ## step 1. mapping the elements in the first polarization in a panel with y-axis (H) first z-axis (V) next; 
    ## step 2. mapping the elements in the second polarization in a panel with y-axis (H) first z-axis (V) next; 
    ## step 3. mapping the remaining panels with y-axis (H) first first and z-axis (V) next, following step 1, step 2; 
    bs_loc = np.zeros([3, Ntx])
    for row_idx in range(tx_panel_Mg):
        for col_idx in range(tx_panel_Ng): 
            start_pos = (row_idx*tx_panel_Ng+col_idx) * Ntx_panel
            bs_loc[1, start_pos : start_pos+Ntx_panel] = dy_tx * np.mod(np.arange(0, Ntx_panel), tx_panel_N) + col_idx * (dgH_tx+dy_tx*tx_panel_N)
            bs_loc[2, start_pos : start_pos+Ntx_panel] = dz_tx * np.mod(np.floor(np.arange(0, Ntx_panel) / tx_panel_N), tx_panel_M) + row_idx * (dgV_tx+dz_tx*tx_panel_M)
    bs_loc = bs_loc.transpose()

    ue_loc = np.zeros([3, Nrx])
    for row_idx in range(rx_panel_Mg):
        for col_idx in range(rx_panel_Ng): 
            start_pos = (row_idx*rx_panel_Ng+col_idx) * Nrx_panel
            ue_loc[1, start_pos : start_pos+Nrx_panel] = dy_rx * np.mod(np.arange(0, Nrx_panel), rx_panel_N) + col_idx * (dgH_rx+dy_rx*rx_panel_N)
            ue_loc[2, start_pos : start_pos+Nrx_panel] = dz_rx * np.mod(np.floor(np.arange(0, Nrx_panel) / rx_panel_N), rx_panel_M) + row_idx * (dgV_rx+dz_rx*rx_panel_M)
    ue_loc = ue_loc.transpose()
    
    h_rx_tx =np.zeros((num_cluster, Nrx, Ntx), dtype='complex64') # initialization of the final channel response
    for cluster_idx in range(num_cluster):
        num_subcluster = 20
        path_powers = pwr[cluster_idx]

        h_per_subcluster = 0
        for subcluster_idx in range(num_subcluster): # loop for all the subclusters in the current cluster
            aoa = aoas[cluster_idx][subcluster_idx]
            aod = aods[cluster_idx][subcluster_idx]
            zoa = zoas[cluster_idx][subcluster_idx]
            zod = zods[cluster_idx][subcluster_idx]
            tx_ant_gain = channel['tx_ant_gain'][cluster_idx][subcluster_idx]
            rx_ant_gain = channel['rx_ant_gain'][cluster_idx][subcluster_idx]

            # Medel-2 in TR 36.873
            if tx_panel_P == 2: 
                tx_pattern = tx_ant_gain * np.array([[cosd(channel['tx_slant'][0]), cosd(channel['tx_slant'][1])],
                    [sind(channel['tx_slant'][0]), sind(channel['tx_slant'][1])]])
            else:
                tx_pattern = tx_ant_gain * np.array([[cosd(channel['tx_slant'][0])], [sind(channel['tx_slant'][0])]])
    
            if rx_panel_P == 2:
                rx_pattern = rx_ant_gain * np.array([[cosd(channel['rx_slant'][0]), cosd(channel['rx_slant'][1])],
                    [sind(channel['rx_slant'][0]), sind(channel['rx_slant'][1])]])
            else:
                rx_pattern = rx_ant_gain * np.array([[cosd(channel['rx_slant'][0])], [sind(channel['rx_slant'][0])]])

            xpr = channel['xpr_mat'][cluster_idx][subcluster_idx]
            phase_vv = channel['phase_vv'][cluster_idx][subcluster_idx]
            phase_hh = channel['phase_hh'][cluster_idx][subcluster_idx]
            phase_vh = channel['phase_vh'][cluster_idx][subcluster_idx]
            phase_hv = channel['phase_hv'][cluster_idx][subcluster_idx]
            rand_phase = np.array([[phase_vv, np.sqrt(1/xpr)*phase_vh], [np.sqrt(1/xpr)*phase_hv, phase_hh]])

            pow_per_subcluster = path_powers/num_subcluster

            rx_n_m_hat = np.array([[sind(zoa)*cosd(aoa)], [sind(zoa)*sind(aoa)], [cosd(zoa)]])
            tx_n_m_hat = np.array([[sind(zod)*cosd(aod)], [sind(zod)*sind(aod)], [cosd(zod)]])

            v_n_m = np.sum(rx_n_m_hat.transpose() * v_bar) / wave_length
            doppler_phase = np.exp(1j*2*np.pi*v_n_m*initial_time)
            rx_phase = np.array([np.exp(1j*2*np.pi/wave_length*(np.sum(rx_n_m_hat.transpose() * ue_loc, 1)))])
            tx_phase = np.array([np.exp(1j*2*np.pi/wave_length*(np.sum(tx_n_m_hat.transpose() * bs_loc, 1)))])

            pattern_element = np.matmul(np.matmul(rx_pattern.transpose(), rand_phase), tx_pattern)
            pattern_panel = np.kron(pattern_element, np.ones((Nrx_panel//rx_panel_P, Ntx_panel//tx_panel_P)))
            pattern_rx_and_tx = np.kron(np.ones((rx_panel_Mg*rx_panel_Ng, tx_panel_Mg*tx_panel_Ng)), pattern_panel)
            h_per_subcluster += np.sqrt(pow_per_subcluster) * pattern_rx_and_tx * (rx_phase.transpose() * tx_phase) * doppler_phase
            #h_per_subcluster=h_per_subcluster + h_per_subcluster
            
        h_rx_tx[cluster_idx] = h_per_subcluster
    
    if channel['is_nlos'] == 0:  # if LOS, add the LOS response to the first culster
        aoa = channel['los_aoa']
        aod = channel['los_aod']
        zoa = channel['los_zoa']
        zod = channel['los_zod']

        los_power = channel['los_power']
        tx_los_ant_gain = channel['tx_los_ant_gain']
        rx_los_ant_gain = channel['rx_los_ant_gain']

        # Medel-2 in TR 36.873
        if tx_panel_P == 2:
            tx_pattern_los = tx_los_ant_gain * np.array([[cosd(channel['tx_slant'][0]), cosd(channel['tx_slant'][1])],
                [sind(channel['tx_slant'][0]), sind(channel['tx_slant'][1])]])
        else:
            tx_pattern_los = tx_los_ant_gain * np.array([[cosd(channel['tx_slant'][0])], [sind(channel['tx_slant'][0])]])
        
        if rx_panel_P == 2:
            rx_pattern_los = rx_los_ant_gain * np.array([[cosd(channel['rx_slant'][0]), cosd(channel['rx_slant'][1])],
                [sind(channel['rx_slant'][0]), sind(channel['rx_slant'][1])]])
        else:
            rx_pattern_los = rx_los_ant_gain * np.array([[cosd(channel['rx_slant'][0])], [sind(channel['rx_slant'][0])]])
        
        phase_vv_los = channel['phase_vv_los']
        rand_phase_los = np.array([[phase_vv_los, 0+0*1j], [0+0*1j, -phase_vv_los]], dtype='complex128')

        rx_hat_los = np.array([[sind(zoa)*cosd(aoa)], [sind(zoa)*sind(aoa)], [cosd(zoa)]])
        tx_hat_los = np.array([[sind(zod)*cosd(aod)], [sind(zod)*sind(aod)], [cosd(zod)]])

        v_los = np.sum(rx_hat_los.transpose() * v_bar) / wave_length
        doppler_phase_los = np.exp(1j*2*np.pi*v_los*initial_time)
        rx_phase_los = np.array([np.exp(1j*2*np.pi/wave_length*(np.sum(rx_hat_los.transpose() * ue_loc, 1)))])
        tx_phase_los = np.array([np.exp(1j*2*np.pi/wave_length*(np.sum(tx_hat_los.transpose() * bs_loc, 1)))])

        pattern_tx_and_rx_los = np.matmul(np.matmul(rx_pattern_los.transpose(), rand_phase_los), tx_pattern_los)
        pattern_tx_and_rx_los = np.kron(pattern_tx_and_rx_los, np.ones((Nrx//rx_panel_P, Ntx//tx_panel_P)))
        h_los = pattern_tx_and_rx_los * (rx_phase_los.transpose() * tx_phase_los) * doppler_phase_los

        h_rx_tx[0] = h_rx_tx[0] + h_los*np.sqrt(los_power)
        
    #h_rx_tx: cluster* rx_antenna_element * tx_antenna_element: the channel of each antenna element on each cluster
    return h_rx_tx, delays

def sind(deg):
    value = np.sin(deg * np.pi / 180)
    return value

def cosd(deg):
    value = np.cos(deg * np.pi / 180)
    return value


### Calculate the antenna gain at the input angle using Table 7.1-1 in
### TR 36.873.
### The unit of the angles are degree, the output antenna gain is linear
### value.
def calculate_antenna_pattern(tht, phi, tht_3db, phi_3d, sla_v, Am):
    vertical_gain = -np.min([12*((tht-90)**2)/tht_3db**2, sla_v])
    horizontal_gain = -np.min([12*(phi**2)/phi_3d**2, Am])
    ant_gain_db = -np.min([-(horizontal_gain + vertical_gain), Am])
    ant_gain = 10**(ant_gain_db/20)

    return ant_gain

### This function generate the DFT codebook.
def get_dft_codebook(num_ant_ver, num_ant_hor, num_polar, dft_offset=0):
    num_ant_ver_per_port = num_ant_ver
    num_ant_hor_per_port = num_ant_hor

    codebook_ver = dft_codebook_1D(num_ant_ver_per_port, dft_offset)
    codebook_hor = dft_codebook_1D(num_ant_hor_per_port, dft_offset)

    codebook_per_port = np.kron(codebook_ver, codebook_hor)
    # one alternative mehod for kron
    # codebook_per_port = [];
    # for hor_idx = 1:num_ant_hor_per_port
    #    for ver_idx = 1:num_ant_ver_per_port
    #       code_per_beam = kron(codebook_hor(:,hor_idx), codebook_ver(:,ver_idx));
    #       codebook_per_port = [codebook_per_port2, code_per_beam];
    #    end
    # end

    if num_polar == 1: # ULA
        codebook = codebook_per_port
    else:  # XPO
        zero_matrix = np.zeros((num_ant_ver_per_port * num_ant_hor_per_port, num_ant_ver_per_port * num_ant_hor_per_port))
        c1 = np.block([codebook_per_port, zero_matrix])
        c2 = np.block([zero_matrix, codebook_per_port])
        codebook = np.block([[c1], [c2]])

    return codebook

def dft_codebook_1D(num_ant, dft_offset=0):
    codebook = np.zeros((num_ant, num_ant), dtype='complex128')
    for col_idx in range(num_ant):
        for row_idx in range(num_ant):
            phase = -2 * np.pi * (row_idx+dft_offset) * col_idx / num_ant
            # phase = -2 * np.pi * row_idx * col_idx / num_ant
            codebook[row_idx][col_idx] = np.exp(1j*phase) / np.sqrt(num_ant)

    return codebook

### function for channel virtualization (beamforming)
def channel_virtualization(chan_coff, rx_weight, tx_weight):
    num_path = chan_coff.shape[0]
    if rx_weight.ndim == 1:
        rx_dim = 1
    else:
        rx_dim = rx_weight.shape[0]

    if tx_weight.ndim == 1:
        tx_dim = 1
    else:
        tx_dim = tx_weight.shape[0]

    chan_coff_aft_bf = np.zeros((chan_coff.shape[0], rx_dim, tx_dim), dtype='complex')
    for path_idx in range(num_path):
        chan_coff_aft_bf[path_idx] = np.dot(np.dot(rx_weight, chan_coff[path_idx]), tx_weight.T)

    chan_coff_aft_bf = np.moveaxis(chan_coff_aft_bf, 0, -1)
    return chan_coff_aft_bf

### function for channel interpolation
#-------------------------------- input ----------------------------------
# ts             sample time of output channel response, unit: s
# delays         path delay, unit: s
# coff           cofficient of taps, cluster
# num_ext = 4    default 4, based on bound sinc(x) < 1/(pi*x).
#-------------------------------- output ---------------------------------
# h_int          the output channel response after interplation    
#-------------------------------------------------------------------------

def channel_time_interpolation(delays, coff, ts, num_ext = 4):
    #获取各个cluster对应的采用点
    delay_taps = np.array([delays/ts])

    #起始采用点往前扩展num_ext个，截至采用点往后扩展num_ext+1个
    tapidx = np.array([np.arange(np.min(np.floor(delay_taps.min() - num_ext), 0), np.ceil(delay_taps.max()) + num_ext + 1)])

    # alpha_matrix.
    alpha_index = np.kron(delay_taps.T, np.ones(tapidx.shape)) - np.kron(np.ones(delay_taps.T.shape), tapidx)
    alpha_matrix = np.sinc(alpha_index)
    h_int = np.matmul(alpha_matrix.T, coff)
    
    num_taps = (np.ceil(delays.max()/ts) + 2 * num_ext + 1).astype('int')
    h_int_test = np.zeros([num_taps],dtype=complex)
    h_int_test[-h_int.shape[0]::]=h_int

    return h_int_test


### This function compute the frequency response of the given time domain channel.
def channel_frequency_response(delays, coff, fc, num_sub_carr, scs):
    num_cluster = len(coff)
    freq_range = np.arange(-num_sub_carr / 2 + 1, num_sub_carr / 2 + 1) * scs + fc
    h_rx_tx_freq = np.zeros((num_sub_carr, 1), dtype='complex')
    for freq_idx in range(num_sub_carr): 
        frquency = freq_range[freq_idx]
        tmp_resp = np.zeros(num_cluster, dtype='complex')
        for cluster_idx in range(num_cluster):
            phase = -2 * np.pi * frquency * delays[cluster_idx]
            tmp_resp[cluster_idx] = np.exp(1j*phase) * coff[cluster_idx]
        
        h_rx_tx_freq[freq_idx] = np.sum(tmp_resp)
    
    return h_rx_tx_freq.flatten()
