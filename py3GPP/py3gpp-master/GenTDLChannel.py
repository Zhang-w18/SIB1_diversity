# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:31:11 2021
 Copyright:  Huawei Technologies Co., Ltd. All rights reserved.
 File name: GenTDLChannel
 
 Description: 
    use to generate TDL channel response
 
 修改记录：
 date name line xxx

@author: x00578758
"""
import numpy as np
#import matplotlib.pyplot as plt

class ChannelInfo:
    def __init__(self, nTx, nRx, Speed, Fc, Fs, ChanType, DS, NFFT, TO=0, FO=0):
    #def __init__(self, nTx, nRx, Speed=3, Fs=4096*120e3, ChanType='TDLA', DS=10, NFFT=4096, TO=0, FO=0):
        assert (nTx > 0)
        assert (nRx > 0)
        assert (Speed > 0)
        
        
        self.nTx = nTx
        self.nRx = nRx
        self.Speed = Speed/3.6              # moving speed, unit: m/s
        self.Fc = Fc                        # carrier frequency, unit: Hz
        #self.Fc = 2e9
        self.Fs = Fs                        # sampling frequency, unit: Hz
        self.NFFT = NFFT                    # size of FFT
        self.timeOff = TO                   # time offset
        self.CFO = FO                       # carrier frequency offset
        self.ChanType = ChanType.upper()    # channel type: TDLA,TDLB,TDLC,TDLD,TDLE
        self.DS = DS                        # delay spread, unit: ns
        self.ChannelTypePara()
        
    def ChannelTypePara(self):
        if self.ChanType.find('TDLA') >= 0:
            '''
            self.DelaysperPath = np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
            0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119,
            3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])*self.DS
            self.PowerperPath = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9,
            6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
            -18.3, -18.9, -16.6, -19.9, -29.7])
            '''
            self.DelaysperPath = np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
            0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119,
            3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043])*self.DS
            self.PowerperPath = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9,
            6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
            -18.3, -18.9, -16.6, -19.9])
            self.first_path_los = 0
        elif self.ChanType.find('TDLB') >= 0:
            self.DelaysperPath = np.array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752,
            0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474,
            1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834])*self.DS
            self.PowerperPath = np.array([0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3,
            -8.9, -9, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8,
            -11.4, -14.9, -9.2, -11.3])
            self.first_path_los = 0
        elif self.ChanType.find('TDLC') >= 0:
            self.DelaysperPath = np.array([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
            0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
            4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])*self.DS
            self.PowerperPath = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1,
            -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
            -17.1, -16, -15.7, -21.6, -22.8])
            self.first_path_los = 0
        elif self.ChanType.find('TDLD') >= 0: # add a LOS path with K factor = 13.3
            self.DelaysperPath = np.array([0, 0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775,
            4.042, 7.937, 9.424, 9.708, 12.525])*self.DS
            self.PowerperPath = np.array([-0.2, -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9,
            -27.8, -23.6, -24.8, -30.0, -27.7])
            self.first_path_los = 1
        elif self.ChanType.find('TDLE') >= 0:# add a LOS path with K factor = 13.3
            self.DelaysperPath = np.array([0, 0.000, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092,
            1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6419])*self.DS
            self.PowerperPath = np.array([-8.73, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6,
            -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2])
            self.first_path_los = 1
        elif self.ChanType == 'RAYLEIGH':
            self.DelaysperPath = np.array([0])
            self.PowerperPath = np.array([0])
            self.first_path_los = 0
        elif self.ChanType == 'AWGN':
            self.DelaysperPath = np.array([0])
            self.PowerperPath = np.array([0])
            self.hf = [1,]
        else:
            print("Not supported channel type!")
            
        self.taps = np.round(self.DelaysperPath*1e-9*self.Fs).astype(int)
        power = 10**(self.PowerperPath/10)
        power = power/sum(power)
        self.taps_power = np.sqrt(power) # amplitude
        self.num_taps = self.taps.size
        self.conv_points = np.unique(self.taps)
        self.DS_tap = np.max(self.taps)+1
        self.doppler = self.Speed*self.Fc/3e8
        
    def gen_Rayleigh(self,num_samples): # can't reflect the influence of speed, random power
        nTRx = self.nTx*self.nRx
        hf = np.zeros([nTRx,self.DS_tap,num_samples],dtype=complex)
        self.total_power = 0
        for isamp in range(num_samples):
            # random a*exp(1j*theta) per ant per tap per sample
            # no time correlation between two samples
            t_hf = (np.random.randn(nTRx,self.num_taps)+1j*np.random.randn(nTRx,self.num_taps))/np.sqrt(2)
            for iant in range(nTRx):
                if self.first_path_los == 1:
                    hf[iant][self.taps[0]][isamp] += self.taps_power[0]
                else:
                    hf[iant][self.taps[0]][isamp] += self.taps_power[0]*t_hf[iant][0]
                for itap in range(1,self.num_taps):
                    hf[iant][self.taps[itap]][isamp] += self.taps_power[itap]*t_hf[iant][itap]
        
        self.time_channel = hf.reshape(self.nRx,self.nTx,self.DS_tap,num_samples)

    def gen_Jakes(self,time_samples):
        # based on the referenece of Yahong Zheng, Chengshan Xiao, IEEE Communications Letterm June 2002
        # "Improved Models for the Generation of Multiple Uncorrelated Rayleigh Fading Waveforms"
        # time_samples: absolute time, can be adjusted according to the time duration of simulation time
        M = 8
        hf_samples = np.zeros([self.nTx*self.nRx,self.DS_tap,time_samples.size],dtype=complex)
        wd_t = 2*np.pi*self.doppler*time_samples
        
        for iant in range(self.nTx*self.nRx):
            for itap in range(self.num_taps):
                alpha = (np.arange(0,M)+np.random.rand(M))*np.pi/(2*M)
                phi = (2*np.random.rand(M)-1)*np.pi
                psi = (2*np.random.rand(M)-1)*np.pi
                for isamp in range(time_samples.size):
                    for k in range(M):
                        hf_samples[iant][self.taps[itap]][isamp] += self.taps_power[itap]*np.cos(wd_t[isamp]*np.cos(alpha[k])+phi[k])
                        hf_samples[iant][self.taps[itap]][isamp] += self.taps_power[itap]*1j*np.cos(wd_t[isamp]*np.sin(alpha[k])+psi[k])
        hf_samples *= np.sqrt(1/M)
        self.time_channel = hf_samples.reshape(self.nRx,self.nTx,self.DS_tap,time_samples.size)

    def gen_Jakes_Accelerate(self, time_samples):
        # based on the referenece of Yahong Zheng, Chengshan Xiao, IEEE Communications Letterm June 2002
        # "Improved Models for the Generation of Multiple Uncorrelated Rayleigh Fading Waveforms"
        # time_samples: absolute time, can be adjusted according to the time duration of simulation time
        # Accelerated by Tangzihan 00585244
        M = 8
        hf_samples = np.zeros([self.nTx * self.nRx, self.DS_tap, time_samples.size], dtype=complex)
        #hf_samples_Accelerate = np.zeros([self.nTx * self.nRx, self.DS_tap, time_samples.size], dtype=complex)
        wd_t = 2 * np.pi * self.doppler * time_samples

        for iant in range(self.nTx * self.nRx):
            for itap in range(self.num_taps):
                alpha = (np.arange(0, M) + np.random.rand(M)) * np.pi / (2 * M)
                phi = (2 * np.random.rand(M) - 1) * np.pi
                psi = (2 * np.random.rand(M) - 1) * np.pi

                # for isamp in range(time_samples.size):
                #    for k in range(M):
                #        hf_samples[iant][self.taps[itap]][isamp] += self.taps_power[itap]*np.cos(wd_t[isamp]*np.cos(alpha[k])+phi[k])
                #        hf_samples[iant][self.taps[itap]][isamp] += self.taps_power[itap]*1j*np.cos(wd_t[isamp]*np.sin(alpha[k])+psi[k])

                kIndex = np.arange(0, M).reshape(M, 1)
                # for isamp in range(time_samples.size):
                #    hf_samples_Accelerate[iant][self.taps[itap]][isamp] += self.taps_power[itap] * (np.sum(
                #        np.cos(np.cos(alpha[kIndex]) * wd_t[isamp] + phi[kIndex]) + 1j * np.cos(
                #            np.sin(alpha[kIndex]) * wd_t[isamp] + psi[kIndex]), axis=0))
                hf_samples[iant][self.taps[itap]][:] += self.taps_power[itap] * (np.sum(np.cos(np.cos(alpha[kIndex]) * wd_t + phi[kIndex]) + 1j * np.cos(np.sin(alpha[kIndex]) * wd_t + psi[kIndex]), axis=0))
                # hf_samples[iant][self.taps[itap]][isamp] = self.taps_power[itap] * np.sum(np.cos(wd_t[isamp] * np.cos(alpha[kIndex]) + phi[kIndex]) + 1j * np.cos(wd_t[isamp] * np.sin(alpha[kIndex]) + psi[kIndex]), axis=0)
        hf_samples *= np.sqrt(1/M)
        #hf_samples_Accelerate *= np.sqrt(1 / M)
        self.time_channel = hf_samples.reshape(self.nRx, self.nTx, self.DS_tap, time_samples.size)
        #self.time_channel_Accelerate = hf_samples_Accelerate.reshape(self.nRx, self.nTx, self.DS_tap, time_samples.size)


    def gen_FreqChannel(self):
        # add by sunyan
        # before reshape: time_channel.shape() = [nRx, nTx, nTap, nSample]
        self.time_channel = self.time_channel.reshape(self.nTx*self.nRx,self.time_channel.shape[2],self.time_channel.shape[3])
        assert(self.time_channel.ndim == 3)
        nTRx = self.time_channel.shape[0]
        numFFT = self.NFFT
        num_samples = self.time_channel.shape[2]
        DS_tap = self.DS_tap
        self.freq_channel = np.zeros([nTRx,numFFT,num_samples],dtype=complex)
        t_vec = np.zeros(numFFT,dtype=complex) # vector
        for iant in range(nTRx):
            for isamp in range(num_samples):
                t_vec[self.timeOff:self.timeOff+DS_tap] = self.time_channel[iant,:,isamp]
                self.freq_channel[iant,:,isamp] = np.fft.fft(t_vec)
        self.freq_channel = self.freq_channel.reshape(self.nRx,self.nTx,self.freq_channel.shape[1],self.freq_channel.shape[2])
        self.time_channel = self.time_channel.reshape(self.nRx,self.nTx,self.time_channel.shape[1],self.time_channel.shape[2])


"""
# example
# parameter initialization
plt.close('all')
N = 5000
SCS = 120e3
Fc = 28e9
DS = 100e-9
NFFT = 4096
Fs = NFFT*SCS
NumTx = 2
NumRx = 2
Speed = 3
TimeDura = N*125e-6
NRB = 64
REstep = 48
CS = 17
# channel initialization
a = ChannelInfo(NumTx,NumRx,Speed,Fc,Fs,'TDLA',DS,NFFT)

# independent multipath channel
a.gen_Rayleigh(N)
total_power = 0
power_samples = np.zeros(N)
for i in range(N):
    power_samples[i] = np.sum(np.abs(a.time_channel[0,0,:,i])**2)
    total_power += power_samples[i]
print("Total power:",total_power/N)

plt.figure('Time domain channel')
plt.stem(power_samples)

a.gen_FreqChannel()

for i in range(0,10):
    plt.figure()
    #plt.plot(np.abs(a.freq_channel[0,0,:,0::500]))
    plt.plot(np.abs(a.freq_channel[0,0,:,i]))

plt.show()



# Jake's model
time_samples = np.linspace(0,TimeDura,N)
a.gen_Jakes(time_samples)
'''
total_power = 0
power_samples = np.zeros(N)
for i in range(N):
    power_samples[i] = np.sum(np.abs(a.time_channel[0,0,:,i])**2)
    total_power += power_samples[i]
print("Total power:",total_power/N)

plt.figure('Time domain channel of Jake-s model')
plt.stem(power_samples)
'''
a.gen_FreqChannel()
plt.figure()
plt.plot(np.abs(a.freq_channel[0,0,:,0::500]))

IdxR15 = (np.arange(0,NRB*12,REstep)+(NFFT-NRB*12)/2+6).astype(int)
IdxBlock = (np.arange(0,17)+NFFT/2-np.ceil(CS/2)).astype(int)
HpowerR15 = np.sum(pow(np.abs(a.freq_channel[:,:,IdxR15,:]),2),2)
HpowerBlock = np.sum(pow(np.abs(a.freq_channel[:,:,IdxBlock,:]),2),2)

plt.figure()
plt.hold('on')
plt.plot(HpowerBlock[0,0,:])
plt.plot(HpowerR15[0,0,:])
plt.figure()
plt.hold('on')
plt.plot(HpowerBlock[0,1,:])
plt.plot(HpowerR15[0,1,:])
plt.figure()
plt.hold('on')
plt.plot(HpowerBlock[1,0,:])
plt.plot(HpowerR15[1,0,:])
plt.figure()
plt.hold('on')
plt.plot(HpowerBlock[1,1,:])
plt.plot(HpowerR15[1,1,:])
plt.figure()
plt.hold('on')
plt.plot(np.sum(np.sum(HpowerBlock,0),0))
plt.plot(np.sum(np.sum(HpowerR15,0),0))

"""