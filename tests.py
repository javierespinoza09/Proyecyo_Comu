import scipy.signal
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

file_name_A="vowel_1.wav"
file_name_B="vowel_2.wav"
file_name_C="vowel_3.wav"
fs,A=wavfile.read(file_name_A)
fs,B=wavfile.read(file_name_B)
fs,C=wavfile.read(file_name_C)

fs_resamp=2000000
t_resamp=np.linspace(0,1,fs_resamp)
resamp_A=scipy.signal.resample(A,fs_resamp)
resamp_B=scipy.signal.resample(B,fs_resamp)
resamp_C=scipy.signal.resample(C,fs_resamp)

resamp_A = np.array(resamp_A)
resamp_B = np.array(resamp_B)
resamp_C = np.array(resamp_C)

x_t = np.array([resamp_A, resamp_B, resamp_C])


f0, Pxx_den0 = scipy.signal.periodogram(resamp_A, fs_resamp)
f1, Pxx_den1 = scipy.signal.periodogram(resamp_B, fs_resamp)
f2, Pxx_den2 = scipy.signal.periodogram(resamp_C, fs_resamp)


fig, axs = plt.subplots(3)

axs[0].semilogy(f0, Pxx_den0)
axs[0].set_ylim([1e-14, 10e7])
axs[0].set_xlim([0, 5e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f1, Pxx_den1)
axs[1].set_ylim([1e-14, 10e7])
axs[1].set_xlim([0, 5e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()

axs[2].semilogy(f2, Pxx_den2)
axs[2].set_ylim([1e-14, 10e7])
axs[2].set_xlim([0, 5e3])
axs[2].set_xlabel('frequency [Hz]')
axs[2].set_ylabel('PSD [V**2/Hz]')
axs[2].grid()

fc1=75000
fc2=85000
fc3=95000
A_xt = x_t[0,:]
B_xt = x_t[1,:]
C_xt = x_t[2,:]
	
ang1=np.multiply(t_resamp,2*np.pi*fc1)
c1=np.cos(ang1)
ang2=np.multiply(t_resamp,2*np.pi*fc2)
c2=np.cos(ang2)
ang3=np.multiply(t_resamp,2*np.pi*fc3)
c3=np.cos(ang3)
	
sA=(1+A_xt*0.3e-3)*c1
sB=(1+B_xt*0.3e-3)*c2
sC=(1+C_xt*0.3e-3)*c3

s_t = sA + sB + sC
	
f4, Pxx_den4 = scipy.signal.periodogram(sA, fs_resamp)
f5, Pxx_den5 = scipy.signal.periodogram(sB, fs_resamp)
f6, Pxx_den6 = scipy.signal.periodogram(sC, fs_resamp)
f7, Pxx_den7 = scipy.signal.periodogram(s_t, fs_resamp)

fig, axs = plt.subplots(4)
axs[0].semilogy(f4, Pxx_den4)
axs[0].set_ylim([1e-14, 10])
axs[0].set_xlim([70e3, 100e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f5, Pxx_den5)
axs[1].set_ylim([1e-14, 10])
axs[1].set_xlim([70e3, 100e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()
	
axs[2].semilogy(f6, Pxx_den6)
axs[2].set_ylim([1e-14, 10])
axs[2].set_xlim([70e3, 100e3])
axs[2].set_xlabel('frequency [Hz]')
axs[2].set_ylabel('PSD [V**2/Hz]')
axs[2].grid()

axs[3].semilogy(f7, Pxx_den7)
axs[3].set_ylim([1e-14, 10])
axs[3].set_xlim([70e3, 100e3])
axs[3].set_xlabel('frequency [Hz]')
axs[3].set_ylabel('PSD [V**2/Hz]')
axs[3].grid()

plt.show()
