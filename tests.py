import scipy.signal
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

file_name="tono.wav"
fs,wav=wavfile.read(file_name)
fs_resamp=2000000
resamp_wav=scipy.signal.resample(wav,fs_resamp)

resamp_n=len(resamp_wav)
peaks=scipy.signal.find_peaks(resamp_wav)
t_resamp=np.linspace(0,1,len(resamp_wav))
t=np.linspace(0,1,len(wav))
fc=85000
#plt.plot(wav)
#plt.show()

ang1=np.multiply(t_resamp,2*np.pi*fc)
c1=np.cos(ang1)
ang2=np.multiply(t_resamp,2*np.pi*fc)+resamp_wav*2.5e-4
c2=np.cos(ang2)

f0, Pxx_den0 = scipy.signal.periodogram(resamp_wav, fs_resamp)
f1, Pxx_den1 = scipy.signal.periodogram(c1, fs_resamp)
f2, Pxx_den2 = scipy.signal.periodogram(c2, fs_resamp)


fig, axs = plt.subplots(3)
axs[0].semilogy(f0, Pxx_den0)
axs[0].set_ylim([1, 1e9])
axs[0].set_xlim([300, 700])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f1, Pxx_den1)
axs[1].set_ylim([1e-14, 1])
axs[1].set_xlim([70e3, 100e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()

axs[2].semilogy(f2, Pxx_den2)
axs[2].set_ylim([1e-14, 1])
axs[2].set_xlim([70e3, 100e3])
axs[2].set_xlabel('frequency [Hz]')
axs[2].set_ylabel('PSD [V**2/Hz]')
axs[2].grid()


plt.show()



peaks_c=scipy.signal.find_peaks(c)
print(peaks_c)
#print(peaks)
#print(resamp_n)
