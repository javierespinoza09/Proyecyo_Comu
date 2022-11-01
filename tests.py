import scipy.signal
import scipy.integrate as integral
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

file_name="vowel_1.wav"
fs,wav=wavfile.read(file_name)
fs_resamp=2000000
resamp_wav=scipy.signal.resample(wav,fs_resamp)

resamp_n=len(resamp_wav)
peaks=scipy.signal.find_peaks(resamp_wav)
t_resamp=np.linspace(0,1,fs_resamp)
t=np.linspace(0,1,fs)
fc=85000
#plt.plot(wav)
#plt.show()

ang1=np.multiply(t_resamp,2*np.pi*fc)
c1=np.cos(ang1)
#ang2=np.multiply(t_resamp,2*np.pi*fc)+resamp_wav*0.05e-3
c2=(1+resamp_wav*0.3e-3)*np.cos(np.multiply(t_resamp,2*np.pi*fc))

f0, Pxx_den0 = scipy.signal.periodogram(resamp_wav, fs_resamp)
f1, Pxx_den1 = scipy.signal.periodogram(c1, fs_resamp)
f2, Pxx_den2 = scipy.signal.periodogram(c2, fs_resamp)


fig, axs = plt.subplots(2)
axs[0].semilogy(f0, Pxx_den0)
axs[0].set_ylim([1, 1e9])
axs[0].set_xlim([0, 2700])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f1, Pxx_den1)
axs[1].set_ylim([1e-14, 1])
axs[1].set_xlim([70e3, 100e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()




fig, axs = plt.subplots(2)

axs[0].semilogy(f2, Pxx_den2)
axs[0].set_ylim([1e-14, 1])
axs[0].set_xlim([70e3, 100e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].plot(c2)

plt.show()
Ic=integral.simpson(Pxx_den1,f1)
Is=integral.simpson(Pxx_den2,f2)
print(Ic)
print(Is)
print((Is-Ic)/Is)
print(len(Pxx_den2))
print(f2)
carry=0
maxim=0
for i in range(len(f2)):
	if Pxx_den2[i]>maxim:
		maxim=Pxx_den2[i]
		carry=i

print(carry)
#peaks_c=scipy.signal.find_peaks(c1)
#print(peaks_c)
#print(peaks[0])
#print(resamp_wav[4000])
#print(resamp_n)
