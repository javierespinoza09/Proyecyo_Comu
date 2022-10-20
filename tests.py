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
ang=np.multiply(t_resamp,2*np.pi*fc)+resamp_wav*5e-5
#ang=np.multiply(t,2*np.pi*fc)
c=np.cos(ang)
f, Pxx_den = scipy.signal.periodogram(c, fs_resamp)
#plt.plot(c)
plt.semilogy(f, Pxx_den)
plt.ylim([1e-14, 1])
plt.xlim([70e3, 100e3])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid(True)
plt.show()
peaks_c=scipy.signal.find_peaks(c)
print(peaks_c)
#print(peaks)
#print(resamp_n)
