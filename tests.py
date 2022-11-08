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

mu=0;
sigma=0.001;
    #Se simula un ruido gauseano
wn = np.random.normal(loc = mu, scale = sigma, size = len(s_t))
    #Se agrega el ruido a la señal transmitida
s_t_prima = s_t + wn
    
f7, Pxx_den7 = scipy.signal.periodogram(s_t_prima, fs_resamp)
f8, Pxx_den8 = scipy.signal.periodogram(s_t, fs_resamp)
    
fig, axs = plt.subplots(2)
axs[0].semilogy(f7, Pxx_den7)
axs[0].set_ylim([1e-17, 10])
axs[0].set_xlim([70e3, 100e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f8, Pxx_den8)
axs[1].set_ylim([1e-14, 1])
axs[1].set_xlim([70e3, 100e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()

frec_range_A = np.arange(74e3, 76e3, 1)
frec_range_B = np.arange(84e3, 86e3, 1)
frec_range_C = np.arange(94e3, 96e3, 1)


max_A = 0
max_B = 0
max_C = 0
carry_A = 0
carry_B = 0
carry_C = 0
	

	
for i in range(len(Pxx_den7)):
	if i>74e3 and i<76e3:
		if Pxx_den7[i] > max_A:
			max_A = Pxx_den7[i]
			carry_A = i
	elif i>84e3 and i<86e3:
		if Pxx_den7[i] > max_B:
			max_B = Pxx_den7[i]
			carry_B = i
	elif i>94e3 and i<96e3:
		if Pxx_den7[i] > max_C:
			max_C = Pxx_den7[i]
			carry_C = i
			

    	
print(carry_A) 	
print(carry_B) 
print(carry_C) 
ang1=np.multiply(t_resamp,2*np.pi*carry_A)
c1=np.cos(ang1)
ang2=np.multiply(t_resamp,2*np.pi*carry_B)
c2=np.cos(ang2)
ang3=np.multiply(t_resamp,2*np.pi*carry_C)
c2=np.cos(ang2)

singal_A_BB = s_t_prima*c1
singal_B_BB = s_t_prima*c2
singal_C_BB = s_t_prima*c3

f10, Pxx_den10 = scipy.signal.periodogram(singal_A_BB, fs_resamp)
f11, Pxx_den11 = scipy.signal.periodogram(singal_B_BB, fs_resamp)
f12, Pxx_den12 = scipy.signal.periodogram(singal_C_BB, fs_resamp)

fig, axs = plt.subplots(3)
axs[0].semilogy(f10, Pxx_den10)
axs[0].set_ylim([1e-14, 10])
axs[0].set_xlim([0, 10e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f11, Pxx_den11)
axs[1].set_ylim([1e-14, 10])
axs[1].set_xlim([0, 10e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()
	
axs[2].semilogy(f12, Pxx_den12)
axs[2].set_ylim([1e-14, 10])
axs[2].set_xlim([0, 10e3])
axs[2].set_xlabel('frequency [Hz]')
axs[2].set_ylabel('PSD [V**2/Hz]')
axs[2].grid()

b, a = scipy.signal.butter(20, 1000, 'low', analog=True)
y = scipy.signal.filtfilt(b, a, singal_A_BB, axis=0)

f12, Pxx_den12 = scipy.signal.periodogram(y, fs_resamp)
fig, axs = plt.subplots(2)
axs[0].semilogy(f12, Pxx_den12)
axs[0].set_ylim([1e-14, 10])
axs[0].set_xlim([0, 10e3])
axs[0].set_xlabel('frequency [Hz]')
axs[0].set_ylabel('PSD [V**2/Hz]')
axs[0].grid()

axs[1].semilogy(f11, Pxx_den11)
axs[1].set_ylim([1e-14, 10])
axs[1].set_xlim([0, 10e3])
axs[1].set_xlabel('frequency [Hz]')
axs[1].set_ylabel('PSD [V**2/Hz]')
axs[1].grid()



plt.show()
