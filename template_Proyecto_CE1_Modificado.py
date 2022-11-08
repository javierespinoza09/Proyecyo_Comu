

import scipy.signal
from scipy.io import wavfile 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


#definicion de 3 bloques principales: TX, canal y RX

def transmisor(x_t,fs_resamp):
	fc1 = 75000
	fc2 = 85000
    	fc3 = 95000
    	
	A_xt = x_t[0,:]
    	B_xt = x_t[1,:]
    	C_xt = x_t[2,:]
	
	ang1=np.multiply(t_resamp,2*np.pi*fc1)
	c1=np.cos(ang1)
	ang2=np.multiply(t_resamp,2*np.pi*fc2)
	c2=np.cos(ang2)
	ang3=np.multiply(t_resamp,2*np.pi*fc3)
	c3=np.cos(ang3)
	
	sA=(A_xt*0.3e-3)*c1
	sB=(B_xt*0.3e-3)*c2
	sC=(C_xt*0.3e-3)*c3
	
	f4, Pxx_den4 = scipy.signal.periodogram(sA, fs_resamp)
	f5, Pxx_den5 = scipy.signal.periodogram(sB, fs_resamp)
	f6, Pxx_den6 = scipy.signal.periodogram(sC, fs_resamp)

	fig, axs = plt.subplots(3)
	axs[0].semilogy(f4, Pxx_den)
	axs[0].set_ylim([1, 1e9])
	axs[0].set_xlim([0, 2700])
	axs[0].set_xlabel('frequency [Hz]')
	axs[0].set_ylabel('PSD [V**2/Hz]')
	axs[0].grid()

	axs[1].semilogy(f5, Pxx_den5)
	axs[1].set_ylim([1e-14, 1])
	axs[1].set_xlim([70e3, 100e3])
	axs[1].set_xlabel('frequency [Hz]')
	axs[1].set_ylabel('PSD [V**2/Hz]')
	axs[1].grid()
	
	axs[2].semilogy(f6, Pxx_den6)
	axs[2].set_ylim([1e-14, 10e7])
	axs[2].set_xlim([0, 5e3])
	axs[2].set_xlabel('frequency [Hz]')
	axs[2].set_ylabel('PSD [V**2/Hz]')
	axs[2].grid()

	return s_t 

def canal(s_t):

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
	axs[0].set_ylim([1, 1e9])
	axs[0].set_xlim([0, 2700])
	axs[0].set_xlabel('frequency [Hz]')
	axs[0].set_ylabel('PSD [V**2/Hz]')
	axs[0].grid()

	axs[1].semilogy(f8, Pxx_den8)
	axs[1].set_ylim([1e-14, 1])
	axs[1].set_xlim([70e3, 100e3])
	axs[1].set_xlabel('frequency [Hz]')
	axs[1].set_ylabel('PSD [V**2/Hz]')
	axs[1].grid()
	
    	plt.plot()
    	return s_t_prima


def receptor(s_t_prima):

	m_t_reconstruida=s_t_prima 

    	return m_t_reconstruida



############################ Inicio de ejecucion #################################
#Se importan los archivos .wav que serán la información 
file_name_A="vowel_1.wav"		
file_name_B="vowel_2.wav"
file_name_C="vowel_3.wav"
fs,A=wavfile.read(file_name_A)
fs,B=wavfile.read(file_name_B)
fs,C=wavfile.read(file_name_C)

#Es necesario realizar modificar la frecuencia de sample para evitar aliasing al momento de la modulación
fs_resamp=2000000
t_resamp=np.linspace(0,1,fs_resamp)
resamp_A=scipy.signal.resample(A,fs_resamp)
resamp_B=scipy.signal.resample(B,fs_resamp)
resamp_C=scipy.signal.resample(C,fs_resamp)

resamp_A = np.array(resamp_A)
resamp_B = np.array(resamp_B)
resamp_C = np.array(resamp_C)

#Se crea un array con los datos con la frecuencia de muestreo modificada
x_t = np.array([resamp_A, resamp_B, resamp_C])

#Utilizando la función de periodgram se obtiene la PFD de las señales de información
f0, Pxx_den0 = scipy.signal.periodogram(resamp_A, fs_resamp)
f1, Pxx_den1 = scipy.signal.periodogram(resamp_B, fs_resamp)
f2, Pxx_den2 = scipy.signal.periodogram(resamp_C, fs_resamp)
f3, Pxx_den3 = scipy.signal.periodogram(C_xt, fs_resamp)

#Por medio de la librería matplotlib.pyplot se genera una impresión a escala logaritmica de la PDF de las señales 
fig, axs = plt.subplots(4)

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

axs[3].semilogy(f3, Pxx_den3)
axs[3].set_ylim([1e-14, 10e7])
axs[3].set_xlim([0, 5e3])
axs[3].set_xlabel('frequency [Hz]')
axs[3].set_ylabel('PSD [V**2/Hz]')
axs[3].grid()


s_t=transmisor(x_t,fs_resamp)

s_t_prima=canal(s_t)

plt.show()
#llamar funcion de receptor
#m_t_reconstruida=receptor(s_t_prima)




