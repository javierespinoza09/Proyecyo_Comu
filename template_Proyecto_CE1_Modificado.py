# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:06:24 2022

Plantilla para Proyecto. 
Curso Comunicaciones Electricas 1. 
Sistema de transmisión y recepción analógica

@author: lcabrera
"""

#importar bibliotecas utiles. De no tenerse alguna (import not found) se debe instalar, generalmente con pip
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
f3, Pxx_den3 = scipy.signal.periodogram(C_xt, fs_resamp)

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

#definicion de 3 bloques principales: TX, canal y RX

def transmisor(x_t,fs_resamp):
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


	plt.show()
	
    
    return s_t #note que s_t es una unica señal utilizando un unico array, NO una lista

def canal(s_t):
    
    #Note que los parámetros mu (media) y sigma (desviacion) del ruido blanco Gaussiano deben cambiarse segun especificaciones
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
	rango_A = np.linspace(74e3, 76e3, 2000)
	rango_B = np.linspace(84e3, 86e3, 2000)
	rango_C = np.linspace(94e3, 96e3, 2000)
	carry_A = 0
	carry_B = 0
	carry_C = 0
	
	f9, Pxx_den9 = scipy.signal.periodogram(s_t_prima, fs_resamp)
	
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
    	
    	
    	ang1=np.multiply(t_resamp,2*np.pi*carry_A)
	c1=np.cos(ang1)
	ang2=np.multiply(t_resamp,2*np.pi*carry_B)
	c2=np.cos(ang2)
	ang3=np.multiply(t_resamp,2*np.pi*carry_C)
	c2=np.cos(ang2)
    	
    	singal_A_BB = s_t_prima*c1
    	singal_B_BB = s_t_prima*c2
    	singal_C_BB = s_t_prima*c3
    
    	
    # Note que f_rf es la frecuencia utilizada para la seleccionar la señal que se desea demodular
    
    #Su codigo para el receptor va aca  
       
    
    m_t_reconstruida=s_t_prima #eliminar cuando se tenga solucion propuesta
    
    #note que en el caso de multiples señales
    
    return m_t_reconstruida



## Inicio de ejecucion ##
#Se da con ejemplo de tono, pasandolo por todo el sistema sin ningun cambio

#leer tono desde archivo
samplerate_tono, tono = wavfile.read("datos/tono.wav")

#oir tono rescatado. Esta funcion sirve tambien como transductor de salida 
#Note la importancia de la frecuencia de muestreo (samplerate), la cual es diferente a la frecuencia fm del tono.
sd.play(tono, samplerate_tono)

#graficar tono
plt.plot(np.linspace(0., tono.shape[0] / samplerate_tono, tono.shape[0]),tono)
plt.xlim([0, 0.01]) #mostrar solo parte de la onda


#agregar el tono a la lista X_t requerida por el transmisor
x_t=[]  #solo para ejemplo, crear lista con el mismo tono 3 veces
x_t.append(tono)
x_t.append(tono)
x_t.append(tono)
print("Se envia una lista con "+str(len(x_t))+" señales")


#llamar funcion de transmisor
s_t=transmisor(wav,fs)


#llamar funcion que modela el canal
s_t_prima=canal(s_t)

#llamar funcion de receptor
m_t_reconstruida=receptor(s_t_prima)

#graficar señal recibida
plt.plot(np.linspace(0., m_t_reconstruida.shape[0] / samplerate_tono, m_t_reconstruida.shape[0]),m_t_reconstruida)
plt.xlim([0, 0.01]) #mostrar solo parte de la onda


