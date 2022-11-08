
import math
from math import pi
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
	
	sA=(1+A_xt*0.3e-3)*c1
	sB=(1+B_xt*0.3e-3)*c2
	sC=(1+C_xt*0.3e-3)*c3
	s_t = sA + sB + sC

	f1, Pxx_den1 = scipy.signal.periodogram(c1, fs_resamp)
	I0 = scipy.integrate.simpson(Pxx_den1, f1)
	f1, Pxx_den1 = scipy.signal.periodogram(c2, fs_resamp)
	I1 = scipy.integrate.simpson(Pxx_den1, f1)
	f1, Pxx_den1 = scipy.signal.periodogram(c3, fs_resamp)
	I2 = scipy.integrate.simpson(Pxx_den1, f1)
    
	f4, Pxx_den4 = scipy.signal.periodogram(sA, fs_resamp)
	I3 = scipy.integrate.simpson(Pxx_den4, f4)
	f5, Pxx_den5 = scipy.signal.periodogram(sB, fs_resamp)
	I4 = scipy.integrate.simpson(Pxx_den5, f5)
	f6, Pxx_den6 = scipy.signal.periodogram(sC, fs_resamp)
	I5 = scipy.integrate.simpson(Pxx_den6, f6)
	f7, Pxx_den7 = scipy.signal.periodogram(s_t, fs_resamp)

	print('Eficiencia de Señal 1 = ', + (I3-I0)/I3)
	print('Eficiencia de Señal 2 = ', + (I4-I1)/I4)
	print('Eficiencia de Señal 3 = ', + (I5-I2)/I5)

	
	fig, axs = plt.subplots(4)
	axs[0].semilogy(f4, Pxx_den4)
	axs[0].set_title('Señales moduladas y s_t')
	axs[0].set_ylim([1e-14, 1])
	axs[0].set_xlim([70e3, 100e3])
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
	axs[2].set_ylim([1e-14, 1])
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
    
    
    
    
    
   
	return s_t 

def canal(s_t):

	mu=0;
	sigma=0.1;
    #Se simula un ruido gauseano
	wn = np.random.normal(loc = mu, scale = sigma, size = len(s_t))
	
	f6, Pxx_den6 = scipy.signal.periodogram(wn, len(s_t))
	I0 = scipy.integrate.simpson(Pxx_den6, f6)
    #Se agrega el ruido a la señal transmitida
	s_t_prima = s_t + wn
	fig, axs = plt.subplots(2)
	axs[0].plot( wn)
	axs[0].set_title('n(t)')
	axs[0].grid()

	axs[1].plot( s_t_prima)
	axs[1].set_title('s(t) + n(t)')
	axs[1].set_xlabel('timepo [s*fs]')
	axs[1].grid()
    
    
	f7, Pxx_den7 = scipy.signal.periodogram(s_t_prima, fs_resamp)
    
	f8, Pxx_den8 = scipy.signal.periodogram(s_t, fs_resamp)
	I1 = scipy.integrate.simpson(Pxx_den8, f8)
    
	print('SNR del canal = ', + I1/I0)
    
	fig, axs = plt.subplots(3)
	axs[0].semilogy(f8, Pxx_den8)
	axs[0].set_title('s_t')
	axs[0].set_ylim([1e-14, 1])
	axs[0].set_xlim([70e3, 100e3])
	axs[0].set_ylabel('PSD [V**2/Hz]')
	axs[0].grid()

	axs[1].semilogy(f6, Pxx_den6)
	axs[1].set_title('noise')
	axs[1].set_ylim([1e-14, 1])
	axs[1].set_xlim([70e3, 100e3])
	axs[1].set_ylabel('PSD [V**2/Hz]')
	axs[1].grid()
    
	axs[2].semilogy(f7, Pxx_den7)
	axs[2].set_title('s_t + noise')
	axs[2].set_ylim([1e-14, 1])
	axs[2].set_xlim([70e3, 100e3])
	axs[2].set_xlabel('frequency [Hz]')
	axs[2].set_ylabel('PSD [V**2/Hz]')
	axs[2].grid()
	

	
	return s_t_prima
	

#########################RECEPTOR######################

def PLL(input_signal, Fs, lenght, N):
   zeta = .707  # damping factor
   k = 1
   Bn = 0.01*Fs  #Noise Bandwidth
   K_0 = 1  # NCO gain
   K_d = 1/2  # Phase Detector gain
   K_p = (1/(K_d*K_0))*((4*zeta)/(zeta+(1/(4*zeta)))) * (Bn/Fs)  # Proporcional gain
   K_i = (1/(K_d*K_0))*(4/(zeta+(1/(4*zeta)**2))) * (Bn/Fs)**2  # Integrator gain
   integrator_out = 0
   phase_estimate = np.zeros(lenght)
   e_D = []  # phase-error output
   e_F = []  # loop filter output
   sin_out_n = np.zeros(lenght)
   cos_out_n = np.ones(lenght)
   for n in range(lenght-1):
      # phase detector
      try:
            e_D.append(
               math.atan(input_signal[n] * (cos_out_n[n] + sin_out_n[n])))
      except IndexError:
            e_D.append(0)
      # loop filter
      integrator_out += K_i * e_D[n]
      e_F.append(K_p * e_D[n] + integrator_out)
      # NCO
      try:
            phase_estimate[n+1] = phase_estimate[n] + K_0 * e_F[n]
      except IndexError:
            phase_estimate[n+1] = K_0 * e_F[n]
      sin_out_n[n+1] = -np.sin(2*np.pi*(k/N)*(n+1) + phase_estimate[n])
      cos_out_n[n+1] = np.cos(2*np.pi*(k/N)*(n+1) + phase_estimate[n])


   return(cos_out_n)
   
   


def receptor_PLL(s_t_prima,t_resamp):
	s_t_A_prima = scipy.signal.sosfilt(scipy.signal.butter(25, [70e3, 80e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima)
	s_t_B_prima = scipy.signal.sosfilt(scipy.signal.butter(25, [80e3, 90e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima)
	s_t_C_prima = scipy.signal.sosfilt(scipy.signal.butter(25, [90e3, 100e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima)
    
	s_t_A_prima_2 = s_t_A_prima^2
	f7, Pxx_den7 = scipy.signal.periodogram(s_t_A_prima, fs_resamp)
	f8, Pxx_den8 = scipy.signal.periodogram(s_t_A_prima_2, fs_resamp)
    #C_A = PLL(s_t_A_prima, fs_resamp, len(s_t_A_prima), 1)
    
	fig, axs = plt.subplots(2)
	axs[0].semilogy(f7, Pxx_den7)
	axs[0].set_ylim([1e-14, 10])
	axs[0].set_xlim([0, 10e3])
	axs[0].set_xlabel('frequency [Hz]')
	axs[0].set_ylabel('PSD [V**2/Hz]')
	axs[0].grid()

	axs[1].semilogy(f8, Pxx_den8)
	axs[1].set_ylim([1e-14, 10])
	axs[1].set_xlim([0, 10e3])
	axs[1].set_xlabel('frequency [Hz]')
	axs[1].set_ylabel('PSD [V**2/Hz]')
	axs[1].grid()
    
	m_t_reconstruida =0
	return m_t_reconstruida
    
def receptor_LC(s_t_prima, fs_resamp, t_resamp):
	max_A = 0
	max_B = 0
	max_C = 0
	carry_A = 0
	carry_B = 0
	carry_C = 0
	f7, Pxx_den7 = scipy.signal.periodogram(s_t_prima, fs_resamp)
	##SE BUSCAN LOS PUNTOS DE MAYOR POTENCIA PARA ASUMIRLOS COMO FRECUENCIA DE PORTADORA##
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
	c3=np.cos(ang3)
	
	
	##FILTRO PASA BAJAS PARA CONSERVAR ÚNICAMENTE LA SEÑAL EN BANDA BASE##
	singal_A_BB = scipy.signal.sosfilt(scipy.signal.butter(25, [10, 5e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima*c1)
	singal_B_BB = scipy.signal.sosfilt(scipy.signal.butter(25, [10, 5e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima*c2)
	singal_C_BB = scipy.signal.sosfilt(scipy.signal.butter(25, [10, 5e3], btype = 'bandpass', fs = fs_resamp, output='sos'), s_t_prima*c3)
	
	##FILTRO DE MEDIA MOVIL CON EL FIN DE ELIMINAR EL RUIDO DE BANDA ANGOSTA##
	signal_A_BB_Filtered = scipy.signal.savgol_filter(singal_A_BB, 10, 3)
	signal_B_BB_Filtered = scipy.signal.savgol_filter(singal_B_BB, 10, 3)
	signal_C_BB_Filtered = scipy.signal.savgol_filter(singal_C_BB, 10, 3)

    
	f10, Pxx_den10 = scipy.signal.periodogram(singal_C_BB, fs_resamp)
	f11, Pxx_den11 = scipy.signal.periodogram(singal_B_BB, fs_resamp)
	f12, Pxx_den12 = scipy.signal.periodogram(singal_A_BB, fs_resamp)

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
	
	
	##DOWN SAMPLE##
	
	A_Recovered=scipy.signal.resample(signal_A_BB_Filtered,5285)
	B_Recovered=scipy.signal.resample(signal_B_BB_Filtered,6040)
	C_Recovered=scipy.signal.resample(signal_C_BB_Filtered,6890)
	
	fig, axs = plt.subplots(3)
	axs[0].plot(range(5285), A_Recovered)
	axs[0].set_title('Información Recuperada')
	axs[1].plot(range(6040), B_Recovered)
	axs[2].plot(range(6890), C_Recovered)
	
	return A_Recovered, B_Recovered, C_Recovered
	
def Save_Result(A_Recovered, B_Recovered, C_Recovered): ##SE ALMACENAN LAS SEÑALES RECONSTRUIDAS EN ARCHIVOS .wav##
	fs_default = 24000

	wavfile.write('Result_vowel_1.wav', fs_default, A_Recovered)
	wavfile.write('Result_vowel_2.wav', fs_default, B_Recovered)
	wavfile.write('Result_vowel_3.wav', fs_default, C_Recovered)
	
	
	

############################ Inicio de ejecucion #################################
#Se importan los archivos .wav que serán la información 
file_name_A="vowel_1.wav"		
file_name_B="vowel_2.wav"
file_name_C="vowel_3.wav"
fs,A=wavfile.read(file_name_A)
fs,B=wavfile.read(file_name_B)
fs,C=wavfile.read(file_name_C)
print(fs)
fig, axs = plt.subplots(3)
axs[0].plot(range(len(A)), A)
axs[0].set_title('Información Original')
axs[1].plot(range(len(B)), B)
axs[2].plot(range(len(C)), C)

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

#Por medio de la librería matplotlib.pyplot se genera una impresión a escala logaritmica de la PDF de las señales 
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



s_t=transmisor(x_t,fs_resamp)

s_t_prima=canal(s_t)

A_Recovered, B_Recovered, C_Recovered = receptor_LC(s_t_prima, fs_resamp, t_resamp)
#receptor_PLL(s_t_prima,t_resamp)
Save_Result(A_Recovered, B_Recovered, C_Recovered)

plt.show()
#llamar funcion de receptor
#m_t_reconstruida=receptor(s_t_prima)




