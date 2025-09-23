# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:22:58 2025

@author: Augusto
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal import windows

def mi_funcion_sin2(vmax=1.0, dc=0.0, ff=1.0, ph=0.0, nn=1000, fs=1000.0):
 t = np.arange(nn, dtype=float)/fs
 f = np.atleast_1d(ff).astype(float)
 ph = np.broadcast_to(np.atleast_1d(ph), f.shape)
 A  = np.broadcast_to(np.atleast_1d(vmax), f.shape)
 DC = np.broadcast_to(np.atleast_1d(dc), f.shape)


 x = DC[None,:] + A[None,:] * np.sin(2*np.pi*t[:,None]*f[None,:] + ph[None,:])

 return t, x

#cte
N=1000 #numero de muestras 
fs=N   #Frecuencia de muestreo
dF=fs/N#Resolucion espectral 
ts=1/fs# Tiempo de muestreo 
Ω0=fs/4# Frecuencia en el primer bin 
Ab=fs/2 #Ancho de banda
R=200 #Realizaciones 
snr=3 #Piso de ruido
amplitud=np.sqrt(2) 
sigma=np.sqrt(10**(-snr/10)*(amplitud)**2/2) #Sigma por definicion
nfft=N #Numero de frecuencias para la FFT 

#Definicion de ventanas.
wH=  np.hanning(N)       #Hanning window
wBH= np.blackman(N)      #Blackman window
wFT= windows.flattop(N)  #flat top window

#Generadores de ruido
fr=np.random.uniform(low=-2, high=2, size=R).ravel() #Generador aleatorio entre 2 y -2
namatrizada= np.random.normal(loc=0.0, scale=sigma, size=(N, R)) #Generador normal

Ω1=(Ω0+fr)*dF #Frecuencia con ruido

ttmatrizada,xxmatrizada=mi_funcion_sin2(vmax=amplitud, dc=0, ff=Ω1, ph=0, nn=N, fs=fs) #Senoideales en matriz con ruido en frecuencia
Xxruido=xxmatrizada+namatrizada #senoideal con ruido 
plt.figure()
plt.plot(ttmatrizada,Xxruido) #Grafico de senoideales con ruido en frecuencia y desfase 
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Figura [1]:Senoideales con ruido')

ffm=np.fft.fftfreq(nfft, ts) #Tira de frecuencias para escala dB

#Senoideales ventaneadas
xxRec=Xxruido    #usando ventana rectangular implicita 
xxwH= Xxruido*wH[:, None] #usando ventana de hanning
xxwBH=Xxruido*wBH[:, None]#usando ventana de Blackman
xxwFT=Xxruido*wFT[:, None]#usando ventana de flat top

#FFT de las senoideales 

FFTxxRec=fft(xxRec,axis=0)/N  #Fft de la  ventana rectangular implicita 
FFTxxwH=fft(xxwH,axis=0)/N  #Fft de la  ventana hanning
FFTxxBH=fft(xxwBH,axis=0)/N  #Fft de la  ventana Blackman 
FFTxxFT=fft(xxwFT,axis=0)/N  #Fft de la  ventana flat top

#FFT en DB 
dBRec=20*np.log10(np.abs(FFTxxRec)) #Fft en escala dB de la  ventana rectangular implicita 
dBwH=20*np.log10(np.abs(FFTxxwH))   #Fft en escala dB de la  ventana hanning
dBBH=20*np.log10(np.abs(FFTxxBH))   #Fft en escala dB de la  ventana Blackman 
dBFT=20*np.log10(np.abs(FFTxxFT))   #Fft en escala dB de la  ventana flat top

#Grafico de las senoideales ventaneadas 
plt.figure()
plt.plot(ffm,dBRec,'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [2]:Grafico de la FFT de una senoideal con ventana rectangular')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm,dBwH,'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [3]:Grafico de la FFT de una senoideal con ventana hanning')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm,dBBH,'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [4]:Grafico de la FFT de una senoideal con ventana Blackman')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm,dBFT,'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [5]:Grafico de la FFT de una senoideal con ventana flat top')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

#Ventanas con zero padding 
nfftp=10*N
FFTxxRec2=fft(xxRec,n=nfftp, axis=0)/N  #Fft de la  ventana rectangular implicita y zero padding 
FFTxxwH2=fft(xxwH,  n=nfftp, axis=0)/N  #Fft de la  ventana hanning y zero padding
FFTxxBH2=fft(xxwBH, n=nfftp, axis=0)/N  #Fft de la  ventana Blackman y zero padding
FFTxxFT2=fft(xxwFT, n=nfftp, axis=0)/N  #Fft de la  ventana flat top y zero padding
ffm2 = np.fft.fftfreq(nfftp, d=ts)     #Tira de frecuencias  con zero padding

#Grafico de las senoideales ventaneadas con zero padding
plt.figure()
plt.plot(ffm2,20*np.log10(np.abs(FFTxxRec2)),'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [6]:Grafico de la FFT de una senoideal con ventana rectangular')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm2,20*np.log10(np.abs(FFTxxwH2)),'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [7]:Grafico de la FFT de una senoideal con ventana hanning')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm2,20*np.log10(np.abs(FFTxxBH2)),'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [8]:Grafico de la FFT de una senoideal con ventana Blackman')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')

plt.figure()
plt.plot(ffm2,20*np.log10(np.abs(FFTxxBH2)),'x',label='ventana rectangular en dB')
plt.xlim([0,fs/2])
plt.title('Figura [9]:Grafico de la FFT de una senoideal con ventana flat top')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]') 

#Estimadores de amplitud (o energia)
EERec=dBRec[N//4, :] #tira del valor maximo de frecuencia para ventana rectangular
EEwH  =dBwH[N//4, :] #tira del valor maximo de frecuencia para ventana hanning
EEBH  =dBBH[N//4, :] #tira del valor maximo de frecuencia para ventana Blackman 
EEFT  =dBFT[N//4, :] #tira del valor maximo de frecuencia para ventana ventana flat top

#Graficos con sesgo 
plt.figure()
plt.hist(EERec, bins=20, edgecolor='k')
plt.xlabel("EE [dB]")
plt.ylabel("Frecuencia")
plt.title("Figura [10]:Histograma de energía estimada para ventana rectangular")

plt.figure()
plt.hist(EEwH, bins=20, edgecolor='k')
plt.xlabel("EE [dB]")
plt.ylabel("Frecuencia")
plt.title("Figura [11]:Histograma de energía estimada para ventana Hanning")

plt.figure()
plt.hist(EEBH, bins=20, edgecolor='k')
plt.xlabel("EE [dB]")
plt.ylabel("Frecuencia")
plt.title("Figura [12]:Histograma de energía estimada para ventana Blackman")

plt.figure()
plt.hist(EEFT, bins=20, edgecolor='k')
plt.xlabel("EE [dB]")
plt.ylabel("Frecuencia")
plt.title("Figura [13]:Histograma de energía estimada para ventana ventana flat top")