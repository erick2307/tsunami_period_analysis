import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import datetime as dt1
from scipy import signal
import scipy.signal              #importing scipy.signal package for detrending
from scipy.fftpack import fft    #importing Fourier transform package
from scipy import stats
from scipy.signal import hilbert
from scipy.signal import butter, lfilter
from waveletFunctions import wavelet, wave_signif
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import scipy.io as sio
import matplotlib.patheffects as PathEffects
#from peakdetect import peakdetect
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator


def matlab2datetime(matlab_datenum):
    day = dt1.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt1.timedelta(days=matlab_datenum%1) - dt1.timedelta(days = 366)
    day_total = day + dayfrac
    micro_mod = round(day_total.microsecond/1e6)
    return day_total.replace(microsecond=0) + dt1.timedelta(seconds=micro_mod)


def datetime2matlabdn(dt):
   ord = dt.toordinal()
   mdn = dt + dt1.timedelta(days = 366)
   frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   return mdn.toordinal() + frac


def sta_lta(tseries,ltw,stw,thresh,dt):
    import numpy as np
    from scipy.signal import hilbert
    il = np.fix(ltw/dt)
    iss = np.fix(stw/dt)
    nt = len(tseries)
    aseries = np.abs(hilbert(tseries))
    sra = np.zeros(nt)

    for ii in range(int(il+1),int(nt)):
        lta = np.mean(aseries[int(ii - il): int(ii)])
        sta = np.mean(aseries[int(ii - iss): int(ii)])
        sra[ii] = sta/lta

    itm = sra[sra>thresh]
    pos = np.where(sra>thresh)[0]
    if np.size(itm)!=0:
        itmax = itm[0]
        pos_max = pos[0]

    return itmax, pos_max

def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    https://scipy.github.io/devdocs/generated/scipy.signal.windows.tukey.html#scipy.signal.windows.tukey

    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

    return w


def wave_spectrum(eta,nfft,Fs):
    n = len(eta)
    nfft = int(nfft - (nfft%2))
    eta = scipy.signal.detrend(eta)
    nBlocks = int(n/nfft)
    eta_new = eta[0:nBlocks*nfft]

    #eta_new = eta_new*tukeywin(len(eta_new))
    etaBlock = np.reshape(eta_new,(nBlocks,nfft))
    etaBlock = etaBlock*tukeywin(nfft)

    df = Fs/nfft
    f = np.arange(0,Fs/2.0+df,df)
    fId = np.arange(0,len(f))

    fft_data = fft(etaBlock,n=nfft,axis=1)
    fft_data = fft_data[:,fId]
    A = 2.0/nfft*np.real(fft_data)
    B = 2.0/nfft*np.imag(fft_data)

    E = (A**2 + B**2)/2

    E = np.mean(E,axis=0)/df

    edf = round(nBlocks*2.0)
    alpha = 0.05#0.1

    confLow = edf/stats.chi2.ppf(1-alpha/2,edf)
    confUpper = edf/stats.chi2.ppf(alpha/2,edf)

    return E, f, confLow, confUpper

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq#
    high = highcut / nyq
    b, a = butter(order, [low, high], analog=False, btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#stations = ['valp']
#names    = [u'Valparaíso']
#year    = '2017'
#event   = u'Valparaíso 2017'

#colores = ['c','m','y','r','b']
colores = ['#e6194B','#f58231','#ffe119','#bfef45','#3cb44b','#42d4f4','#4363d8','#911eb4','#f032e6']
t_total = 84 #horas

alto  = 4
ancho = 12

stations = ['2010-valp','Lorito','Moreno','USGS']
#dt = 15  # time step (sec)
dt = 60  # time step (sec)


fig1=plt.figure(1,figsize=(ancho, alto))
fig1.subplots_adjust(hspace=0.025, wspace=0.025)
fig2=plt.figure(2,figsize=(ancho, alto))
fig2.subplots_adjust(hspace=0.025, wspace=0.025)
fig3=plt.figure(3,figsize=(ancho, alto))
fig3.subplots_adjust(hspace=0.025, wspace=0.025)
for i in range(4):#len(stations)):
    if i == 0:
        file = sio.loadmat(stations[i]+'_notide_long_filtered'+'.mat')
        data   = np.squeeze(file['time'])
        data_date = [matlab2datetime(tval) for tval in data] #convert to Matlab julian number
        eta = np.squeeze(file['eta'])*100 # free surface in cm
        TA, posTA = sta_lta(eta,2700,600,2.15,60) # STA/LTA and calculate Tsunami Arrival Time  en 2010: sta_lta(eta,10800,7200,1.2419,60)
        posTA = posTA + 20
        eta  = eta[posTA-(60*6):posTA+(60*t_total)] # tsunami signal in cm
        time_corr = data_date[posTA-(60*6):posTA+(60*t_total)] # time(julian) of tsunami signal
        # relative time to the Arrival time (AT)
        time = np.zeros(len(time_corr))
        for tt in range(len(time_corr)):
            time[tt] = (1.0/3600.0)*(time_corr[tt]-data_date[posTA]).total_seconds()
    else:
        data = np.loadtxt('TG_valp_zuv_'+stations[i]+'.dat') #load file
        eta  = data[:,0] # free surface in (cm)
        dt = 60  # time step (sec)
        time = np.arange(0,(dt*len(eta)),dt) #time in seconds
        time = (time/(60*60)) - 15/60 #time in hours
        eta = eta[::4]
        time = time[::4]
        eta = eta[:-1]
        time = time[:-1]
    #print(len(time),len(eta))

    # WAVELET PARAMETERS
    n    = len(eta)
    #print n
    dt   = dt/60.0 # en minutos
    pad  = 0  # pad the time series with zeroes (recommended)
    dj   = 0.125  # this will do 4 sub-octaves per octave
    s0   = 2 * dt  # this says start at a scale of 6 months
    j1   = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.72  # lag-1 autocorrelation for red noise background
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelet(eta/100.0, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    Frec = 1.0/period
    global_ws = (np.sum(power, axis=1) / n )  # time-average over all times
    average_time = (np.sum(power, axis=0)) #/ n
    integrate_time = np.trapz(power,x=period,axis=0)


    ## relative time to the Arrival time (AT)
    #time_corr_rel = np.zeros(len(time_corr))
    #for tt in range(len(time_corr)):
    #    time_corr_rel[tt] = (1.0/3600.0)*(time_corr[tt]-data_date[posTA]).total_seconds()

    # Wavelet distribution time-Frequency
    plt.figure(1) # set current figure to fig1
    ax = plt.subplot2grid((len(stations), 6), (i, 0), colspan=5)
    levels = [0.0019/2048, 0.0019/1024, 0.0019/512, 0.0019/256, 0.0019/128, 0.0019/64, 0.0019/32, 0.0019/16,  0.0019/8, 0.0019/4, 0.0019/2, 0.0019, 0.0039, 0.0078, 0.0157, 0.0313, 0.0625,0.125,0.25,0.5,1,2,4,8,16,32]#,64,128]
    cs = ax.contourf(time,np.log2(period),np.log2(power),np.log2(levels),cmap=plt.cm.jet)
    txtl = ax.text(t_total-1,np.log2(180),stations[i],fontsize=16, fontweight='bold',horizontalalignment='right',verticalalignment='top')
    txtl.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='w')])
    Yticks = 2**np.log2([7.5, 15, 30, 60, 120, 240])
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels([7.5, 15, 30, 60, 120, 240])
    ax.set_ylim(np.log2(5),np.log2(256))
    ax.set_xlim(-6,t_total)
    #ax.yaxis.set_tick_params(labelsize=7)
    #ax.grid()

    #if i == len(stations)-1:
    #    cbbox = inset_axes(ax, '18%', '60%', loc = 9)
    #    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    #    cbbox.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    #    cbbox.set_facecolor([1,1,1,0.7])
    #    axins1 = inset_axes(cbbox, width="85%",  # width = 50% of parent_bbox width
    #                             height="10%",  # height : 5%
    #                             loc=9)
    #    cbar = fig1.colorbar(cs, cax=axins1, orientation="horizontal", ticks=[-16, -12, -8, -4, 0, 4])
    #    cbar.set_label('$\mathbf{log_{2}}$(Energy Density) (m$\mathbf{^2}$)',Fontsize=7, fontweight='bold', labelpad=-1.0)#, rotation=270)
    #    axins1.tick_params(axis='both', labelsize=7)

    # energia maxima en funcion del tiempo
    Emax      = np.zeros(len(time))
    Frec_Emax = np.zeros(len(time))
    for ttt in range(len(time)):
        Emax[ttt] = np.max(power[:,ttt])
        ind_Emax  = np.where(power[:,ttt] == power[:,ttt].max())
        period_Emax = period[ind_Emax]
        Frec_Emax[ttt] = Frec[ind_Emax]

    ax.plot(time,np.log2(1/Frec_Emax),'-k',linewidth=0.85)
    ax.axvline(x=0,linewidth=1.5, linestyle='--', color='w')
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Wavelet Spectrum Plot
    ax2 = plt.subplot2grid((len(stations), 6), (i, 5))
    ax2.semilogx(global_ws, np.log2(period))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yticks(np.log2(Yticks))
    ax2.set_yticklabels([7.5, 15, 30, 60, 120, 240])
    ax2.set_ylim(np.log2(5),np.log2(256))
    ax2.set_xlim(10**-4,10**1)
    ax2.grid(color='lightgray', linestyle='--', linewidth=0.5)
    #ax2.yaxis.set_tick_params(labelsize=7)





    if i == len(stations)-1:
        ax.set_xlabel('Relative time (hours)')
        ax.set_ylabel('Period (min)')

    #if i< len(stations)-1:
    #    ax.tick_params(axis='x',labelbottom='off') #set_xticks([])
        #ax2.tick_params(axis='x',labelbottom='off') #set_xticks([])
        #ax3.tick_params(axis='x',labelbottom='off') #set_xticks([])
        #ax4.tick_params(axis='x',labelbottom='off') #set_xticks([])


#fig1.suptitle('Wavelet Analysis - Event: '+event, fontsize=16, fontweight='bold', y=.935)
#fig2.suptitle('Wavelet Analysis - Event: '+event, fontsize=16, fontweight='bold', y=.935)
#fig3.suptitle('Wavelet Analysis - Event: '+event, fontsize=16, fontweight='bold', y=.935)

event   = 'Valparaiso_2017'

#fig1.savefig('WA_'+event+'_1', bbox_inches='tight', transparent=True, dpi=200)
#fig2.savefig('WA_'+event+'_2', bbox_inches='tight', transparent=True, dpi=200)
#fig3.savefig('WA_'+event+'_3', bbox_inches='tight', transparent=True, dpi=200)

plt.show()
