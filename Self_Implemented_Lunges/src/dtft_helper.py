import math
from re import T
import params
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
from scipy import fftpack

def plot_dtft(cur_loss_id,save_=False,path=None):
    x1,mag,phase = plt_dtft_helper(cur_loss_id)
    # fig, axs = plt.subplots(1, 2,figsize=(15,15))
    # axs[0, 0].plot(mag)
    # axs[0, 0].set_title('MAG')
    # axs[0, 1].plot(phase)
    # axs[0, 1].set_title('Phase')

    #Plotting the Magnitude
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.scatter(x1,mag)
    plt.xlabel('$\omega$')
    plt.ylabel('$|Y(e^{j\omega})|$')
    plt.grid()

    #Plotting the Phase angle
    plt.subplot(2,1,2)
    plt.scatter(x1,phase)
    plt.xlabel('$\omega$')
    plt.ylabel('$\phi$')
    plt.grid()
    if save_:
        plt.savefig(path+"_img.png")
    else:
        plt.show()
    plt.close()

# NEED TO EXPLORE WHY SOME OF THE REPS HAVE SUCH LOW COUNT , maybe coz of removing rows - how to handle them more efficiently
def scp_dtft_helper(cur_loss_id,wd=5): 
    # print("SCIPY DTFT, n = ",wd)
    wd = params.dtft_width
    X_fft = fftpack.fft(cur_loss_id)
    X_rfft = fftpack.rfft(cur_loss_id)

    X_fft=X_fft[::2]
    X_fft_abs= np.abs(X_fft)
    X_rfft_abs= np.abs(X_rfft)

    # X = fftpack.fftshift(X)
    freqs_rfft = fftpack.rfftfreq(len(cur_loss_id)) 
    freqs_fft = fftpack.fftfreq(len(cur_loss_id)) 

    ind_fft = np.argsort(X_fft_abs)
    # print(len(cur_loss_id),"fft size",len(X_fft_abs[ind_fft]))
    X_fft = X_fft[ind_fft][-wd:]
    X_fft_abs=X_fft_abs[ind_fft][-wd:]
    freqs_fft = freqs_fft[ind_fft][-wd:]

    ind_rfft = np.argsort(X_rfft_abs)
    X_rfft = X_rfft[ind_rfft][-wd:]
    X_rfft_abs=X_rfft_abs[ind_rfft][-wd:]
    freqs_rfft = freqs_rfft[ind_rfft][-wd:]

    # return freqs_fft,X_fft_abs,np.angle(X_fft)
    return freqs_rfft,X_rfft_abs,np.angle(X_rfft)


def dtft_pos_conversion(target_pos):
    arr = []
    def write(lst):
        if not isinstance(lst, list):
            lst = lst.tolist()
        # _, mag, phs = plt_dtft_helper(lst)
        _, mag, phs = scp_dtft_helper(lst,params.dtft_width)

        for i in mag:
            arr.append(i)
        for i in phs:
            arr.append(i)
    for row in range(len(target_pos)):
        # import pdb;pdb.set_trace()
        write(target_pos[row])
    return arr

def dtft_conversion(cur_loss_id):
    arr = []

    def write(lst):
        # _, mag, phs = plt_dtft_helper(lst)
        _, mag, phs = scp_dtft_helper(lst,params.dtft_width)

        for i in mag:
            arr.append(i)
        for i in phs:
            arr.append(i)
    # print("loss id size",len(cur_loss_id['bt']))
    for key in params.keys:
        # print(key)
        write(cur_loss_id[key])
    # write(mse_n)
    # write(mse_lh)
    # write(mse_rh)
    # write(mse_lk)
    # write(mse_rk)
    # write(mse_t)

    return arr

def plt_dtft_helper(f):
  # (https://gist.github.com/TheRealMentor/018aab68dc4bb55bb8d9a390f657bd1d)

  #Defining DTFT function
  def dtft(f,pt):
      output = [0]*n
      for k in range(n):  
          s = 0
          p = 0
          for t in range(len(f)): 
              s += f[t] * cm.exp(-1j * pt[k] * t)
          output[k] = s
      return output

  #Calculating the magnitude of DTFT
  def magnitude(inp,n):
      output = [0]*n
      for t in range(0,n):
          tmp=inp[t]
          output[t]= math.sqrt(tmp.real**2 + tmp.imag**2)
      return output

  #Calculating the phase 
  def phase(inp,n):
      output = [0]*n
      for t in range(0,n):
          tmp=inp[t]
          output[t]= math.atan2(tmp.imag,tmp.real)
      return output

  n = 11
#   print("GITHUB DTFT , n = ",n)
  #Defining the x-limits
  N = 2*((math.pi)/n)
  x = np.arange(-(math.pi),math.pi,N)
  x1 = np.fft.fftshift(x)
  x1 = x1[:n]

  #Using the function that I made
  made_func = dtft(f,x)
  made_func_shift=np.fft.fftshift(made_func)
  made_func_shift_mag = magnitude(made_func_shift,n)
  made_func_shift_phs = phase (made_func_shift,n)

  #Using the inbuilt function
  #inbuilt = np.fft.fft(f,n)
  #inbuilt_mag = magnitude(inbuilt,n)
  #inbuilt_phs = phase (inbuilt,n)

  return x1, made_func_shift_mag, made_func_shift_phs

