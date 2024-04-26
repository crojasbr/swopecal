import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from pdastro import pdastrostatsclass
from scipy.odr import ODR, Model, Data, RealData
from scipy.optimize import curve_fit


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

def load_and_prepare_data(filename, g_col,gerr_col, i_col,ierr_col, insmag_col, insmagerr_col,zpt_col, delta_mag_col,delta_mag_col_err):
    df = pd.read_csv(filename)
    df['gPS1-iPS1'] = df[g_col] - df[i_col]
    df['gPS1-iPS1_errors'] = np.sqrt(df[gerr_col]**2 + df[ierr_col]**2)
    
    df['delta'] = df[delta_mag_col] - (df[insmag_col] + df[zpt_col])
    df['delta_err'] = np.sqrt(df[delta_mag_col_err]**2+df[insmagerr_col]**2)
    
    print (np.max(df['delta_err']),df[insmagerr_col][np.argmax(df['delta_err'])])
    
    return df

# def linear_model(params, x): #for odr
#     return params[0] * x + params[1]

def linear_model(x, a, b):
    return a * x + b

def plot_and_fit(df_synthetic, df_observed, title, x_col, xerr_col,y_col, fit_col):
    coefficients_syn = np.polyfit(df_synthetic[x_col], df_synthetic[fit_col], 1)
    x_fit_syn = np.linspace(-1, 3, 400)  # Extended range for the fit line
    fit_line_syn = np.polyval(coefficients_syn, x_fit_syn)
    line_eq_syn = f'y = {coefficients_syn[0]:.3f}x + {coefficients_syn[1]:.3f}'

    clipped = sigma_clip(df_observed['delta'], sigma=3, maxiters=5)
    valid_indices = ~clipped.mask  
    clipped_x = df_observed[x_col][valid_indices]
    clipped_xerr = df_observed[xerr_col][valid_indices]
    clipped_y = clipped[valid_indices]
    clipped_yerr = df_observed['delta_err'][valid_indices]
    
    

    
#using armins pdastro
#     residuals = pdastrostatsclass()
#     residuals.t['residual']=df_observed['delta']
#     residuals.t['residualerr']=df_observed['delta_err']

#     residuals.calcaverage_sigmacutloop('residual',verbose=3,noisecol='residualerr',Nsigma=3.0,percentile_cut_firstiteration=75)

#     idx=residuals.statparams['ix_good']
    
#     clipped_y=df_observed['delta'][idx]
#     clipped_yerr=df_observed['delta_err'][idx]
#     clipped_x = df_observed[x_col][idx]
#     clipped_xerr = df_observed[xerr_col][idx]



    plt.figure(figsize=(10, 6))
    plt.scatter(df_synthetic[x_col], df_synthetic[fit_col], color='blue', label='calspec synth mags', zorder=10)
    
    
    plt.plot(x_fit_syn, fit_line_syn, color='red', label='Linear fit to synth mags (' + line_eq_syn + ')')

#     plt.scatter(clipped_x, clipped_y, color='orange', label='Observed mags', marker='.', zorder=1, alpha=0.5,s=15)
    plt.errorbar(clipped_x, clipped_y, yerr=clipped_yerr,xerr=clipped_xerr,fmt='.',label='Observed mags', zorder=1, alpha=0.2,color='orange',ms=3)
    if len(clipped_x) > 1:
        try:
#             data = RealData(clipped_x, clipped_y, sx=clipped_xerr, sy=clipped_yerr) #odr
#             model = Model(linear_model)
#             if title=='PS g- Swope g':
#                 b1=-0.059
#                 b2=-0.016
#             if title=='PS i- Swope i':
#                 b1=0.024
#                 b2=-0.006
#             print (b1,b2)
#             odr = ODR(data, model, beta0=[b1, b2])
#             output = odr.run()
#             x_fit = np.linspace(clipped_x.min(), clipped_x.max(), 400)
#             fit_line = linear_model(output.beta, x_fit)
#             plt.plot(x_fit, fit_line, 'g-', label=f'Fitted line: y = {output.beta[0]:.3f}x + {output.beta[1]:.3f}')

            
    
            popt, pcov = curve_fit(linear_model, clipped_x, clipped_y, sigma=clipped_yerr, absolute_sigma=True)
            a, b = popt
            sigma_a, sigma_b = np.sqrt(np.diag(pcov))
            x_fit_obs = np.linspace(-1, 3, 400)
            fit_line = linear_model(x_fit_obs, *popt)
            plt.plot(x_fit_obs, fit_line, 'g-', label=f'Fitted line: y = {a:.3f}x + {b:.3f}')

            
#             coefficients_obs = np.polyfit(clipped_x, clipped_y, 1)
#             x_fit_obs = np.linspace(-1, 3, 400)
#             fit_line_obs = np.polyval(coefficients_obs, x_fit_obs)
#             line_eq_obs = f'y = {coefficients_obs[0]:.3f}x + {coefficients_obs[1]:.3f}'
#             plt.plot(x_fit_obs, fit_line_obs, color='green', label='Linear fit to observed mags (' + line_eq_obs + ')')
        except Exception as e:
            print(f"Failed to fit observed data for {title}: {str(e)}")
    else:
        print(f"Not enough valid points to fit observed data for {title}")

    plt.xlabel('g - i')
    plt.ylabel(title)
    plt.title(f'{title}')
    plt.legend()
    plt.grid(True)
    plt.xlim(-1, 3) 
#     plt.ylim(-0.2, 0.2) 
    plt.show()

df_synthetic = pd.read_csv('synth_phot_calspec.txt')
df_synthetic['gPS1-iPS1'] = df_synthetic['PS1-g'] - df_synthetic['PS1-i']

dfg = load_and_prepare_data('g_final.csv', 'g', 'dg','i','di','insmag_g', 'insmagerr_g','zpt', 'g','dg')
dfi = load_and_prepare_data('i_final.csv', 'g', 'dg','i','di','insmag_i', 'insmagerr_i','zpt', 'i','di')
dfr = load_and_prepare_data('r_final.csv', 'g', 'dg','i','di','insmag_r', 'insmagerr_r','zpt', 'r','dr')
dfv = load_and_prepare_data('V_final.csv', 'g', 'dg','i','di','insmag_V', 'insmagerr_V','zpt', 'r','dr')
dfb = load_and_prepare_data('b_final.csv', 'g', 'dg','i','di','insmag_B', 'insmagerr_B','zpt', 'g','dg')

combinations = {
    'PS g- Swope g': (dfg, 'PS1-g', 'Swope-g'),
    'PS i- Swope i': (dfi, 'PS1-i', 'Swope-i'),
    'PS r- Swope r': (dfr, 'PS1-r', 'Swope-r'),
    'PS r- Swope V': (dfv, 'PS1-r', 'Swope-V'),
    'PS g- Swope B': (dfb, 'PS1-g', 'Swope-B'),
}

for title, (df_observed, ps1, swope) in combinations.items():
    print (title)
#     print (df_observed)
    df_synthetic[title] = df_synthetic[ps1] - df_synthetic[swope]
    plot_and_fit(df_synthetic, df_observed, title, 'gPS1-iPS1','gPS1-iPS1_errors', title, title)

