import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1Getting started
''' This section is setting up pycharm, I'll assume that the reader of this project can configure their own IDE '''


# 2: Velocity shifts

# Reading dataframes for the various text files using Pandas:



def dfcalls(file_name):
    return pd.read_csv(file_name, skiprows=0, delimiter=' ')


df_NGC4258 = dfcalls('C:/Users/Charlie Masters/Desktop/CodeDesktop/NGC4258.txt')
df_NGC4639 = dfcalls('C:/Users/Charlie Masters/Desktop/CodeDesktop/NGC4639.txt' )
df_SDSSTemplate_late = dfcalls('C:/Users/Charlie Masters/Desktop/CodeDesktop/SDSSTemplate_Late.txt')
df_SDSSTemplate_Early = dfcalls('C:/Users/Charlie Masters/Desktop/CodeDesktop/SDSSTemplate_Early.txt')

# Normalising the flux functions for each of the datasets:

df_NGC4258['Flux'] = df_NGC4258['Flux'].div(1.887269897460937500e+02)

df_NGC4639['Flux'] = df_NGC4639['Flux'].div(1.294947147369384766e+01)

df_SDSSTemplate_late['Flux'] = df_SDSSTemplate_late['Flux'].div(9.238700382411479950e-03)

df_SDSSTemplate_Early['Flux'] = df_SDSSTemplate_Early['Flux'].div(4.559100139886140823e-03)

# Plotting the normalised Fluxes on one graph

plt.plot(df_NGC4258['#Wavelength_(Angstroem)'], df_NGC4258['Flux'], "r", label='NGC4258')
plt.plot(df_NGC4639['#Wavelength_(Angstroem)'], df_NGC4639['Flux'], "black", label='NGC4639')
plt.plot(df_SDSSTemplate_late['#Wavelength_(Angstroem)'], df_SDSSTemplate_late['Flux'], label='SDSSTemplate_late')
plt.plot(df_SDSSTemplate_Early['Wavelength_(Angstroem)'], df_SDSSTemplate_Early['Flux'], label='SDSSTemplate_Early')

plt.xlabel("Wavelength /Angstroem")
plt.ylabel("Normalised flux")
plt.title("4 hubble datasets normalised flux against wavelength")
plt.legend()
plt.show()


# Finding the maximum of each of the datasets


def Data_max(FLux_name):
    return (pd.sort_values('Flux', ascending=False).head(1))


NGC4258_max = df_NGC4258.sort_values('Flux', ascending=False).head(1)
NGC4639_max = df_NGC4639.sort_values('Flux', ascending=False).head(1)
SDSSTemplate_late_max = df_SDSSTemplate_late.sort_values('Flux', ascending=False).head(1)
SDSSTemplate_early_max = df_SDSSTemplate_Early.sort_values('Flux', ascending=False).head(1)

print(NGC4258_max)
print(NGC4639_max)
print(SDSSTemplate_late_max)
print(SDSSTemplate_early_max)


# finding the recessional velocity away


def velocityshift(L0, L):
    C = 3 * (10 ** 8)
    V = C * (L + L0) / L0

    return V


Hb_L0 = 4862.7
O3_L0 = 5008.2
O3_l1 = 4960.3

RecessionV_NGC4258 = print(velocityshift(O3_l1, 5014.181641))

RecessionV_NGC4639 = print(velocityshift(O3_L0, 6588.703613))


# Section 3: Measuring Distance

# Creating dataframes for the various cepheriods
def DfCalls2(FilenamePath):
    return (pd.read_csv(FilenamePath, skiprows=0, delimiter=' ', ))


df_I057665 = DfCalls2('C:/Users/Charlie Masters/Desktop/CodeDesktop/I-057665_lc_v.txt')
df_I001625 = DfCalls2('C:/Users/Charlie Masters/Desktop/CodeDesktop/I-001625_lc_v.txt')
df_I106960 = DfCalls2('C:/Users/Charlie Masters/Desktop/CodeDesktop/I-106960_lc_v.txt')
df_I025811 = DfCalls2('C:/Users/Charlie Masters/Desktop/CodeDesktop/I-025811_lc_v.txt')

# Zero dating the julian day data

df_I057665['#Modified_Julian_Data_(days)'] = df_I057665['#Modified_Julian_Data_(days)'] - 2.452978814699999988e+06

df_I001625['#Modified_Julian_Data_(days)'] = df_I001625['#Modified_Julian_Data_(days)'] - 2.452978814699999988e06

df_I106960['#Modified_Julian_Data_(days)'] = df_I106960['#Modified_Julian_Data_(days)'] - 2.452978814699999988e6

df_I025811['#Modified_Julian_Data_(days)'] = df_I025811['#Modified_Julian_Data_(days)'] - 2.452978814699999988e6

# Plotting the cepheriod data on same graph
plt.plot(df_I057665['#Modified_Julian_Data_(days)'], df_I057665['V_band_magnitude'], "r", label='I057665')
plt.plot(df_I001625['#Modified_Julian_Data_(days)'], df_I001625['V_band_magnitude'], "g", label='I001625')
plt.plot(df_I106960['#Modified_Julian_Data_(days)'], df_I106960['V_band_magnitude'], "b", label='I106960')
plt.plot(df_I025811['#Modified_Julian_Data_(days)'], df_I025811['V_band_magnitude'], "y", label='I001625')
plt.xlabel("Modified_Julian_Data_(days)")
plt.ylabel("V_band_magnitude")
plt.title("4 hubble datasets normalised flux against wavelength")
plt.legend()
plt.show()


# finding absolute mag and period:

def Absolute_Mag(t):
    M = -1.43 - 2.81 * np.log(t)
    return M


def distance_formula(M, m, t):
    d = 10 / 3.0857E+32 * (10 ** (m - M))
    return d


M_dI057665 = Absolute_Mag(4.722300e+00)
M_I001625 = Absolute_Mag(32.24e+00)
M_I106960 = Absolute_Mag(32.24e+00)
M_I025811 = Absolute_Mag(6.793000e+00)

d_dI057665 = distance_formula(M_dI057665, df_I057665['V_band_magnitude'].mean(), 4.722300e+00)
d_I001625 = distance_formula(M_I001625, df_I001625['V_band_magnitude'].mean(), 32.24e+00)
d_I106960 = distance_formula(M_I106960, df_I106960['V_band_magnitude'].mean(), 32.24e+00)
d_I025811 = distance_formula(M_I025811, df_I025811['V_band_magnitude'].mean(), 6.793000e+00)

print(d_dI057665, d_I001625, d_I106960, d_I025811)


# begin curve fitting:


def Sine1(x, a, b):
    return np.sin(x) + b  # straight line, non-zero intercept


x1 = df_I057665['#Modified_Julian_Data_(days)']
y1 = df_I057665['V_band_magnitude']

x2 = df_I001625['#Modified_Julian_Data_(days)']
y2 = df_I001625['V_band_magnitude']

x3 = df_I106960['#Modified_Julian_Data_(days)']
y3 = df_I106960['V_band_magnitude']

x4 = df_I025811['#Modified_Julian_Data_(days)']
y4 = df_I025811['V_band_magnitude']

popt1, pcov1 = curve_fit(Sine1, x1, y1)
popt2, pcov2 = curve_fit(Sine1, x2, y2)
popt3, pcov3 = curve_fit(Sine1, x3, y3)
popt4, pcov4 = curve_fit(Sine1, x4, y4)

# Model The best guesses on a graph

plt.plot(x1, Sine1(x1, *popt1))
plt.plot(x2, Sine1(x2, *popt2))
plt.plot(x3, Sine1(x3, *popt3))
plt.plot(x4, Sine1(x4, *popt4))
plt.xlabel("Modified_Julian_Data_(days)")
plt.ylabel("V_band_magnitude")
plt.title("4 hubble datasets fitted to a sinisoduial curve")
plt.legend()
plt.show()

# Plot Real Data

plt.plot(df_I057665['#Modified_Julian_Data_(days)'], df_I057665['V_band_magnitude'], "r", label='I057665')
plt.plot(df_I001625['#Modified_Julian_Data_(days)'], df_I001625['V_band_magnitude'], "g", label='I001625')
plt.plot(df_I106960['#Modified_Julian_Data_(days)'], df_I106960['V_band_magnitude'], "b", label='I106960')
plt.plot(df_I025811['#Modified_Julian_Data_(days)'], df_I025811['V_band_magnitude'], "y", label='I001625')
plt.xlabel("Modified_Julian_Data_(days)")
plt.ylabel("V_band_magnitude")
plt.title("4 hubble datasets normalised flux against wavelength")
plt.legend()
plt.show()

# 4: Hubbles law:


# read in data
def dfCalls(file_name):
    return (pd.read_csv(file_name, skiprows=1, delimiter=','))


hubble = dfCalls('C:/Users/Charlie Masters/Desktop/ahh.csv')

# find H0

VelocityMean = (hubble['Velocity'].mean())
DistanceMean = (hubble['Distance'].mean())

H0 = VelocityMean / DistanceMean
print(H0)

t = 1 / H0
print(t)


# Read in friendmans
def dfCalls2(file_name):
    return (pd.read_csv(file_name, skiprows=0, delimiter='	'))


Freedman = dfCalls2('C:/Users/Charlie Masters/Desktop/freedman.txt')

# Graphing


x = Freedman['V_Virgo(km/s)']
y = Freedman['Do(Mpc)']


def line(x, a, b):
    return a * x + b  # straight line, non-zero intercept


popt, pcov = curve_fit(line, x, y)

plt.scatter(Freedman['V_Virgo(km/s)'], Freedman['Do(Mpc)'])
plt.plot(x, line(x, *popt))
plt.xlabel("Do(Mpc)")
plt.ylabel("V_Virgo(km/s)")
plt.legend()
plt.show()

x = Freedman['V_Virgo(km/s)']
y = Freedman['Do(Mpc)']

# Localgroup.txt

LocalGroup = dfCalls2('C:/Users/Charlie Masters/Desktop/localgroup.txt')
print(LocalGroup)

plt.scatter(LocalGroup['Distance(kpc) '], LocalGroup['Velocity(km/s)'])
plt.xlabel("Distance(kpc)")
plt.ylabel("Velocity(km/s)")
plt.legend()
plt.show()

H0 = LocalGroup['Velocity(km/s)'].mean() / LocalGroup['Distance(kpc) '].mean()
print(H0)
