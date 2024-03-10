# analytic WBGT formulation developed by Qinqin Kong based on liljegren et al (2008)
import xarray as xr
import numpy as np
from numba import vectorize, njit
import dask
import os

# define globe constants:
diamglobe = 0.0508 # diameter of globe (m)
emisglobe = 0.95 # emissivity of globe
albglobe = 0.05 # albedo of globe
albsfc=0.45 # albedo of ground surface

#wick constants
emiswick = 0.95 # emissivity of the wick
albwick = 0.4 # albedo of the wick
diamwick = 0.007 # diameter of the wick
lenwick = 0.0254 # length of the wick

# physical constants:
stefanb = 0.000000056696  # stefan-boltzmann constant
mair = 28.97 # molecular weight of dry air (grams per mole)
mh2o = 18.015 # molecular weight of water vapor (grams per mole)
rgas = 8314.34 # ideal gas constant (J/kg mol Â· K)
rair = rgas * (mair**(-1))
cp = 1003.5 # Specific heat capacity of air at constant pressure (JÂ·kg-1Â·K-1)
Pr = cp * ((cp + 1.25 * rair)**(-1)) # Prandtl number 

@vectorize
def esat(tas,ps):
    # tas: 2m air temperature (K)
    # ps: surface pressure (Pa)
    # return saturation vapor pressure (Pa)
    if tas>273.15:
        es=(1.0007 + (3.46*10**(-6) * ps/100))*(611.21 * np.exp(17.502 * (tas - 273.15) *((tas - 32.18)**(-1))))
    else:
        es=(1.0003 + (4.18*10**(-6) * ps/100))*(611.15 * np.exp(22.452 * (tas - 273.15) * ((tas - 0.6)**(-1))))
    return es

@vectorize
def desat_dT(tas,ps):
    # tas: 2m air temperature (K)
    # ps: surface pressure (Pa)
    # return derivative of saturation vapor pressure wrt to temperature 
    if tas>273.15:
        desdT=esat(tas,ps)*17.502*(273.15-32.18)/((tas-32.18)**2)
    else:
        desdT=esat(tas,ps)*22.452*(273.15-0.6)/((tas-0.6)**2)
    return desdT

def h_evap(tas):
    # tas: air temperature (K)
    # return heat of evaporation (J/(kg K))
    return ((313.15 - tas)/30. * (-71100.) + 2.4073e6 )

def viscosity(tas):
    # tas: air temperature (K)
    # return air viscosity (kg/(m s))
    omega=1.2945-tas/1141.176470588
    visc = 0.0000026693 * (np.sqrt(28.97 * tas)) * ((13.082689 * omega)**(-1))
    return visc
def thermcond(tas):
    # tas: air temperature (K)
    # return thermal conductivity of air (W/(m K))
    tc = (cp + 1.25 * rair) * viscosity(tas)
    return tc
def diffusivity(tas,ps):
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # return diffusivity of water vapor in air (m2/s)
    return 2.471773765165648e-05 * ((tas *0.0034210563748421257) ** 2.334) * ((ps / 101325)**(-1))

def h_cylinder_in_air(tas,ps,sfcwind):
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat transfer coefficient for a long cylinder (W/(m2 K))
    thermcon = thermcond(tas)
    density = ps * ((rair * tas)**(-1))
    Re = sfcwind * density * diamwick * ((viscosity(tas))**(-1))
    Nu = 0.281 * (Re ** 0.6) * (Pr ** 0.44)
    h = Nu * thermcon * (diamwick**(-1))
    return h
def h_sphere_in_air(tas, ps, sfcwind):
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective heat tranfer coefficient for flow around a sphere (W/(m2 K))
    thermcon = thermcond(tas)
    density = ps * ((rair * tas)**(-1))
    Re = sfcwind * density * diamglobe * ((viscosity(tas))**(-1))
    Nu = 2 + 0.6 * np.sqrt(Re) * (Pr**0.3333)
    h = Nu * thermcon * (diamglobe**(-1))
    return h
def conv_mass(tas,ps,sfcwind):
    # tas: air temperature (K)
    # ps: surface pressure (Pa)
    # sfcwind: 2 meter wind (m/s)
    # return convective mass transfer coefficient for flow around a cylinder
    h=h_cylinder_in_air(tas, ps, sfcwind)
    density=ps/(tas*rair)
    Sc=viscosity(tas)*((density*diffusivity(tas,ps))**(-1))
    return h/(cp*mair)*((Pr/Sc)**0.56)

@vectorize
def getexp(cosz, wind, rsds):
    # just tries to get the exponential index that is needed for transfering 10m wind speed to 2m level for urban area
    result=wind*0+999
    c11=np.logical_and(cosz>0,np.logical_and(rsds>=925,wind>=5))
    c12=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=675,rsds<925),np.logical_and(wind>=5,wind<6)))
    c13=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=175,rsds<675),np.logical_and(wind>=2,wind<5)))
    c1=np.logical_or(c11,np.logical_or(c12,c13))

    c21=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=675,rsds<925),wind>=6))
    c22=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=175,rsds<675),wind>=5))
    c23=np.logical_and(cosz>0,rsds<175)
    c24=np.logical_and(cosz<=0,wind>=2.5)
    c2=np.logical_or(np.logical_or(c21,c22),np.logical_or(c23,c24))


    c31=np.logical_and(cosz>0,np.logical_and(rsds>=925,wind<5))
    c32=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=675,rsds<925),wind<5))
    c33=np.logical_and(cosz>0,np.logical_and(np.logical_and(rsds>=175,rsds<675),wind<2))
    c3=np.logical_or(c31,np.logical_or(c32,c33))

    c4=np.logical_and(cosz<=0,wind<2.5)
    
    if c1:
        result=0.2
    elif c2:
        result=0.25
    elif c3:
        result=0.15
    elif c4:
        result=0.3
    else:
        result=np.NaN
    return result

@vectorize
def getwind2m(wind10m,cosz,rsds):
    # obtain 2m wind from 10m wind
    wind2m=wind10m * ((2.0/10.0)** getexp(cosz, wind10m, rsds))
    wind2m=0.13 if wind2m<0.13 else wind2m
    return wind2m

def fdir(cosza,coszda,rsds):
    # calculate the fraction of direct solar radiation
    # cosza: average cosine zenith angle during certain interval (e.g. hourly or 3-hourly depending on data temporal resolution)
    # coszda: average cosine zenith angle during the daytime part of certain interval (e.g. hourly or 3-hourly depending on data temporal resolution)
    # mind the difference between cosza nand coszda and its use in this function
    # rsds: downward solar radiation (W/m2)
    
    s_star=rsds*((1367*coszda)**(-1))
    s_star[np.where(s_star>0.85)]=0.85
    f=np.exp(3-1.34*s_star-1.65*(s_star**(-1)))
    f[np.where(f>0.9)]=0.9
    f[np.logical_or(np.logical_or(f<0,cosza<=np.cos(np.deg2rad(89.5))),rsds<0)]=0
    return f


def calculate_wbt(t2_k, rh):
    # calcualte wet-bulb temperature using the emprirical formula developed by Stull (2011)
    # this will be used as a first guess to natural wet-bulb temperature
    # t2_k: 2 meter temperature (K)
    # rh: relative humidity (%)
    # return wet-bulb temperature (K)
    t2_c = t2_k-273.15
    tw = t2_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))+ np.arctan(t2_c + rh)- np.arctan(rh - 1.676331)+ 0.00391838 * (rh) ** (3 / 2) * np.arctan(0.023101 * rh)- 4.686035+273.15
    return tw

def calc_Tg(tas,ps,wind2m,coszda,rsds,rlds,rsus,rlus,f):
    # calculate black globe temperature
    # tas: 2m temperature (K)
    # ps: surface pressure (Pa)
    # wind2m: 2m wind (m/s)
    # coszda: average cosine zenith angle during the daytime part of the interval (e.g. hourly or 3-hourly depending on data temporal resolution)
    # rsds: downward solar radiation (W/m2)
    # rlds: downward themral radiation (W/m2)
    # rsus: surface reflected solar radiation (W/m2)
    # rlus: upwelling thermal radiation (W/m2)
    # f: fraction of direct solar radiation
    
    # using tas as a first-guess to Tg when calculating convective and thermal radiative heat transfer coefficients
    hc=h_sphere_in_air(tas, ps, wind2m) 
    hr=4*emisglobe*stefanb*(tas**3)
    
    SR=0.5*(1-albglobe)*rsds*(1-f+0.5*f/coszda)+0.5*(1-albglobe)*rsus
    LR=0.5*emisglobe*(rlds+rlus)-stefanb*emisglobe*(tas**4)
    Tg=tas+(SR+LR)/(hc+hr)
    return Tg

def calc_Tnw(tas,ea,ps,wind2m,coszda,rsds,rlds,rsus,rlus,f):
    # calculate natural wet-bulb temperature  
    # tas: 2m temperature (K)
    # ea: vapor pressure (Pa)
    # ps: surface pressure (Pa)
    # wind2m: 2m wind (m/s)
    # coszda: average cosine zenith angle during the daytime part of the interval (e.g. hourly or 3-hourly depending on data temporal resolution)
    # rsds: downward solar radiation (W/m2)
    # rlds: downward themral radiation (W/m2)
    # rsus: surface reflected solar radiation (W/m2)
    # rlus: upwelling thermal radiation (W/m2)
    # f: fraction of direct solar radiation
    
    es=esat(tas,ps)
    rh=ea/es*100
    
    # using Stull's Tw as a first-guess to Tnw when calculating mass and heat transfer coefficients
    tw_stull=calculate_wbt(tas, rh)
    tf=(tw_stull+tas)/2 # film temperature
    lv=h_evap(tf)
    kx=conv_mass(tf,ps,wind2m)
    hc=h_cylinder_in_air(tf,ps,wind2m)
    beta=kx*mh2o*lv/(ps-esat(tw_stull,ps))
    he=beta*desat_dT(tf,ps)
    hr=stefanb*emiswick*(tw_stull**2+tas**2)*(tw_stull+tas)
    SR=(1-albwick)*((1+0.25*diamwick/lenwick)*(1-f)*rsds+(np.tan(np.arccos(coszda))/np.pi+0.25*diamwick/lenwick)*f*rsds+rsus)
    LR=0.5*emiswick*(rlds+rlus)-stefanb*emiswick*tas**4
    VPD=beta*(es-ea)
    Tnw=tas+(SR+LR-VPD)/(he+hc+hr)
    return Tnw
def calc_WBGT(tas,ea,ps,wind2m,coszda,rsds,rlds,rsus,rlus,f):
    # calculate wet-bulb globe temperature
    # tas: 2m temperature (K)
    # ea: vapor pressure (Pa)
    # ps: surface pressure (Pa)
    # wind2m: 2m wind (m/s)
    # coszda: average cosine zenith angle during the daytime part of the interval (e.g. hourly or 3-hourly depending on data temporal resolution)
    # rsds: downward solar radiation (W/m2)
    # rlds: downward themral radiation (W/m2)
    # rsus: surface reflected solar radiation (W/m2)
    # rlus: upwelling thermal radiation (W/m2)
    # f: fraction of direct solar radiation
    
    Tg=calc_Tg(tas,ps,wind2m,coszda,rsds,rlds,rsus,rlus,f)
    Tnw=calc_Tnw(tas,ea,ps,wind2m,coszda,rsds,rlds,rsus,rlus,f)
    WBGT=0.7*Tnw+0.2*Tg+0.1*tas
    return WBGT

