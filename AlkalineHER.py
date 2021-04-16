#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import scipy.optimize as optimization
from tqdm import tqdm
from copy import deepcopy
from DiabeticPES import *


#constants to be used
attempt_frequency = 0.621*10**(13)

k_b = 8.617E-5

H2_referenced_to_H_vaccum = -6.728 # -7.0042 - 2*-0.13812265 (VASP 500 Cutoff PBE vdW=12) where -0.13812265 is the H (without spin) in vacuum

half_H2_referenced_to_H_vaccum = 0.5*H2_referenced_to_H_vaccum

OH_solvation_eng = -1.90 # partial solvation energy of OH- from water-layer surface to water-layer bulk

H_O_bond_length = 0.997

H_H_distance = 0.903

H_metal_distance ={'H|Pt_100': 1.109,'H|Pt_111': 1.012}

morse_a_dict ={'a_H_on_Pt_100': 1.089, 'a_H_on_H_Pt_100': 1.726
               ,'a_H_on_Pt_111': 1.056, 'a_H_on_H_Pt_111': 1.726, # parameter a if fitted from the data of potential energy surfaces (PES) of H on Pt, H on H@Pt, and H in H2O;
               'a_H_on_water': 1.787} # these PES files (*.tsv) are provided in folder of 'PES_files'

electrode_area = {'Pt100':2.672*2.672, # Pt(100) 1x1 unitcell, unit = Angstrom
                  'Pt111':5.345*4.629} # Pt(111) 1x1 unitcell, unit = Angstrom

electrode_reaction_sites = {'Pt100':2, # Pt(100) 1x1 unitcell
                            'Pt111':4} # Pt 1x1 unitcell

sq_angstrom_2_sq_centimeter = 1*10**(-16) # convert the unit of squared angstrom to squred centimeter

e_charge = 1.60217662*10**(-19) # elementary charge 


'''using sigmoid function to fit the well depth and equilibrium position of Morse potentisl;
the simulations of constants (H* coverage-dependent) in 'c_wd' and 'c-a' are introduced in the SI of our paper
entitled 'Modelling Potential-dependent Electrochemical Activation Barriers - Revisiting the Alkaline Hydrogen Evolution Reaction' '''
def morse_wd_deq(H_cov=0.75,c_wd=[14.645,0.287,0.285,-0.438],c_a=[12.783,0.402,0.986,2.914]):
    c1,c2,c3,c4=c_wd[0],c_wd[1],c_wd[2],c_wd[3]
    wd=c3/(1+np.e**(-c1*(H_cov-c2)))+c4
    C1,C2,C3,C4=c_a[0],c_a[1],c_a[2],c_a[3]
    deq=C3/(1+np.e**(-C1*(H_cov-C2)))+C4
    return {'well_depth':-wd,'optimal_d':deq}

'''using 4-order polynormial function to fit the H* coverage-dependent parameter a in Morse potential; See SI in the paper'''
def fitting_morse_a(H_cov=0.75,c_d=[0.4267,-1.7067,2.3733,-1.3733,1.41]):
    a=c_d[0]*H_cov**4+c_d[1]*H_cov**3+c_d[2]*H_cov**2+c_d[3]*H_cov+c_d[4]
    return a

'''using the fitted Morse a, well depth and equilibrium distance to generate electrode-solvent distance distribution;
the distribution is H* coverage-dependent and also electrode-electrolyte-distance dependent;
assume the distribution follows the Boltzmann distribution'''
def cov2distribution_Pt_100(H_cov=0.75,d_ew=[2,7],N_sam=200,plot=False, T=300,show_plot=False,save_fig=False):
    wd=morse_wd_deq(H_cov)['well_depth']
    deq=morse_wd_deq(H_cov)['optimal_d']
    a=fitting_morse_a(H_cov)
    #print('well depth: ',wd,'optimal distance: ',deq,'morse_a:',a)
    d_ew_all=np.linspace(d_ew[0],d_ew[1],N_sam)
    morse_P=[]
    KbT=8.6173E-5*T
    Prob_all=[]
    for d in d_ew_all:
        P=wd*((1-np.e**(-a*(d-deq)))**2-1)
        morse_P.append(P)
        Boltz_prob=np.e**(-(P+wd)/KbT)
        Prob_all.append(Boltz_prob)
    pre_f=np.sum(Prob_all) 
    Boltz_P_final=(100/pre_f)*np.array(Prob_all)
    #print('P_sum: ',np.sum(Boltz_P_final))
    if show_plot==True:
        plt.plot(d_ew_all,Boltz_P_final)
        if save_fig==True:
            plt.savefig('metal_water_distribution.pdf',bbox_inches='tight')
    return {'d_ew_list':list(d_ew_all),'morse_peotential':morse_P,'Probability':list(Boltz_P_final)} 


'''using the electroe-solvent distribution of Pt(100) at high coverage, i.e., 0.75 to represent the distribution for Pt(111).
See details in the paper mentioned above.'''
def cov2distribution_Pt_111(H_cov=0.75,d_ew=[2,7],N_sam=200,plot=False, T=300,show_plot=False,save_fig=False):
    wd=morse_wd_deq(0.75)['well_depth']
    deq=morse_wd_deq(0.75)['optimal_d']
    a=fitting_morse_a(0.75)
    #print('well depth: ',wd,'optimal distance: ',deq,'morse_a:',a)
    d_ew_all=np.linspace(d_ew[0],d_ew[1],N_sam)
    morse_P=[]
    KbT=8.6173E-5*T
    Prob_all=[]
    for d in d_ew_all:
        P=wd*((1-np.e**(-a*(d-deq)))**2-1)
        morse_P.append(P)
        Boltz_prob=np.e**(-(P+wd)/KbT)
        Prob_all.append(Boltz_prob)
    pre_f=np.sum(Prob_all) 
    Boltz_P_final=(100/pre_f)*np.array(Prob_all)
    #print('P_sum: ',np.sum(Boltz_P_final))
    if show_plot==True:
        plt.plot(d_ew_all,Boltz_P_final)
        if save_fig==True:
            plt.savefig('metal_water_distribution.pdf',bbox_inches='tight')
    return {'d_ew_list':list(d_ew_all),'morse_peotential':morse_P,'Probability':list(Boltz_P_final)} 


'''the following is the H* coverage-dependent adsorption energy referenced to H in vacuum on Pt(100) and on Pt(111) surfaces;
all the optimized structures are available at https://www.catalysis-hub.org/publications/JiangModelling2021'''
def coverage_dependence(metal = 'Pt', facet='111', adsorbate = 'H',theta = None):
    if metal == 'Pt' and facet == '100':
        if adsorbate == 'H':
            ad_eng = 3.0468*theta**3-2.8995*theta**2 +1.294*theta-4.1187 + 0.22 # 0.22 is the correction for free energy 
        elif adsorbate == 'H@H':
            ad_eng = -3.0468*theta**3+2.8995*theta**2 -1.294*theta-2.6093 + 0.35 # Free energy correction of H2 above Pt 
    if metal == 'Pt' and facet == '111':
        if adsorbate == 'H':
            ad_eng = 2.9848*theta**3-4.3245*theta**2+1.864*theta -3.9144 + 0.22 # 0.22 is the correction for free energy 
        elif adsorbate == 'H@H':
            ad_eng = -2.9848*theta**3+4.3245*theta**2-1.864*theta -2.8136 + 0.35 # Free energy correction of H2 above Pt 
    return ad_eng


'''solve the alkaline HER kinetics with steady-state assumption'''
def get_coverage(k_volmer_f,k_volmer_r, k_heyrov_f, k_heyrov_r,k_tafel_f, k_tafel_r, pH=None, C_H2O =1, P_H2=0):
    C_OH=10**(pH-14)
    C_H2O=1
    
    k1f=k_volmer_f  
    k1r=k_volmer_r
    k2f=k_heyrov_f
    k2r=k_heyrov_r
    k3f=k_tafel_f
    k3r=k_tafel_r
    
    a=(2*k3r*P_H2-2*k3f)
    b=-(k1f*C_H2O+k1r*C_OH+k2f*C_H2O+k2r*C_OH*P_H2+4*k3r*P_H2)
    c=(k1f*C_H2O+k2r*C_OH*P_H2+2*k3r*P_H2)
    
    delta  = b**2-4*a*c    
    
    if np.abs(a/b) >= 1e-3:
        theta_H = (-b-delta**(0.5))/(2*a) # delta is a extremely large number; only this solution is reasonable
    
    elif np.abs(a/b) < 1e-3:
        theta_H = c/-b
                                               
    return theta_H


'''plot 2D heatmap to show kinetic data'''
def plot_heatmap(X, Y, Z, reselution = 950, cmap = 'jet', c_rev=False, fig_size=[4,3],file_name = None,v_range=None, nv=None):
    if c_rev==True:
        cmap = 'jet_r'
    if v_range and nv is not None:
        fig, ax = plt.subplots(figsize=fig_size,dpi=300)
        CS = plt.contourf(X, Y, Z, reselution,vmin = v_range[0], vmax = v_range[1], cmap=cmap)
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(Z)
        m.set_clim(v_range[0], v_range[1])
        plt.colorbar(m,ticks=np.linspace(v_range[0], v_range[1],nv))
        #plt.colorbar(m,boundaries=np.linspace(v_range[0], v_range[1],nv))
        plt.xlabel("d-ew")
        plt.ylabel("U-RHE")
        #ax.set_xlim([2, 7])
        plt.xticks(np.arange(2, 7+1, 1.0)) # dew is from 2 to 7 Angstrom
        plt.savefig('%s.pdf'%(file_name),bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=fig_size,dpi=300)
        plt.contourf(X, Y, Z, reselution, fontsize='large',cmap=cmap)
        plt.xlabel("d-ew")
        plt.ylabel("U-RHE")
        plt.colorbar(extend='both')
        ax.set_xlim([2, 7])
        plt.savefig('%s.pdf'%(file_name),bbox_inches='tight')


'''plot 3D surface to show kinetic data'''       
def plot_3D_surface(X,Y,Z,Z_label = 'barrier-vomer-fwd', file_name = None):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', #'viridis',#cmap = 'jet' 'hot' 'rainbow'
                           linewidth=0, antialiased=False)#rstride=1, cstride=1, cmap='viridis',
    # Customize 
    ax.view_init(elev=30,azim=-70)
    #ax.set_zlim(0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("d-ew")
    ax.set_ylabel("U-RHE")
    ax.set_zlabel(Z_label)
    plt.savefig('%s.pdf'%(file_name),bbox_inches='tight')


"""define a metal-water interface class with considering the distribution of water and flucturation of distance
    between the metal and water layers; at present, only the Pt(100) and Pt(111) data are added
"""
class ElectrodeWaterInterface:

    def __init__(self, metal, facet, T = 298.15):
        self.metal = metal
        self.facet = facet
        self.T = T
        
    def get_volmer_parameters(self, coverage, U, pH, d_ew, E_H_OH = -6.614, show_details = False):
        
        E_H_ad_metal = coverage_dependence(metal = self.metal, facet=self.facet, adsorbate = 'H',theta = coverage) 
        E_H_OH_corrected = E_H_OH - OH_solvation_eng - (U- k_b*self.T*np.log(10)*pH)+0.3 # U is v.s. RHE
        d_ew_corrected = d_ew - H_O_bond_length-H_metal_distance['H|%s_%s'%(self.metal,self.facet)]
        
        pes_H_on_Water = PES.init_from_parameters(De_U0=np.abs(E_H_OH_corrected),a=morse_a_dict['a_H_on_water'],position='left',deq=0)
        pes_H_on_metal = PES.init_from_parameters(De_U0=np.abs(E_H_ad_metal),a=morse_a_dict['a_H_on_%s_%s'%(self.metal, self.facet)],position='right',deq=d_ew_corrected)
        e = Energy(pes_H_on_Water, pes_H_on_metal)
        delta_Ea_vol=0.0 # for sensitivity study, set this variable to be a very small value, e.g., 0.001, to check the change of total rate; See  Campbell's paper https://doi.org/10.1021/acscatal.7b00115     
        barrier_di_f = e.Ea_di_left+delta_Ea_vol
        barrier_di_r = e.Ea_di_right-delta_Ea_vol
        barrier_ad_f = e.Ea_ad_left+delta_Ea_vol
        barrier_ad_r = e.Ea_ad_right-delta_Ea_vol
        
        k_di_f  = attempt_frequency*np.exp(-barrier_di_f/(k_b*self.T))  
        k_di_r  = attempt_frequency*np.exp(-barrier_di_r/(k_b*self.T))
        k_ad_f  = attempt_frequency*np.exp(-barrier_ad_f/(k_b*self.T))
        k_ad_r  = attempt_frequency*np.exp(-barrier_ad_r/(k_b*self.T))
        
        if show_details == True:
            e.plot_intercepts(adiabatic=True, xlim=(0.0, d_ew_corrected),ylim=(-6, 0))
            print('energy difference: %5.2f V'%(E_H_ad_metal-E_H_OH_corrected))
            print('E_H-OH in water: %5.2f' %E_H_OH_corrected)
            print('E_H on metal: %5.2f' %E_H_ad_metal)
            print('d_O-metal: %5.2f' %d_ew)
            print('bond_length_O-H: %5.2f' %H_O_bond_length)
            print('bond_length_H-metal: %5.2f' %H_metal_distance['H|%s_%s'%(self.metal,self.facet)])
            print('final_distance_metal-H: %5.2f' %d_ew_corrected)

        return {'delta G': E_H_ad_metal-E_H_OH_corrected,
                'barrier_di_forward':barrier_di_f,
                'barrier_di_reverse':barrier_di_r,
                'barrier_ad_forward':barrier_ad_f,
                'barrier_ad_reverse':barrier_ad_r,
                'k_di_forward':k_di_f,
                'k_di_reverse':k_di_r,
                'k_ad_forward':k_ad_f,
                'k_ad_reverse':k_ad_r}
    
    def get_heyrovsky_parameters(self, coverage, U, pH, d_ew, E_H_OH, show_details = False):
        
        E_H_ad_H_metal = coverage_dependence(metal = self.metal, facet=self.facet, adsorbate = 'H@H',theta = coverage) 
        E_H_OH_corrected = E_H_OH - OH_solvation_eng - (U- k_b*self.T*np.log(10)*pH)+0.3 # U is v.s. RHE
        d_ew_corrected = d_ew - H_O_bond_length- H_H_distance -H_metal_distance['H|%s_%s'%(self.metal,self.facet)]
        if d_ew_corrected < 0:
            d_ew_corrected = 0.001 # for heyrovsky step, if the metal-water distance is too close, the barrier will be dG
        
        pes_H_on_Water = PES.init_from_parameters(De_U0=np.abs(E_H_OH_corrected), a=morse_a_dict['a_H_on_water'],position='left',deq=0)
        pes_H_on_metal = PES.init_from_parameters(De_U0=np.abs(E_H_ad_H_metal), a=morse_a_dict['a_H_on_H_%s_%s'%(self.metal, self.facet)],position='right',deq=d_ew_corrected)
        e = Energy(pes_H_on_Water, pes_H_on_metal)
        delta_Ea_hey=0.0  # for sensitivity study, set this variable to be a very small value, e.g., 0.001, to check the change of total rate; See  Campbell's paper https://doi.org/10.1021/acscatal.7b00115
        barrier_di_f = e.Ea_di_left+delta_Ea_hey
        barrier_di_r = e.Ea_di_right-delta_Ea_hey
        barrier_ad_f = e.Ea_ad_left+delta_Ea_hey
        barrier_ad_r = e.Ea_ad_right-delta_Ea_hey
        
        k_di_f  = attempt_frequency*np.exp(-barrier_di_f/(k_b*self.T))  
        k_di_r  = attempt_frequency*np.exp(-barrier_di_r/(k_b*self.T))
        k_ad_f  = attempt_frequency*np.exp(-barrier_ad_f/(k_b*self.T))
        k_ad_r  = attempt_frequency*np.exp(-barrier_ad_r/(k_b*self.T))
        
        if show_details == True:
            e.plot_intercepts(adiabatic=True, xlim=(0.0, d_ew_corrected),ylim=(-6, 0))
            print('energy difference: %5.2f V'%(E_H_ad_H_metal-E_H_OH_corrected))
            print('E_H-OH in water: %5.2f' %E_H_OH_corrected)
            print('E_H_ad_H_metal: %5.2f' %E_H_ad_H_metal)
            print('d_O-metal: %5.2f' %d_ew)
            print('bond_length_O-H: %5.2f' %H_O_bond_length)
            print('bond_length_H-metal: %5.2f' %H_metal_distance['H|%s_%s'%(self.metal,self.facet)])
            print('bond_length_H-H: %5.2f' %H_H_distance)
            print('final_distance_metal-H: %5.2f' %d_ew_corrected)

        return {'barrier_di_forward':barrier_di_f,
                'barrier_di_reverse':barrier_di_r,
                'barrier_ad_forward':barrier_ad_f,
                'barrier_ad_reverse':barrier_ad_r,
                'k_di_forward':k_di_f,
                'k_di_reverse':k_di_r,
                'k_ad_forward':k_ad_f,
                'k_ad_reverse':k_ad_r}
    
    def get_tafel_parameters(self, coverage, show_details = False):
        
        E_H_ad_metal = coverage_dependence(metal = self.metal, facet=self.facet, adsorbate = 'H',theta = coverage)
        delta_E = H2_referenced_to_H_vaccum - E_H_ad_metal*2
        delta_Ea_taf=0.0  # for sensitivity study, set this variable to be a very small value, e.g., 0.001, to check the change of total rate; See  Campbell's paper https://doi.org/10.1021/acscatal.7b00115
        if self.metal=='Pt' and self.facet=='100':
            barrier_forward = -0.992*coverage + 1.51 + 0.229+delta_Ea_taf
        elif self.metal=='Pt' and self.facet=='111':
            barrier_forward = -0.4157*coverage + 0.7628 + 0.229 + delta_Ea_taf 
        
        if barrier_forward <0:
            barrier_forward =0
            
        elif 0<barrier_forward<delta_E:
            barrier_forward = delta_E 
            
        barrier_reverse = barrier_forward - (delta_E)  # delta_E is free energy difference
        k_forward  = attempt_frequency*np.exp(-barrier_forward/(k_b*self.T))  
        k_reverse  = attempt_frequency*np.exp(-barrier_reverse/(k_b*self.T))
        
        if show_details == True:
            print('2*E_H_ad_metal: %5.2f eV' %(2*E_H_ad_metal))
            print('H2_referenced_to_H_vaccum: %5.2f eV' %H2_referenced_to_H_vaccum)
            print('delta_E: %5.2f eV' %delta_E)
            print('forward barrier: %5.2f eV'%barrier_forward)
            print('reverse barrier: %5.2f eV'%barrier_reverse)
            
        return {'barrier_forward':barrier_forward,
                'barrier_reverse':barrier_reverse,
                'k_forward':k_forward,
                'k_reverse':k_reverse}
    
    def coverage_iteration(self, U, pH, d_ew, E_H_OH, barrier_type = 'ad_diabetic', 
                           initial_coverage = 0.7, iteration_number = 100, mixing = 0.2, 
                           precision=0.001, show_details = False):
        
        U_RHE = U
        U_SHE = -k_b*self.T*pH*np.log(10) + U_RHE  
        theta_H_0 = initial_coverage
        iteration_number = iteration_number
        
        for i in range(iteration_number):
            volmer_parameters = self.get_volmer_parameters(coverage = theta_H_0, U = U_RHE, pH = pH, d_ew = d_ew, E_H_OH = E_H_OH)
            heyrov_parameters = self.get_heyrovsky_parameters(coverage = theta_H_0, U = U_RHE, pH = pH, d_ew = d_ew, E_H_OH = E_H_OH)
            tafel_parameters = self.get_tafel_parameters(coverage = theta_H_0)
            
            if barrier_type == 'ad_diabetic':
                #volmer step forward and reverse ad-diabetic barriers
                barrier_volmer_f = volmer_parameters['barrier_ad_forward']
                barrier_volmer_r = volmer_parameters['barrier_ad_reverse']
                
                #herrovsky step forward and reverse ad-diabetic barriers
                barrier_heyrov_f = heyrov_parameters['barrier_ad_forward']
                barrier_heyrov_r = heyrov_parameters['barrier_ad_reverse']
                
                #tafel step forward and reverse barriers from NEB calculations
                barrier_tafel_f = tafel_parameters['barrier_forward']
                barrier_tafel_r = tafel_parameters['barrier_reverse']
                
                #volmer step forward and reverse rate constant without attempt frequency
                k_volmer_f = volmer_parameters['k_ad_forward']
                #k_volmer_r = volmer_parameters['k_ad_reverse']
                k_volmer_r = 0
                
                #heyrovsky step forward and reverse rate constant without attempt frequency
                k_heyrov_f = heyrov_parameters['k_ad_forward'] 
                #k_heyrov_r = heyrov_parameters['k_ad_reverse']
                k_heyrov_r = 0  # the reverse reaction actually never happens because OH- always transfers to the bulk water once it forms
                
                #tafel step forward and reverse rate constant without attempt frequency
                k_tafel_f = tafel_parameters['k_forward']
                k_tafel_r = tafel_parameters['k_reverse']
                #k_tafel_r= 0 # ignore the oxidation of H2, i.e., reverse reaction of Tafel step
            if barrier_type == 'di_diabetic':
                #volmer step forward and reverse di-diabetic barriers
                barrier_volmer_f = volmer_parameters['barrier_di_forward']
                barrier_volmer_r = volmer_parameters['barrier_di_reverse']
                
                #herrovsky step forward and reverse di-diabetic barriers
                barrier_heyrov_f = heyrov_parameters['barrier_di_forward']
                barrier_heyrov_r = heyrov_parameters['barrier_di_reverse']
                
                #tafel step forward and reverse barriers from NEB calculations
                barrier_tafel_f = tafel_parameters['barrier_forward']
                barrier_tafel_r = tafel_parameters['barrier_reverse']
                
                #volmer step forward and reverse rate constant with out attempt frequency
                k_volmer_f = volmer_parameters['k_di_forward']
                #k_volmer_r = volmer_parameters['k_di_reverse']
                k_volmer_r = 0
                
                #heyrovsky step forward and reverse rate constant with out attempt frequency
                k_heyrov_f = heyrov_parameters['k_di_forward'] 
                #k_heyrov_r = heyrov_parameters['k_di_reverse']
                k_heyrov_r = 0 # the reverse reaction actually never happens because OH- always transfers to the bulk water once it forms
                
                #tafel step forward and reverse rate constant with out attempt frequency
                k_tafel_f = tafel_parameters['k_forward']
                k_tafel_r = tafel_parameters['k_reverse']
                #k_tafel_r= 0 # ignore the oxidation of H2, i.e., reverse reaction of Tafel step
            
            if iteration_number == 1:  # when runing simulate_IV with adjust= True, we return the following without iteration
                return {'theta_H':theta_H_0,
                        'barrier_volmer_f':barrier_volmer_f,'barrier_volmer_r':barrier_volmer_r,
                        'barrier_heyrov_f':barrier_heyrov_f,'barrier_heyrov_r':barrier_heyrov_r,
                        'barrier_tafel_f':barrier_tafel_f,'barrier_tafel_r':barrier_tafel_r,
                        'k_volmer_f':k_volmer_f,'k_volmer_r':k_volmer_r,
                        'k_heyrov_f':k_heyrov_f,'k_heyrov_r':k_heyrov_r,
                        'k_tafel_f':k_tafel_f,'k_tafel_r':k_tafel_r}
                break
            
            else:
                pass
                 
            theta_H = get_coverage(k_volmer_f,k_volmer_r, k_heyrov_f, k_heyrov_r,k_tafel_f, k_tafel_r,pH=pH)  
            differ = theta_H-theta_H_0
            
            if show_details == True:
                print(theta_H_0, theta_H)
            if differ > 0:
                theta_H_0 += mixing*differ
            elif differ < 0:
                theta_H_0 = theta_H + mixing*differ
            if theta_H_0 < 0:
                theta_H_0 = 0
            if np.abs(differ) <= precision:
                print('coverage at U=%5.2f V iteration complete'%(U_RHE))
                print(theta_H_0, theta_H)
                
                return {'theta_H':theta_H_0,
                        'barrier_volmer_f':barrier_volmer_f,'barrier_volmer_r':barrier_volmer_r,
                        'barrier_heyrov_f':barrier_heyrov_f,'barrier_heyrov_r':barrier_heyrov_r,
                        'barrier_tafel_f':barrier_tafel_f,'barrier_tafel_r':barrier_tafel_r,
                        'k_volmer_f':k_volmer_f,'k_volmer_r':k_volmer_r,
                        'k_heyrov_f':k_heyrov_f,'k_heyrov_r':k_heyrov_r,
                        'k_tafel_f':k_tafel_f,'k_tafel_r':k_tafel_r}
                break
            
            if i ==(iteration_number -1) and np.abs(differ) >= precision:
                print('coverage iteration at U_RHE=%5.2f is not complete'%(U_RHE))
                print(theta_H_0, theta_H)
                
                return {'theta_H':initial_coverage,
                        'barrier_volmer_f':barrier_volmer_f,'barrier_volmer_r':barrier_volmer_r,
                        'barrier_heyrov_f':barrier_heyrov_f,'barrier_heyrov_r':barrier_heyrov_r,
                        'barrier_tafel_f':barrier_tafel_f,'barrier_tafel_r':barrier_tafel_r,
                        'k_volmer_f':k_volmer_f,'k_volmer_r':k_volmer_r,
                        'k_heyrov_f':k_heyrov_f,'k_heyrov_r':k_heyrov_r,
                        'k_tafel_f':k_tafel_f,'k_tafel_r':k_tafel_r}

    def get_distributions(self, barrier_type = 'ad_diabetic', E_H_OH = -6.614, range_U = (-0.2,0), U_num = 10, 
                      pH = 4, initial_coverage = 0.7, iteration_number = 100, 
                      mixing = 0.02, precision=0.001, coverage_input = None, N_distribution=200):
        
        U_list = list(np.linspace(range_U[0],range_U[1],U_num))
        d_ew_distribution=cov2distribution_Pt_100(initial_coverage,N_sam=N_distribution)
        d_ew_list=d_ew_distribution['d_ew_list']
        X_U, Y_d_ew = np.meshgrid(U_list, d_ew_list) # varying metal-water distance and U_RHE
        row_num, col_num = np.shape(X_U)
        Z = np.zeros((row_num, col_num))
        Z_theta_H = deepcopy(Z)
        
        Z_barrier_volmer_f = deepcopy(Z)
        Z_barrier_volmer_r = deepcopy(Z)
        Z_barrier_heyrov_f = deepcopy(Z)
        Z_barrier_heyrov_r = deepcopy(Z)
        Z_barrier_tafel_f = deepcopy(Z)
        Z_barrier_tafel_r = deepcopy(Z)
        
        Z_k_volmer_f = deepcopy(Z)
        Z_k_volmer_r = deepcopy(Z)
        Z_k_heyrov_f = deepcopy(Z)
        Z_k_heyrov_r = deepcopy(Z)
        Z_k_tafel_f = deepcopy(Z)
        Z_k_tafel_r = deepcopy(Z)
        
        Z_k_volmer_f_probability_corrected = deepcopy(Z)
        Z_k_volmer_r_probability_corrected = deepcopy(Z)
        Z_k_heyrov_f_probability_corrected = deepcopy(Z)
        Z_k_heyrov_r_probability_corrected = deepcopy(Z)
        Z_k_tafel_f_probability_corrected = deepcopy(Z)
        Z_k_tafel_r_probability_corrected = deepcopy(Z)
        
        theta_H_0 = initial_coverage
        d_ew_probabilities = []
        for i in tqdm(range(row_num)):
            d_ew_P_list=[]
            for j in range(col_num):
                u = X_U[i,j]
                d_ew = Y_d_ew[i,j]
                
                if coverage_input is not None: # this is used to get the simulate_IV when adjust = True
                    test_r, test_c = np.shape(coverage_input)
                    if test_r != row_num or test_c != col_num:
                        print('Warning: the coverage_input is in a wrong dimension')
                    else:
                        iteration_number = 1 
                        theta_H_0 = coverage_input[i,j]
                
                iteration = self.coverage_iteration(U = X_U[i,j], pH = pH, d_ew = Y_d_ew[i,j], E_H_OH = E_H_OH, 
                                                    barrier_type = barrier_type, initial_coverage = theta_H_0, 
                                                    iteration_number = iteration_number, mixing = mixing, 
                                                    precision=precision, show_details = False)
                theta_H = iteration['theta_H']
                d_ew_distribution=cov2distribution_Pt_100(theta_H,N_sam=N_distribution)
                P_dew=d_ew_distribution['Probability'][i]
                d_ew_P_list.append(P_dew)
                
                barrier_volmer_f = iteration['barrier_volmer_f']
                barrier_volmer_r = iteration['barrier_volmer_r']
                barrier_heyrov_f = iteration['barrier_heyrov_f']
                barrier_heyrov_r = iteration['barrier_heyrov_r']
                barrier_tafel_f = iteration['barrier_tafel_f']
                barrier_tafel_r = iteration['barrier_tafel_r']
                
                k_volmer_f = iteration['k_volmer_f']
                k_volmer_r = iteration['k_volmer_r']
                k_heyrov_f = iteration['k_heyrov_f']
                k_heyrov_r = iteration['k_heyrov_r']
                k_tafel_f = iteration['k_tafel_f']
                k_tafel_r = iteration['k_tafel_r']
                
                Z_theta_H[i,j] = theta_H
                
                Z_barrier_volmer_f[i,j] = barrier_volmer_f
                Z_barrier_volmer_r[i,j] = barrier_volmer_r
                Z_barrier_heyrov_f[i,j] = barrier_heyrov_f
                Z_barrier_heyrov_r[i,j] = barrier_heyrov_r
                Z_barrier_tafel_f[i,j] = barrier_tafel_f
                Z_barrier_tafel_r[i,j] = barrier_tafel_r
                
                Z_k_volmer_f[i,j] = k_volmer_f
                Z_k_volmer_r[i,j] = k_volmer_r
                Z_k_heyrov_f[i,j] = k_heyrov_f
                Z_k_heyrov_r[i,j] = k_heyrov_r
                Z_k_tafel_f[i,j] = k_tafel_f
                Z_k_tafel_r[i,j] = k_tafel_r
                
                theta_H_0 = theta_H
            d_ew_probabilities.append(d_ew_P_list)
        
        d_ew_P_array=np.array(d_ew_probabilities)
        
        # correct the reaction constant with probabilies of metal-water distances
        for i in range(row_num):
            for j in range(col_num):
                Z_k_volmer_f_probability_corrected[i,j] = Z_k_volmer_f[i,j]*d_ew_P_array[i,j]
                Z_k_volmer_r_probability_corrected[i,j] = Z_k_volmer_r[i,j]*d_ew_P_array[i,j]
                Z_k_heyrov_f_probability_corrected[i,j] = Z_k_heyrov_f[i,j]*d_ew_P_array[i,j]
                Z_k_heyrov_r_probability_corrected[i,j] = Z_k_heyrov_r[i,j]*d_ew_P_array[i,j]
                Z_k_tafel_f_probability_corrected[i,j] = Z_k_tafel_f[i,j]*d_ew_P_array[i,j]
                Z_k_tafel_r_probability_corrected[i,j] = Z_k_tafel_r[i,j]*d_ew_P_array[i,j]
            
        return {'X_U':X_U, 'Y_d_ew':Y_d_ew, 'd_ew_probablities':d_ew_P_array,
                'Z_theta_H':Z_theta_H, 
                'Z_barrier_volmer_f':Z_barrier_volmer_f,'Z_barrier_volmer_r':Z_barrier_volmer_r, 
                'Z_barrier_heyrov_f':Z_barrier_heyrov_f, 'Z_barrier_heyrov_r':Z_barrier_heyrov_r,
                'Z_barrier_tafel_f':Z_barrier_tafel_f, 'Z_barrier_tafel_r':Z_barrier_tafel_r,
                'Z_k_volmer_f':Z_k_volmer_f, 'Z_k_volmer_r': Z_k_volmer_r,
                'Z_k_heyrov_f':Z_k_heyrov_f,'Z_k_heyrov_r':Z_k_heyrov_r,
                'Z_k_tafel_f':Z_k_tafel_f, 'Z_k_tafel_r':Z_k_tafel_r,
                'Z_k_volmer_f_probability_corrected':Z_k_volmer_f_probability_corrected,
                'Z_k_volmer_r_probability_corrected':Z_k_volmer_r_probability_corrected,
                'Z_k_heyrov_f_probability_corrected':Z_k_heyrov_f_probability_corrected,
                'Z_k_heyrov_r_probability_corrected':Z_k_heyrov_r_probability_corrected,
                'Z_k_tafel_f_probability_corrected':Z_k_tafel_f_probability_corrected,
                'Z_k_tafel_r_probability_corrected':Z_k_tafel_r_probability_corrected}

    def simulate_IV(self, barrier_type = 'ad_diabetic',range_U = (-0.2,0), U_ref='RHE', U_num = 10, pH = 4, P_H2=0,initial_coverage = 0.5, iteration_number = 100,
                   mixing = 0.02, precision=0.001, adjust = False, plot = False,save_fig=False, H2O_var = 0): 
        sigma = 0.11 #standard deviation of the water distribution
        E_H_OH_mean = -6.614 #expectation value of H-OH binding energy of the water distribution
        E_H_OH = E_H_OH_mean+H2O_var*sigma
        U0 = np.linspace(range_U[0],range_U[1],U_num)
        if U_ref=='SHE':
            U = U0+k_b*self.T*pH*np.log(10)
        else:
            U= U0
        print('range_U_RHE',np.min(U),np.max(U))
        distributions = self.get_distributions(barrier_type = barrier_type, E_H_OH = E_H_OH, range_U = (np.min(U),np.max(U)), U_num = U_num, 
                              pH = pH, initial_coverage = initial_coverage, iteration_number = iteration_number, mixing = mixing, precision = precision)
        
        H_coverage = distributions['Z_theta_H']
        k_volmer_f_probability_corrected = distributions['Z_k_volmer_f_probability_corrected']
        k_volmer_r_probability_corrected = distributions['Z_k_volmer_r_probability_corrected']
        k_heyrov_f_probability_corrected = distributions['Z_k_heyrov_f_probability_corrected']
        k_heyrov_r_probability_corrected = distributions['Z_k_heyrov_r_probability_corrected']
        k_tafel_f_probability_corrected = distributions['Z_k_tafel_f_probability_corrected']
        k_tafel_r_probability_corrected = distributions['Z_k_tafel_r_probability_corrected']
        Prob_matrix= distributions['d_ew_probablities']
        
        row_num, col_num = np.shape(H_coverage)
        TOF_H2O_volmer = np.zeros((row_num, col_num))
        TOF_H2_heyrov = np.zeros((row_num, col_num))
        TOF_H2_tafel = np.zeros((row_num, col_num))
        J_volmer = np.zeros((row_num, col_num))
        J_heyrov = np.zeros((row_num, col_num))
        J_tafel = np.zeros((row_num, col_num))

        for i in range(row_num):
            for j in range(col_num):
                TOF_H2O_volmer[i,j] = k_volmer_f_probability_corrected[i,j]*(1-H_coverage[i,j]) - k_volmer_r_probability_corrected[i,j]*H_coverage[i,j]*10**(pH-14)
                TOF_H2_heyrov[i,j] = k_heyrov_f_probability_corrected[i,j]*H_coverage[i,j] - k_heyrov_r_probability_corrected[i,j]*(1-H_coverage[i,j])
                TOF_H2_tafel[i,j] = k_tafel_f_probability_corrected[i,j]*H_coverage[i,j]**2 - P_H2*k_tafel_r_probability_corrected[i,j]*(1-H_coverage[i,j])**2 # assume the H2 partial pressure at the interface is close to 0
                
        TOF_H2O_volmer_final = np.zeros(col_num)
        TOF_H2_heyrov_final = np.zeros(col_num)
        TOF_H2_tafel_final = np.zeros(col_num)
        
        for j in range(col_num):
            TOF_H2O_volmer_final[j] = np.sum(TOF_H2O_volmer[:,j])
            TOF_H2_heyrov_final[j] = np.sum(TOF_H2_heyrov[:,j])
            TOF_H2_tafel_final[j] = np.sum(TOF_H2_tafel[:,j])
        
        for i in range(len(TOF_H2_tafel_final)): 
            if TOF_H2_tafel_final[i] < 0:
                TOF_H2_tafel_final[i] = 0  # minus TOF_H2 means H2 dissociating to H*, but this step dosn't transfer electron

        TOF_H2_total = TOF_H2_heyrov_final + TOF_H2_tafel_final
        J_volmer = (1*TOF_H2O_volmer_final)*e_charge*electrode_reaction_sites['%s'%(self.metal+self.facet)]/(electrode_area['%s'%(self.metal+self.facet)]*sq_angstrom_2_sq_centimeter)
        J_heyrov = (2*TOF_H2_heyrov_final)*e_charge*electrode_reaction_sites['%s'%(self.metal+self.facet)]/(electrode_area['%s'%(self.metal+self.facet)]*sq_angstrom_2_sq_centimeter)# unit is A/cm2
        J_tafel = (2*TOF_H2_tafel_final)*e_charge*electrode_reaction_sites['%s'%(self.metal+self.facet)]/(electrode_area['%s'%(self.metal+self.facet)]*sq_angstrom_2_sq_centimeter)
        J_total = J_heyrov+J_tafel
        #J_total = J_volmer+0.5*J_heyrov
        #J_tafel = J_volmer-0.5*J_heyrov
        
        U_RHE = U
        U_SHE = U_RHE-k_b*self.T*pH*np.log(10) 
        
        if plot == True:
            fig, ax1 = plt.subplots()
            if U_ref=='SHE':
                j_hyr = ax1.scatter(U_SHE,-1000*J_heyrov,marker = '^',color="darkred",linewidths=1,s=35,edgecolors='white',label='j-Heyrovsky')
                j_taf = ax1.scatter(U_SHE,-1000*J_tafel,marker = 'o',color="blue",linewidths=1,s=35,edgecolors='white',label='j-Tafel')
                j_total = ax1.plot(U_SHE,-1000*J_total,'g-', lw =1,label='j-Total')
                ax1.set_ylabel('$j$ (mA cm$^{-2}$)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
                ax1.set_xlabel('$E$ (V vs. SHE)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
            else:
                j_hyr = ax1.scatter(U_RHE,-1000*J_heyrov,marker = '^',color="darkred",linewidths=1,s=35,edgecolors='white',label='j-Heyrovsky')
                j_taf = ax1.scatter(U_RHE,-1000*J_tafel,marker = 'o',color="blue",linewidths=1,s=35,edgecolors='white',label='j-Tafel')
                j_total = ax1.plot(U_RHE,-1000*J_total,'g-', lw =1,label='j-Total')
                ax1.set_ylabel('$j$ (mA cm$^{-2}$)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
                ax1.set_xlabel('$E$ (V vs. RHE)', fontdict={'family' : 'Times New Roman', 'size'   : 22})

            #ax1.set_xlim([-0.5, 1.0])
            #ax1.set_ylim([0, 1])
            plt.yticks(fontproperties = 'Times New Roman', size = 22)
            plt.xticks(fontproperties = 'Times New Roman', size = 22)
            plt.title('I-V curve')
            plt.legend()
            if save_fig:
                plt.savefig('IV_curve.pdf')
            plt.show()

        
        if  adjust == True:
            H_coverage_origin = deepcopy(H_coverage)
            row_num, col_num = np.shape(H_coverage_origin) # row_num is the number of distance; col_num is the number of U values
            H_coverage_fitted = []
            for i in range(row_num):
                para_f = parabola_fitting(U_SHE, H_coverage_origin[i]) # use parabola function to fit the H_coverage
                H_cov = parabola_function(U_SHE, para_f[0], para_f[1],para_f[2])
                H_coverage_fitted.append(H_cov)
            
            H_coverage_array = np.array(H_coverage_fitted)
            distributions = self.get_distributions(barrier_type = barrier_type, E_H_OH = E_H_OH, range_U = (np.min(U),np.max(U)), U_num = U_num, 
                            pH = pH, initial_coverage = initial_coverage, iteration_number = iteration_number, 
                            mixing = mixing, precision = precision, coverage_input = H_coverage_array)
        
            H_coverage = distributions['Z_theta_H']
            k_heyrov_f_probability_corrected = distributions['Z_k_heyrov_f_probability_corrected']
            k_heyrov_r_probability_corrected = distributions['Z_k_heyrov_r_probability_corrected']
            k_tafel_f_probability_corrected = distributions['Z_k_tafel_f_probability_corrected']
            k_tafel_r_probability_corrected = distributions['Z_k_tafel_r_probability_corrected']
            
            row_num, col_num = np.shape(H_coverage)
            TOF_H2_heyrov = np.zeros((row_num, col_num))
            TOF_H2_tafel = np.zeros((row_num, col_num))
            J_heyrov = np.zeros((row_num, col_num))
            J_tafel = np.zeros((row_num, col_num))
            
            for i in range(row_num):
                for j in range(col_num):
                    TOF_H2_heyrov[i,j] = k_heyrov_f_probability_corrected[i,j]*H_coverage[i,j] - k_heyrov_r_probability_corrected[i,j]*(1-H_coverage[i,j])
                    TOF_H2_tafel[i,j] = k_tafel_f_probability_corrected[i,j]*H_coverage[i,j]**2 - P_H2*k_tafel_r_probability_corrected[i,j]*(1-H_coverage[i,j])**2

            TOF_H2_heyrov_final = np.zeros(col_num)
            TOF_H2_tafel_final = np.zeros(col_num)

            for j in range(col_num):
                TOF_H2_heyrov_final[j] = np.sum(TOF_H2_heyrov[:,j])
                TOF_H2_tafel_final[j] = np.sum(TOF_H2_tafel[:,j])

            for i in range(len(TOF_H2_tafel_final)): 
                if TOF_H2_tafel_final[i] < 0:
                    TOF_H2_tafel_final[i] = 0  # minus TOF_H2 means H2 dissociating to H*, but this step dosn't transfer electron

            TOF_H2_total = TOF_H2_heyrov_final + TOF_H2_tafel_final
            J_heyrov = (2*TOF_H2_heyrov_final)*e_charge*electrode_reaction_sites['%s'%(self.metal+self.facet)]/(electrode_area['%s'%(self.metal+self.facet)]*sq_angstrom_2_sq_centimeter)# unit is A/cm2
            J_tafel = (2*TOF_H2_tafel_final)*e_charge*electrode_reaction_sites['%s'%(self.metal+self.facet)]/(electrode_area['%s'%(self.metal+self.facet)]*sq_angstrom_2_sq_centimeter)
            J_total = J_heyrov + J_tafel

        
        if plot == True and  adjust == True:
            print('adjusted I-V curve:')
            #fig, ax1 = plt.subplots(figsize=fig_size,dpi=300)
            fig, ax1 = plt.subplots()
            if U_ref=='SHE':
                j_hyr = ax1.scatter(U_SHE,-1000*J_heyrov,marker = '^',color="darkred",linewidths=1,s=35,edgecolors='white',label='j-Heyrovsky')
                j_taf = ax1.scatter(U_SHE,-1000*J_tafel,marker = 'o',color="blue",linewidths=1,s=35,edgecolors='white',label='j-Tafel')
                j_total = ax1.plot(U_SHE,-1000*J_total,'orange', linestyle='-', lw =1,label='j-Total')
                ax1.set_xlabel('$E$ (V vs. SHE)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
                ax1.set_ylabel('$j$ (mA cm$^{-2}$)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
            else:
                j_hyr = ax1.scatter(U_RHE,-1000*J_heyrov,marker = '^',color="darkred",linewidths=1,s=35,edgecolors='white',label='j-Heyrovsky')
                j_taf = ax1.scatter(U_RHE,-1000*J_tafel,marker = 'o',color="blue",linewidths=1,s=35,edgecolors='white',label='j-Tafel')
                j_total = ax1.plot(U_RHE,-1000*J_total,'orange', linestyle='-', lw =1,label='j-Total')
                ax1.set_xlabel('$E$ (V vs. RHE)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
                ax1.set_ylabel('$j$ (mA cm$^{-2}$)', fontdict={'family' : 'Times New Roman', 'size'   : 22})
            #ax1.set_xlim([-0.5, 1.0])
            #ax1.set_ylim([0, 1])
            plt.yticks(fontproperties = 'Times New Roman', size = 22)
            plt.xticks(fontproperties = 'Times New Roman', size = 22)
            plt.title('I-V curve')
            plt.legend()
            if save_fig:
                plt.savefig('IV_curve_adjusted.pdf')
            plt.show()   
        
        return {'U_RHE': U, 'U_SHE': U_SHE, 'H_coverage':H_coverage,'Prob_matrix':Prob_matrix,
                'k_heyrov_f':k_heyrov_f_probability_corrected, 'k_heyrov_r':k_heyrov_r_probability_corrected,
                'k_tafel_f':k_tafel_f_probability_corrected, 'k_tafel_r':k_tafel_r_probability_corrected,
                'TOF_H2_heyrov_2D':TOF_H2_heyrov,'TOF_H2_tafel_2D':TOF_H2_tafel,
                'TOF_H2_heyrov_final':TOF_H2_heyrov_final, 'TOF_H2_tafel_final':TOF_H2_tafel_final, 'TOF_H2_total':TOF_H2_total,
                'J_heyrov': J_heyrov, 'J_tafel': J_tafel, 'J_total':J_total}
