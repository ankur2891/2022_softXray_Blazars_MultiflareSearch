#!/usr/bin/env python

import numpy as np
import astropy
import matplotlib.pyplot as plt
from scipy import stats
import os, sys
import glob
import histlite as hl
import csky as cy


import os
import pandas as pd
import random 
from astropy.coordinates import SkyCoord
import pickle

from csky.ipyconfig import *
from csky import *
import matplotlib

import math
from scipy.integrate import quad

from scipy.interpolate import make_interp_spline, BSpline
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import stats

cy.plotting.mrichman_mpl()

timer = cy.timing.Timer()
time = timer.time


# load the catalog and downselect to 1000 sources
roma_bzcat = "/home/asharma/csky/roma_bzcat5.txt"

romabzcat = pd.read_csv(roma_bzcat, sep='[|]', header=0, comment='#', index_col=None, skip_blank_lines=True,  engine='python')

#romabzcat.drop(romabzcat.columns[0], inplace=True, axis=1)
#romabzcat.drop(romabzcat.columns[13], inplace=True, axis=1)
romabzcat.rename(columns={'id ': 'id', ' BZCAT5 Source name ': 'BZCAT5_Source_name', ' Other name ':'Other_name', ' RA (J2000.0) ': 'RA',\
                        ' Dec (J2000.0) ': 'Dec', ' Redshift ': 'Redshift', ' Redshift_flag ': 'Redshift_flag', ' Rmag ': 'Rmag',\
                        ' Source classification ': 'Category', ' Flux density 1.4/0.843GHz(mJy) ': 'Flux_density_1p4_0p843_GHz',\
                         ' Flux density 5.0GHz(mJy) ': 'Flux_density_5GHz', ' Flux density143GHz(mJy) ': 'Flux_density_143GHz',\
                         ' X-ray flux0.1-2.4 keV(1.e-12 cgs) ': 'Xflux_0p1_2p4_keV_cgs', ' Fermi flux1-100 GeV(ph/cm2/s) ': \
                          'Fermi_flux_1_100_GeV_phcms', ' aro-aox-arx': 'idx_ro_ox_rx'}, inplace=True)

#convert columns from string to float and select sources with X-ray flux present, non-zero redshift, dec (-85, 85)
romabzcat['Xflux_0p1_2p4_keV_cgs']=romabzcat.Xflux_0p1_2p4_keV_cgs.replace(' ', 0.0).astype(float)
romabzcat.Xflux_0p1_2p4_keV_cgs = romabzcat.Xflux_0p1_2p4_keV_cgs.astype(float)

romabzcat['Redshift']=romabzcat.Redshift.replace(' ', 0.0).astype(float)
romabzcat.Redshift = romabzcat.Redshift.astype(float)

romabzcat = romabzcat[romabzcat.Xflux_0p1_2p4_keV_cgs > 0]
#romabzcat = romabzcat[romabzcat.Redshift > 0]
romabzcat = romabzcat[(romabzcat.Dec > -85) & (romabzcat.Dec < 85)]


romabzcat['Flux_density_1p4_0p843_GHz']=romabzcat.Flux_density_1p4_0p843_GHz.replace(' ', 0.0).astype(float)
romabzcat.Flux_density_1p4_0p843_GHz = romabzcat.Flux_density_1p4_0p843_GHz.astype(float)
romabzcat['Flux_density_1p4_0p843_GHz'].dtype

romabzcat['Fermi_flux_1_100_GeV_phcms']=romabzcat.Fermi_flux_1_100_GeV_phcms.replace(' ', 0.0).astype(float)
romabzcat.Fermi_flux_1_100_GeV_phcms = romabzcat.Fermi_flux_1_100_GeV_phcms.astype(float)

#Top 1K sources from Northern hemisphere catalog
all_north = romabzcat[romabzcat['Dec'] > -5.]  # all blazars in Northern sky
top1k_north = all_north.sort_values(by='Xflux_0p1_2p4_keV_cgs', ascending=False)
top1k_north = top1k_north[0:1000]
top1k_north.reset_index(drop=True, inplace=True)

top1k_srclist = cy.utils.Sources(ra = np.radians(top1k_north.RA), dec = np.radians(top1k_north.Dec), name = top1k_north.Other_name)


def BinomialTest(pValues, kmax, returnArray=True):
    '''
    This function is used to perform the Binomial test for a list of p-values.
    
    Input: 
        pValues: array of p-values
        kmax: int, the kmax to be used for the Binomial test
        returnArray: True if you want the array containing the computed Binomial Probabilities at each k.
        
    Output:
        pThresh: The local p-value at 'k' where we have the Best (lowest) Binomial Probabilitiy for the Binomial test.
        kBest: (int) The value of 'k' at which we have the Best (lowest) Binomial Probabilitiy 
        BpBest: (float) The value fo the Best Binomial Probabilitiy.
        BpArr: (array) The array containing the computed Binomial Probabilities at each k.
    '''
    
    pSorted = np.sort(pValues)
    n = len(pSorted)
    if (kmax==0):
        kmax = n       # default:  search the whole array for best p-value
    i = np.arange(0,n)
    # NB the array index i = k-1.  We just need i in next line since we want sf(k-1) = sf(k)+pmf(k)
    BpArr = stats.binom.sf(i[:kmax],n,pSorted[:kmax])  # remember: test is w.r.t. n, even if we only search up to kmax
    BpBest = BpArr.min()
    kBest = BpArr.argmin()+1
    pThresh = pSorted[kBest-1]
    if returnArray:
        return pThresh, kBest, BpBest, BpArr
    else:
        return pThresh, kBest, BpBest

def ndarray_to_Chi2TSD(trials):
    return cy.dists.Chi2TSD(cy.utils.Arrays(trials))

#load the pre-calculated background pre-trial p-values by trials and by source
pval_trials = np.load('/data/user/asharma/csky/test/top1k_trials/pval_trials_13k_bkgtrials.npy', allow_pickle=True)
pval_src = np.load('/data/user/asharma/csky/test/top1k_trials/pval_src_13k_bkgtrials.npy', allow_pickle=True)


#Binomial tests on all 1000 sources over all trials
pThr_arr, kThr_arr, Bbest_arr, Bparr_arr = [], [], [], []

for i in range (len(pval_trials)):
    #rand_pval_test = np.random.uniform(1e-4, 1e+0, 1800)
    pThr, kThr, Bbest, Bparr = BinomialTest(pval_trials[i], 1000)
    pThr_arr.append(pThr)
    kThr_arr.append(kThr)
    Bbest_arr.append(Bbest)
    Bparr_arr.append(Bparr)


# shorten catalog to relevant columns and make one catalog for each trial (more like a sourcelist with p-values and sky positions)
src_trials_data = [] 
for t in range(len(pval_trials)):
    src_data = {'name': top1k_srclist.name, 'ra': top1k_srclist.ra_deg, 'dec': top1k_srclist.dec_deg, 'pval': pval_trials[t]}
    src_trials_data.append(pd.DataFrame(src_data))

# sort by p-value to be abl
for t in range(len(src_trials_data)):
    src_trials_data[t].sort_values(by='pval', inplace=True)


# calculate the correlations: how many sources within 1 degree of each other for each top-k value over all the trials
corr_cnt = []
with time('finding correlations within top-k sources'):
    for i in range (len(src_trials_data)):
        topk = kThr_arr[i]

        topk_cat = src_trials_data[i].head(topk)
        #find sources very close together in romabzcat
        overlap = 0

        for index, row in topk_cat.iterrows():
            for index1, row1 in topk_cat.iterrows():
                if (index != index1):
                    c111 = (np.abs(row['ra'] - row1['ra']) < 1) 
                    c222 = (np.abs(row['dec'] - row1['dec']) < 1)
                    if (c111 & c222):
                        overlap = overlap + 1

        corr_cnt.append(overlap)

    corr_cnt = np.array(corr_cnt)
print(corr_cnt.shape)
np.save("/home/asharma/csky/correlations_count_topk.npy", corr_cnt)
