from root_pandas import read_root
import numpy as np
from ROOT import TFile,TTree
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import pandas as pd

# =====================
# Variable, input data path
# TO DO: Make them command line arguments!
varSim    = "Cluster"
Path      = '/afs/cern.ch/work/c/cbarrera/private/BRIL/clusterData_Joseph/'
OutPath   = 'clusterPlots/'


# =====================
# Geometrical Filters:
# These will allow us to identify the disk/ring in which the cluster was found
rMaskRange = { 
    1:[6.5,9.7], 
    2:[10.9,13.7], 
    3:[14.5,17.3], 
    4:[18.4,20.9], 
    5:[21.9,25.3] 
}

zMaskRange = { 
    1:[174,176], 
    2:[200,202], 
    3:[230,232], 
    4:[264,266] 
}

zMaskRangeInner = {
    1:{
        1:[174  ,174.5],
        2:[174.5,175  ],
        3:[175  ,175.5],
        4:[175.5,176 ]
    },
    2:{
        1:[200  ,200.5],
        2:[200.5,201  ],
        3:[201  ,201.5],
        4:[202.5,203  ]
    },
    3:{
        1:[230  ,230.5],
        2:[230.5,230.7],
        3:[231.7,231.2],
        4:[231.2,231.6]
    },
    4:{
        1:[264  ,264.5],
        2:[264.5,265  ],
        3:[265  ,265.5],
        4:[265.5,266   ]
    }
}
    
NRing = {
    1:10,
    2:14,
    3:18,
    4:22,
    5:24
}


# =====================
# Cluster Studies
PU = ['100']

data = {}
xP, yP, zP  = {},{},{}
phiP, thetaP, mergeP, rP = {},{},{},{}

Disk      = {}
InnerDisk = {}
Ring      = {}

for pu in PU:

    # Loading data
    fileName = Path+varSim+'_'+pu+'.0_00.root'
    treeName = varSim.lower()+'_tree'

    data[pu] = read_root(fileName,treeName)

    xP[pu]      = np.array(data[pu].CluX)
    yP[pu]      = np.array(data[pu].CluY)
    zP[pu]      = np.array(data[pu].CluZ)
    phiP[pu]    = np.array(data[pu].CluPhi)
    thetaP[pu]  = np.array(data[pu].CluTheta)
    mergeP[pu]  = np.array(data[pu].CluMerge)
    rP[pu]      = np.sqrt(xP[pu]**2 + yP[pu]**2)

    # Identifying clusters per disk and ring
    Disk[pu]      = {}
    InnerDisk[pu] = {}
    Ring[pu]      = {}
    for zi in zMaskRange:
        PosD = np.logical_and( zP[pu] >  zMaskRange[zi][0] , zP[pu] <  zMaskRange[zi][1])
        NegD = np.logical_and( zP[pu] < -zMaskRange[zi][0] , zP[pu] > -zMaskRange[zi][1])
        Disk[pu][zi]      = np.logical_or(PosD,NegD) 
        InnerDisk[pu][zi] = {}
        for inzi in zMaskRangeInner[zi]:
            InnerPosD = np.logical_and( zP[pu] >  zMaskRangeInner[zi][inzi][0] , zP[pu] <  zMaskRangeInner[zi][inzi][1])
            InnerNegD = np.logical_and( zP[pu] < -zMaskRangeInner[zi][inzi][0] , zP[pu] > -zMaskRangeInner[zi][inzi][1])
            InnerNegD = InnerPosD 
            InnerDisk[pu][zi][inzi] = np.logical_or(InnerPosD, InnerNegD)

    for ri in rMaskRange:
        Ring[pu][ri] = np.logical_and( rP[pu] > rMaskRange[ri][0] , rP[pu] < rMaskRange[ri][1])

    # Drawing cluster distributions
    # --Plot 1: Number of clusters as a function of r for each ring 
    mask = []
    n = 0
    fig = plt.figure(figsize=(10,10)) 
    for di in Disk[pu]:
        for ri in Ring[pu]:
            mask = np.logical_and(Disk[pu][di],Ring[pu][ri])
            n+=1
            plt.subplot(4,5,n)
            hist = plt.hist(rP[pu][mask], bins = 40, histtype = 'step')
            plt.title('Disk '+str(di)+' Ring '+str(ri))
            plt.xlabel('r [cm]')
            plt.ylabel('# of Clusters')
            plt.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(OutPath+'radialDist-perRing_PU'+str(pu)+'.png')

    # --Plot 2: Number of clusters as a function of r for each disk
    mask = []
    n = 0
    fig = plt.figure(figsize=(10,10))
    for di in Disk[pu]:
        mask = Disk[pu][di]
        masking = Ring[pu][1]
        for ri in range(2,len(Ring[pu])):
            masking = np.logical_or(masking,Ring[pu][ri])
        mask = np.logical_and(mask,masking)
        n+=1
        plt.subplot(2,2,n)
        hist = plt.hist(rP[pu][mask], bins = 40, histtype = 'step')
        plt.title('Disk '+str(di))
        plt.xlabel('r [cm]')
        plt.ylabel('# of Clusters')
        plt.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(OutPath+'radialDist-perDisk_PU'+str(pu)+'.png')   

    # --Plot 3: z vs r distribution
    fig = plt.figure(figsize=(20,10))
    plt.hist2d(zP[pu],rP[pu],bins = 300,range = [[-300,300],[0,30]])
    fig.savefig(OutPath+'ZvsR_PU'+str(pu)+'.png')

    # --Plot 4: z vs r distribution (+ side)
    fig = plt.figure(figsize=(20,10))
    plt.hist2d(zP[pu],rP[pu],bins = 300,range = [[160,275],[0,30]])
    fig.savefig(OutPath+'ZvsR_PU'+str(pu)+'_Pos.png')

    # --Plot 5: z vs r distribution (disk 4)
    d = 4
    mask = Disk[pu][d]
    fig = plt.figure(figsize=(10,10))
    plt.hist2d(zP[pu][mask],rP[pu][mask],bins = 100,range = [zMaskRange[d],[0,30]])
    fig.savefig(OutPath+'ZvsR_PU'+str(pu)+'_D'+str(d)+'.png')
