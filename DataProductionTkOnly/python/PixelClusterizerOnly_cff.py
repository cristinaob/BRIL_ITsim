from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import *
from RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
 
def PixelClusterizerOnly(process):
    print("!!! Special version of the local reco: running pixel clusterizer only !!!")
    if hasattr(process,'PixelClusterizer_step'):
        process=customise_PixelClusterizer(process)

    return process

def customise_PixelClusterizer(process):
    process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')
    # print(process.PixelClusterizer_step)
    process.PixelClusterizer_step.remove(siPixelRecHitsPreSplitting)
    # print(process.PixelClusterizer_step)

    # keep new digis
    alist=['FEVTDEBUG','FEVTDEBUGHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep Phase2TrackerDigiedmDetSetVector_*_*_*')
    return process

