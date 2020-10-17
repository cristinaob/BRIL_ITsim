import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2026Geometry.py
# If you notice a mistake, please update the generating script, not just this config

#Tracker stuff
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = cms.bool(False)

## Here we put the xml stuff for the tracker-only geometry
#
# Need to remove the rest in order to avoid SD-related crashes in Geant4

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2026/v3/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsTracker.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v2/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/CMSCommonData/data/muonBase/2026/v3/muonBase.xml',
        'Geometry/CMSCommonData/data/cmsMuon.xml',
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/beampipe/2026/v1/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam/2026/v1/cmsBeam.xml',
        'Geometry/CMSCommonData/data/muonMB.xml',
        'Geometry/CMSCommonData/data/muonMagnet.xml',
        'Geometry/CMSCommonData/data/cavern/2021/v1/cavern.xml',
        'Geometry/CMSCommonData/data/cavernFloor/2017/v1/cavernFloor.xml',
        'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
        'Geometry/TrackerCommonData/data/PhaseII/OuterTracker616_2020_04/pixfwd.xml',
        'Geometry/TrackerCommonData/data/PhaseII/OuterTracker616_2020_04/pixbar.xml',
        'Geometry/TrackerCommonData/data/trackermaterial.xml',
        'Geometry/TrackerCommonData/data/PhaseII/OuterTracker616_2020_04/otst.xml',
        'Geometry/TrackerCommonData/data/PhaseII/OuterTracker800_2020_07/tracker.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker615/pixel.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerbar.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerfwd.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerStructureTopology.xml',
        'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613/pixelStructureTopology.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/trackersens.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/pixelsens.xml',
        'Geometry/TrackerRecoData/data/PhaseII/OuterTracker616_2020_04/trackerRecoMaterial.xml',
        'SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups_ForPhaseII.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/trackerProdCuts.xml',
        'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/pixelProdCuts.xml',
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml',
        # 'Geometry/TrackerCommonData/data/pixfwdCommon.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613_MB_2019_04/pixfwd.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613_MB_2019_04/pixbar.xml',
        # 'Geometry/TrackerCommonData/data/trackermaterial.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/otst.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613_MB_2019_04/tracker.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613_MB_2019_04/pixel.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerbar.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerfwd.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker404/trackerStructureTopology.xml',
        # 'Geometry/TrackerCommonData/data/PhaseII/TiltedTracker613/pixelStructureTopology.xml',
        # 'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/trackersens.xml',
        # 'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/pixelsens.xml',
        # 'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker613_MB_2019_04/trackerRecoMaterial.xml',
        # 'SimTracker/TrackerMaterialAnalysis/data/trackingMaterialGroups_ForPhaseII.xml',
        # 'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/trackerProdCuts.xml',
        # 'Geometry/TrackerSimData/data/PhaseII/TiltedTracker404/pixelProdCuts.xml',
        # 'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml',
    )+
    cms.vstring(
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)
