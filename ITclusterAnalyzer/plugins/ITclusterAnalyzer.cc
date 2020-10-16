// -*- C++ -*-
//
// Package:    BRIL_ITsim/ITclusterAnalyzer
// Class:      ITclusterAnalyzer
//
/**\class ITclusterAnalyzer ITclusterAnalyzer.cc BRIL_ITsim/ITclusterAnalyzer/plugins/ITclusterAnalyzer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Georg Auzinger
//         Created:  Wed, 05 Dec 2018 15:10:16 GMT
//
//

// system include files
#include <algorithm>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
//#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH2F.h>
#include <TTree.h>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

// a struct to hold the residuals for each matched cluster
struct Residual {

    double dx;
    double dy;
    double dr;

    Residual(double x, double y) : dx(x), dy(y) {
        dr = sqrt(pow(dx, 2) + pow(dy, 2));
    }

    void print() {
        std::cout << "(dx: " << dx << " dy: " << dy << " dr: " << dr << ") ";
    }

};

class ITclusterAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
    explicit ITclusterAnalyzer(const edm::ParameterSet&);
    ~ITclusterAnalyzer();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    //bool findCoincidence(DetId, Global3DPoint, bool);
    bool findCoincidence2x(DetId, Global3DPoint, bool, unsigned int&, edmNew::DetSet<SiPixelCluster>::const_iterator&);
    bool findCoincidence3x(DetId, Global3DPoint, bool, unsigned int&, edmNew::DetSet<SiPixelCluster>::const_iterator&);
    bool findCoincidenceInR2x(DetId, Global3DPoint, bool, std::vector<unsigned int>&, std::vector<edmNew::DetSet<SiPixelCluster>::const_iterator>&, std::vector<float>&);
    edm::DetSetVector<PixelDigiSimLink>::const_iterator findSimLinkDetSet(unsigned int thedetid);
    std::set<unsigned int> getSimTrackId(edm::DetSetVector<PixelDigiSimLink>::const_iterator, edmNew::DetSet<SiPixelCluster>::const_iterator, bool print);
    bool areSameSimTrackId(std::set<unsigned int> first, std::set<unsigned int> second, std::set<unsigned int>&);
    uint32_t getModuleID(bool, unsigned int, unsigned int, unsigned int);
    // ----------member data ---------------------------
    edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> m_tokenClusters;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> m_tokenSimLinks;
    edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> m_tokenDigis;

    // the pointers to geometry, topology and clusters
    // these are members so all functions can access them without passing as argument
    const TrackerTopology* tTopo = NULL;
    const TrackerGeometry* tkGeom = NULL;
    const edmNew::DetSetVector<SiPixelCluster>* clusters = NULL;
    const edm::DetSetVector<PixelDigiSimLink>* simlinks = NULL;
    const edm::DetSetVector<PixelDigi>* digis = NULL;  //defining pointer to digis - COB 26.02.19

    //max bins of Counting histogram
    uint32_t m_maxBin;
    //flag for checking coincidences
    bool m_docoincidence;

    //array of TH2F for clusters per disk per ring
    TH2F* m_diskHistosCluster[8];
    TH2F* m_diskHistosCluster_TFPX[16];
    //array of TH2F for hits per disk per ring
    TH2F* m_diskHistosHits[8];
    TH2F* m_diskHistosHits_TFPX[16];

    //tracker maps for clusters
    TH2F* m_trackerLayoutClustersZR;
    TH2F* m_trackerLayoutClustersYX;
    TH2F* m_trackerLayoutClustersZR_TFPX;
    TH2F* m_trackerLayoutClustersYX_TFPX;
    //tracker maps for hits
    TH2F* m_trackerLayoutHitsZR;
    TH2F* m_trackerLayoutHitsYX;
    TH2F* m_trackerLayoutHitsZR_TFPX;
    TH2F* m_trackerLayoutHitsYX_TFPX;

    //array of TH2F for 2xcoinc per disk per ring
    //first all coincidences
    TH2F* m_diskHistos2x[8];
    TH2F* m_diskHistos2x_TFPX[16];
    TH2F* m_diskHistos2xInR[8];
    TH2F* m_diskHistos2xInR_TFPX[16];
    //and the real ones
    TH2F* m_diskHistos2xreal[8];
    TH2F* m_diskHistos2xrealInR[8];
    TH2F* m_diskHistos2xreal_TFPX[16];
    TH2F* m_diskHistos2xrealInR_TFPX[16];
    //tracker maps for 2xcoinc
    TH2F* m_trackerLayout2xZR;
    TH2F* m_trackerLayout2xYX;
    TH2F* m_trackerLayout2xZR_TFPX;
    TH2F* m_trackerLayout2xYX_TFPX;
    TH2F* m_trackerLayout2xZR_InR;
    TH2F* m_trackerLayout2xYX_InR;
    TH2F* m_trackerLayout2xZR_InR_TFPX;
    TH2F* m_trackerLayout2xYX_InR_TFPX;

    //array of TH2F for 3xcoinc per disk per ring
    //first all coincidences
    TH2F* m_diskHistos3x[8];
    TH2F* m_diskHistos3x_TFPX[16];
    //and the real ones
    TH2F* m_diskHistos3xreal[8];
    TH2F* m_diskHistos3xreal_TFPX[16];
    //tracker maps for 3xcoinc
    TH2F* m_trackerLayout3xZR;
    TH2F* m_trackerLayout3xYX;
    TH2F* m_trackerLayout3xZR_TFPX;
    TH2F* m_trackerLayout3xYX_TFPX;

    //simple residual histograms for the cuts
    TH1F* m_residualX;
    TH1F* m_residualY;
    TH1F* m_residualR;
    TH1F* m_residualX_TFPX;
    TH1F* m_residualY_TFPX;
    TH1F* m_residualR_TFPX;
    TH1F* m_residualX_InR;
    TH1F* m_residualY_InR;
    TH1F* m_residualR_InR;
    TH1F* m_residualX_TFPX_InR;
    TH1F* m_residualY_TFPX_InR;
    TH1F* m_residualR_TFPX_InR;

    //the number of clusters per module
    TH1F* m_nClusters;
    TH1F* m_nHits;

    TH1F* m_nHits_TFPX;
    TH1F* m_nClusters_TFPX;

    //cuts for the coincidence
    double m_dx;
    double m_dy;
    double m_dz;

    //event counter
    uint32_t m_nevents;
    //coincidence counter
    uint32_t m_total2xcoincidences;
    uint32_t m_total2xcoincidences_TFPX;
    uint32_t m_total2xcoincidencesInR;
    uint32_t m_total2xcoincidencesInR_TFPX;
    uint32_t m_fake2xcoincidences;
    uint32_t m_fake2xcoincidences_TFPX;
    uint32_t m_fake2xcoincidencesInR;
    uint32_t m_fake2xcoincidencesInR_TFPX;
    uint32_t m_total3xcoincidences;
    uint32_t m_total3xcoincidences_TFPX;
    uint32_t m_fake3xcoincidences;
    uint32_t m_fake3xcoincidences_TFPX;

    // --
    // Variables for cluster parameterization studies
    bool m_storeClusterTree;
    TFile *outFileCluster;
    TTree *outTreeCluster;
    double CluX;
    double CluY;
    double CluZ;
    double CluArea;
    double CluSize;
    double CluTheta;
    double CluPhi;
    double CluCharge;
    unsigned int CluNum;
    unsigned int CluMerge;
    unsigned int mergeClu;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ITclusterAnalyzer::ITclusterAnalyzer(const edm::ParameterSet& iConfig)
        : //m_tokenClusters(consumes<edmNew::DetSetVector<SiPixelCluster>> ("clusters"))
        m_tokenClusters(consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("clusters")))
        , m_tokenSimLinks(consumes<edm::DetSetVector<PixelDigiSimLink>>(iConfig.getParameter<edm::InputTag>("simlinks")))
        , m_tokenDigis(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("digis"))) //adding digis variable - COB 26.02.19
        , m_maxBin(iConfig.getUntrackedParameter<uint32_t>("maxBin"))
        , m_docoincidence(iConfig.getUntrackedParameter<bool>("docoincidence"))
        , m_dx(iConfig.getParameter<double>("dx_cut"))
        , m_dy(iConfig.getParameter<double>("dy_cut"))
        , m_dz(iConfig.getParameter<double>("dz_cut"))
        , m_storeClusterTree(iConfig.getUntrackedParameter<bool>("storeClusterTree")) {
    //now do what ever initialization is needed
    m_nevents = 0;
    m_total2xcoincidences = 0;
    m_total2xcoincidences_TFPX = 0;
    m_total2xcoincidencesInR = 0;
    m_total2xcoincidencesInR_TFPX = 0;
    m_fake2xcoincidences = 0;
    m_fake2xcoincidences_TFPX = 0;
    m_fake2xcoincidencesInR = 0;
    m_fake2xcoincidencesInR_TFPX = 0;
    m_total3xcoincidences = 0;
    m_total3xcoincidences_TFPX = 0;
    m_fake3xcoincidences = 0;
    m_fake3xcoincidences_TFPX = 0;
}

ITclusterAnalyzer::~ITclusterAnalyzer() {
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void ITclusterAnalyzer::beginJob() {

    edm::Service<TFileService> fs;

    fs->file().cd("/");
    TFileDirectory td = fs->mkdir("TEPX");

    td = fs->mkdir("TEPX/Residuals");

    m_residualX = td.make<TH1F>("ResidualsX", "ResidualsX;deltaX;counts", 1000, 0, 1);
    m_residualY = td.make<TH1F>("ResidualsY", "ResidualsY;deltaY;counts", 1000, 0, 1);
    m_residualR = td.make<TH1F>("ResidualsR", "ResidualsR;deltaR;counts", 1000, 0, 1);

    m_residualX_InR = td.make<TH1F>("ResidualsXInR", "ResidualsX;deltaX;counts", 1000, 0, 1);
    m_residualY_InR = td.make<TH1F>("ResidualsYInR", "ResidualsY;deltaY;counts", 1000, 0, 1);
    m_residualR_InR = td.make<TH1F>("ResidualsRInR", "ResidualsR;deltaR;counts", 1000, 0, 1);

    fs->file().cd("/");
    td = fs->mkdir("TEPX/perModule");
    m_nClusters = td.make<TH1F>("Number of Clusters per module per event", "# of Clusters;# of Clusters; Occurence", 500, 0, 500);
    m_nHits = td.make<TH1F>("Number of Hits per module per event", "# of Hits; # of Hits; Occurrences", 500, 0, 500);

    fs->file().cd("/");
    td = fs->mkdir("TEPX/Clusters");

    //now lets create the histograms
    for (unsigned int i = 0; i < 8; i++) {
        int disk = (i < 4) ? i - 4 : i - 3;
        std::stringstream histoname;
        histoname << "Number of clusters for Disk " << disk << ";Ring;# of Clusters per event";
        std::stringstream histotitle;
        histotitle << "Number of clusters for Disk " << disk;
        //name, name, nbinX, Xlow, Xhigh, nbinY, Ylow, Yhigh
        m_diskHistosCluster[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
    }
    m_trackerLayoutClustersZR = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
    m_trackerLayoutClustersYX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

    fs->file().cd("/");
    td = fs->mkdir("TEPX/Hits");

    //histograms
    for (unsigned int i = 0; i < 8; i++) {
        int disk = (i < 4) ? i - 4 : i - 3;
        std::stringstream histoname;
        histoname << "Number of hits for Disk " << disk << ";Ring;# of Hits per event";
        std::stringstream histotitle;
        histotitle << "Number of hits for Disk " << disk;
        m_diskHistosHits[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
    }

    m_trackerLayoutHitsZR = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
    m_trackerLayoutHitsYX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

    if (m_docoincidence) {
        fs->file().cd("/");
        td = fs->mkdir("TEPX/2xCoincidences");
        //now lets create the histograms
        for (unsigned int i = 0; i < 8; i++) {
            int disk = (i < 4) ? i - 4 : i - 3;
            std::stringstream histoname;
            histoname << "Number of 2x Coincidences for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitle;
            histotitle << "Number of 2x Coincidences for Disk " << disk;
            //name, name, nbinX, Xlow, Xhigh, nbinY, Ylow, Yhigh
            m_diskHistos2x[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamereal;
            histonamereal << "Number of real 2x Coincidences for Disk " << disk << ";Ring;# of real coincidences per event";
            std::stringstream histotitlereal;
            histotitlereal << "Number of real 2x Coincidences for Disk " << disk;
            //name, name, nbinX, Xlow, Xhigh, nbinY, Ylow, Yhigh
            m_diskHistos2xreal[i] = td.make<TH2F>(histotitlereal.str().c_str(), histonamereal.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonameInR;
            histonameInR << "Number of 2x Coincidences in R for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitleInR;
            histotitleInR << "Number of 2x Coincidences in R for Disk " << disk;
            m_diskHistos2xInR[i] = td.make<TH2F>(histotitleInR.str().c_str(), histonameInR.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamerealInR;
            histonamerealInR << "Number of real 2x Coincidences in R for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitlerealInR;
            histotitlerealInR << "Number of real 2x Coincidences in R for Disk " << disk;
            m_diskHistos2xrealInR[i] = td.make<TH2F>(histotitlerealInR.str().c_str(), histonamerealInR.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
        }
        m_trackerLayout2xZR = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout2xYX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);
        m_trackerLayout2xZR_InR = td.make<TH2F>("RVsZ_InR", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout2xYX_InR = td.make<TH2F>("XVsY_InR", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);
    }

    if (m_docoincidence) {
        fs->file().cd("/");
        td = fs->mkdir("TEPX/3xCoincidences");
        //now lets create the histograms
        for (unsigned int i = 0; i < 8; i++) {
            int disk = (i < 4) ? i - 4 : i - 3;
            std::stringstream histoname;
            histoname << "Number of 3x Coincidences for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitle;
            histotitle << "Number of 3x Coincidences for Disk " << disk;
            //name, name, nbinX, Xlow, Xhigh, nbinY, Ylow, Yhigh
            m_diskHistos3x[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamereal;
            histonamereal << "Number of real 3x Coincidences for Disk " << disk << ";Ring;# of real coincidences per event";
            std::stringstream histotitlereal;
            histotitlereal << "Number of real 3x Coincidences for Disk " << disk;
            //name, name, nbinX, Xlow, Xhigh, nbinY, Ylow, Yhigh
            m_diskHistos3xreal[i] = td.make<TH2F>(histotitlereal.str().c_str(), histonamereal.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
        }
        m_trackerLayout3xZR = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout3xYX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);
    }

    // ---------------------------------------------------

    fs->file().cd("/");
    td = fs->mkdir("TFPX");

    fs->file().cd("/");
    td = fs->mkdir("TFPX/Residuals");

    m_residualX_TFPX = td.make<TH1F>("ResidualsX", "ResidualsX;deltaX;counts", 1000, 0, 1);
    m_residualY_TFPX = td.make<TH1F>("ResidualsY", "ResidualsY;deltaY;counts", 1000, 0, 1);
    m_residualR_TFPX = td.make<TH1F>("ResidualsR", "ResidualsR;deltaR;counts", 1000, 0, 1);

    m_residualX_TFPX_InR = td.make<TH1F>("ResidualsXInR", "ResidualsX;deltaX;counts", 1000, 0, 1);
    m_residualY_TFPX_InR = td.make<TH1F>("ResidualsYInR", "ResidualsY;deltaY;counts", 1000, 0, 1);
    m_residualR_TFPX_InR = td.make<TH1F>("ResidualsRInR", "ResidualsR;deltaR;counts", 1000, 0, 1);

    fs->file().cd("/");
    td = fs->mkdir("TFPX/perModule");
    m_nHits_TFPX = td.make<TH1F>("Number of Hits per module per event", "# of Hits; # of Hits; Occurrences", 500, 0, 500);
    m_nClusters_TFPX = td.make<TH1F>("Number of Clusters per module per event", "# of Clusters;# of Clusters; Occurence", 500, 0, 500);

    fs->file().cd("/");
    td = fs->mkdir("TFPX/Hits");

    for (unsigned int i = 0; i < 16; i++) {
        int disk = (i < 8) ? i - 8 : i - 7;
        std::stringstream histoname;
        histoname << "Number of hits for Disk " << disk << ";Ring;# of Hits per event";
        std::stringstream histotitle;
        histotitle << "Number of hits for Disk " << disk;
        m_diskHistosHits_TFPX[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
    }

    m_trackerLayoutHitsZR_TFPX = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
    m_trackerLayoutHitsYX_TFPX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

    fs->file().cd("/");
    td = fs->mkdir("TFPX/Clusters");

    for (unsigned int i = 0; i < 16; i++) {
        int disk = (i < 8) ? i - 8 : i - 7;
        std::stringstream histoname;
        histoname << "Number of clusters for Disk " << disk << ";Ring;# of Clusters per event";
        std::stringstream histotitle;
        histotitle << "Number of clusters for Disk " << disk;
        m_diskHistosCluster_TFPX[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
    }

    m_trackerLayoutClustersZR_TFPX = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
    m_trackerLayoutClustersYX_TFPX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

    if (m_docoincidence) {

        fs->file().cd("/");
        td = fs->mkdir("TFPX/2xCoincidences");

        for (unsigned int i = 0; i < 16; i++) {
            int disk = (i < 8) ? i - 8 : i - 7;
            std::stringstream histoname;
            histoname << "Number of 2x Coincidences for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitle;
            histotitle << "Number of 2x Coincidences for Disk " << disk;
            m_diskHistos2x_TFPX[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamereal;
            histonamereal << "Number of real 2x Coincidences for Disk " << disk << ";Ring;# of real coincidences per event";
            std::stringstream histotitlereal;
            histotitlereal << "Number of real 2x Coincidences for Disk " << disk;
            m_diskHistos2xreal_TFPX[i] = td.make<TH2F>(histotitlereal.str().c_str(), histonamereal.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonameInR;
            histonameInR << "Number of 2x Coincidences in R for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitleInR;
            histotitleInR << "Number of 2x Coincidences in R for Disk " << disk;
            m_diskHistos2xInR_TFPX[i] = td.make<TH2F>(histotitleInR.str().c_str(), histonameInR.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamerealInR;
            histonamerealInR << "Number of real 2x Coincidences in R for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitlerealInR;
            histotitlerealInR << "Number of real 2x Coincidences in R for Disk " << disk;
            m_diskHistos2xrealInR_TFPX[i] = td.make<TH2F>(histotitlerealInR.str().c_str(), histonamerealInR.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
        }
        m_trackerLayout2xZR_TFPX = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout2xYX_TFPX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);           

        m_trackerLayout2xZR_InR_TFPX = td.make<TH2F>("RVsZ_InR", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout2xYX_InR_TFPX = td.make<TH2F>("XVsY_InR", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

        fs->file().cd("/");
        td = fs->mkdir("TFPX/3xCoincidences");

        for (unsigned int i = 0; i < 16; i++) {
            int disk = (i < 8) ? i - 8 : i - 7;
            std::stringstream histoname;
            histoname << "Number of 3x Coincidences for Disk " << disk << ";Ring;# of coincidences per event";
            std::stringstream histotitle;
            histotitle << "Number of 3x Coincidences for Disk " << disk;
            m_diskHistos3x_TFPX[i] = td.make<TH2F>(histotitle.str().c_str(), histoname.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);

            std::stringstream histonamereal;
            histonamereal << "Number of real 3x Coincidences for Disk " << disk << ";Ring;# of real coincidences per event";
            std::stringstream histotitlereal;
            histotitlereal << "Number of real 3x Coincidences for Disk " << disk;
            m_diskHistos3xreal_TFPX[i] = td.make<TH2F>(histotitlereal.str().c_str(), histonamereal.str().c_str(), 5, .5, 5.5, m_maxBin, 0, m_maxBin);
        }

        m_trackerLayout3xZR_TFPX = td.make<TH2F>("RVsZ", "R vs. z position", 6000, -300.0, 300.0, 600, 0.0, 30.0);
        m_trackerLayout3xYX_TFPX = td.make<TH2F>("XVsY", "x vs. y position", 1000, -50.0, 50.0, 1000, -50.0, 50.0);

    }

    // ---------------------------------------------------

    if (m_storeClusterTree) {

        outFileCluster = new TFile("Cluster.root","RECREATE");
        outFileCluster->cd();
        outTreeCluster = new TTree("cluster_tree","cluster");

        outTreeCluster->Branch("CluX", &CluX);
        outTreeCluster->Branch("CluY", &CluY);
        outTreeCluster->Branch("CluZ", &CluZ);
        outTreeCluster->Branch("CluTheta", &CluTheta);
        outTreeCluster->Branch("CluPhi", &CluPhi);
        outTreeCluster->Branch("CluCharge", &CluCharge);
        outTreeCluster->Branch("CluArea", &CluArea);
        outTreeCluster->Branch("CluSize", &CluSize);
        outTreeCluster->Branch("CluMerge", &CluMerge);
        outTreeCluster->Branch("CluNum", &CluNum);

    }

}

// ------------ method called for each event  ------------
void ITclusterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

    //get the digis - COB 26.02.19
    edm::Handle<edm::DetSetVector<PixelDigi>> tdigis;
    iEvent.getByToken(m_tokenDigis, tdigis);

    //get the clusters
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> tclusters;
    iEvent.getByToken(m_tokenClusters, tclusters);

    //get the simlinks
    edm::Handle<edm::DetSetVector<PixelDigiSimLink>> tsimlinks;
    iEvent.getByToken(m_tokenSimLinks, tsimlinks);

    // Get the geometry
    edm::ESHandle<TrackerGeometry> tgeomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get("idealForDigi", tgeomHandle);

    // Get the topology
    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);

    //get the pointers to geometry, topology and clusters
    tTopo = tTopoHandle.product();
    //const TrackerGeometry* tkGeom = &(*geomHandle);
    tkGeom = tgeomHandle.product();
    clusters = tclusters.product();
    simlinks = tsimlinks.product();
    digis = tdigis.product();  //pointer to digis - COB 26.02.19

    //a 2D counter array to count the number of clusters per disk and per ring
    unsigned int cluCounter[8][5];
    memset(cluCounter, 0, sizeof(cluCounter));
    unsigned int cluCounter_TFPX[16][4];
    memset(cluCounter_TFPX, 0, sizeof(cluCounter_TFPX));
    //counter array for hits per disk and per ring
    unsigned int hitCounter[8][5];
    memset(hitCounter, 0, sizeof(hitCounter));
    unsigned int hitCounter_TFPX[16][4];
    memset(hitCounter_TFPX, 0, sizeof(hitCounter_TFPX));
    //counter for 2x coincidences
    unsigned int x2Counter[8][5];
    memset(x2Counter, 0, sizeof(x2Counter));
    unsigned int x2Counter_TFPX[16][4];
    memset(x2Counter_TFPX, 0, sizeof(x2Counter_TFPX));
    unsigned int x2CounterInR[8][5];
    memset(x2CounterInR, 0, sizeof(x2CounterInR));
    unsigned int x2CounterInR_TFPX[16][4];
    memset(x2CounterInR_TFPX, 0, sizeof(x2CounterInR_TFPX));
    unsigned int x2Counterreal[8][5];
    memset(x2Counterreal, 0, sizeof(x2Counterreal));
    unsigned int x2Counterreal_TFPX[16][4];
    memset(x2Counterreal_TFPX, 0, sizeof(x2Counterreal_TFPX));
    unsigned int x2CounterrealInR[8][5];
    memset(x2CounterrealInR, 0, sizeof(x2CounterrealInR));
    unsigned int x2CounterrealInR_TFPX[16][4];
    memset(x2CounterrealInR_TFPX, 0, sizeof(x2CounterrealInR_TFPX));
    //counter for 3x coincidences
    unsigned int x3Counter[8][5];
    memset(x3Counter, 0, sizeof(x3Counter));
    unsigned int x3Counter_TFPX[16][4];
    memset(x3Counter_TFPX, 0, sizeof(x3Counter_TFPX));
    unsigned int x3Counterreal[8][5];
    memset(x3Counterreal, 0, sizeof(x3Counterreal));
    unsigned int x3Counterreal_TFPX[16][4];
    memset(x3Counterreal_TFPX, 0, sizeof(x3Counterreal_TFPX));

    //-------------------------------------------------------------
    //loop over digis - COB 26.02.19
    //debugging...
    //std::cout << "--- New event ---" << std::endl;
    for (typename edm::DetSetVector<PixelDigi>::const_iterator DSVit = digis->begin(); DSVit != digis->end(); DSVit++) {

        //get the detid
        unsigned int rawid(DSVit->detId());
        DetId detId(rawid);

        //debugging...
        //std::cout << "DetId " << std::hex << "0x" << rawid << std::dec << " " << detId.det() << " " << detId.subdetId() << " " 
        //          << ((rawid >> 23) & 0x3) << " " << ((rawid >> 18) & 0xF) << " " << ((rawid >> 12) & 0x3F) << " " 
        //          << ((rawid >> 2) & 0xFF) << std::endl;
        //std::cout << "side " << tTopo->pxfSide(detId) << " layer " << tTopo->pxfDisk(detId) << " ring " << tTopo->pxfBlade(detId) 
        //          << " module " << tTopo->pxfModule(detId) << std::endl;       

        //module type => need phase 2 pixel forward module, in endcap
        TrackerGeometry::ModuleType mType = tkGeom->getDetectorType(detId);
        if (mType != TrackerGeometry::ModuleType::Ph2PXF && detId.subdetId() != PixelSubdetector::PixelEndcap) 
            continue;

        //obtaining side, disk, ring of module
        unsigned int side = (tTopo->pxfSide(detId));   
        unsigned int disk = (tTopo->pxfDisk(detId));
        unsigned int ring = (tTopo->pxfBlade(detId));

        if (disk > 8) { // TEPX modules

            //index in histogram map
            int hist_id = -1;
            unsigned int ring_id = ring - 1;
            if (side == 1) {
                //this is a TEPX- hit on side1
                hist_id = disk - 9;
            } else if (side == 2) {
                //this is a TEPX+ hit on side 2
                hist_id = 4 + disk - 9;
            }

            //find the geometry of the module associated to the digi
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            if (!geomDetUnit)
                continue;

            //store the total number of digis in module
            m_nHits->Fill(DSVit->size());

            //loop over the digis in each module
            for (edm::DetSet<PixelDigi>::const_iterator digit = DSVit->begin(); digit != DSVit->end(); digit++) {

                hitCounter[hist_id][ring_id]++;

                //finding the position of the hit
                MeasurementPoint mpHit(digit->row(), digit->column());
                Local3DPoint localPosHit = geomDetUnit->topology().localPosition(mpHit);
                Global3DPoint globalPosHit = geomDetUnit->surface().toGlobal(localPosHit);

                //debugging...
                //std::cout << "---- Hit info:" << std::endl;
                //std::cout << "measurement point of hit => row " << digit->row() << " column " << digit->column() << std::endl;
                //std::cout << "local position of hit => x " << localPosHit.x() << " y " << localPosHit.y() 
                //          << " z " << localPosHit.z() << std::endl;
                //std::cout << "global position of hit => x " << globalPosHit.x() << " y " << globalPosHit.y() 
                //          << " z " << globalPosHit.z() << std::endl;

                //fill layout histograms
                m_trackerLayoutHitsZR->Fill(globalPosHit.z(), globalPosHit.perp());
                m_trackerLayoutHitsYX->Fill(globalPosHit.x(), globalPosHit.y());

            }

        } else { // TFPX modules

            //debugging...
            //std::cout << "side " << tTopo->pxfSide(detId) << " layer " << tTopo->pxfDisk(detId) << " ring " << tTopo->pxfBlade(detId) 
            //          << " module " << tTopo->pxfModule(detId) << std::endl;

            //index in histogram map
            int hist_id = -1;
            unsigned int ring_id = ring - 1;
            if (side == 1) {
                //this is a TFPX- hit on side 1
                hist_id = disk - 1;
            } else if (side == 2) {
                //this is a TFPX+ hit on side 2
                hist_id = 8 + disk - 1;
            } 

            //debugging...
            //std::cout << "side " << side << " hist_id " << hist_id << " ring_id " << ring_id << std::endl;

            //find the geometry of the module associated to the digi
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            if (!geomDetUnit)
                continue;

            //store the total number of digis in module
            m_nHits_TFPX->Fill(DSVit->size());

            //loop over the digis in each module
            for (edm::DetSet<PixelDigi>::const_iterator digit = DSVit->begin(); digit != DSVit->end(); digit++) {

                hitCounter_TFPX[hist_id][ring_id]++;

                //finding the position of the hit
                MeasurementPoint mpHit(digit->row(), digit->column());
                Local3DPoint localPosHit = geomDetUnit->topology().localPosition(mpHit);
                Global3DPoint globalPosHit = geomDetUnit->surface().toGlobal(localPosHit);

                //fill layout histograms
                m_trackerLayoutHitsZR_TFPX->Fill(globalPosHit.z(), globalPosHit.perp());
                m_trackerLayoutHitsYX_TFPX->Fill(globalPosHit.x(), globalPosHit.y());

            }

        }    

    }
    //-------------------------------------------------------------

    //loop the modules in the cluster collection
    for (typename edmNew::DetSetVector<SiPixelCluster>::const_iterator DSVit = clusters->begin(); DSVit != clusters->end(); DSVit++) {
        //get the detid
        unsigned int rawid(DSVit->detId());
        DetId detId(rawid);

        //figure out the module type using the detID
        TrackerGeometry::ModuleType mType = tkGeom->getDetectorType(detId);
        if (mType != TrackerGeometry::ModuleType::Ph2PXF && detId.subdetId() != PixelSubdetector::PixelEndcap)
            continue;
        //std::cout << "DetID " << std::hex << "0x" << rawid << std::dec << " " << detId.det() << " " 
        //          << detId.subdetId() << " " << ((rawid >> 23) & 0x3) << " " << ((rawid >> 18) & 0xF) << " " 
        //          << ((rawid >> 12) & 0x3F) << " " << ((rawid >> 2) & 0xFF) << std::endl;

        //find out which layer, side and ring
        unsigned int side = (tTopo->pxfSide(detId));  // values are 1 and 2 for -+Z
        unsigned int layer = (tTopo->pxfDisk(detId)); //values are 1 to 12 for disks TFPX1 to TFPX 8  and TEPX1 to TEPX 4
        unsigned int ring = (tTopo->pxfBlade(detId));

        if (layer > 8) { // TEPX modules

            //the index in my histogram map
            int hist_id = -1;
            unsigned int ring_id = ring - 1;

            if (side == 1) {
                //this is a TEPX- hit on side1
                hist_id = layer - 9;
            } else if (side == 2) {
                //this is a TEPX+ hit on side 2
                hist_id = 4 + layer - 9;
            }

            // Get the geomdet
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            if (!geomDetUnit)
                continue;

            unsigned int nClu = 0;

            //fill the number of clusters for this module
            m_nClusters->Fill(DSVit->size());

            //now loop the clusters for each detector
            for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = DSVit->begin(); cluit != DSVit->end(); cluit++) {
                //increment the counters
                nClu++;
                cluCounter[hist_id][ring_id]++;

                // determine the position
                MeasurementPoint mpClu(cluit->x(), cluit->y());
                Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
                Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

                //fill TkLayout histos
                m_trackerLayoutClustersZR->Fill(globalPosClu.z(), globalPosClu.perp());
                m_trackerLayoutClustersYX->Fill(globalPosClu.x(), globalPosClu.y());

                //for cluster parameterization studies...
                if (m_storeClusterTree) {

                    edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter = findSimLinkDetSet(rawid);
                    std::set<unsigned int> mergeClu = this->getSimTrackId(simLinkDSViter, cluit, false);

                    CluX = globalPosClu.x();
	            CluY = globalPosClu.y();
	            CluZ = globalPosClu.z();
	            CluPhi = globalPosClu.phi();
	            CluTheta = globalPosClu.theta();
	            CluCharge = cluit->charge();
	            CluArea = (cluit->sizeY())*(cluit->sizeX());
	            CluSize = cluit->size();
	            CluMerge = mergeClu.size();

                    outTreeCluster->Fill();
                 }

                //std::cout << globalPosClu.x() << " " << globalPosClu.y() << std::endl;
                if (m_docoincidence) {
                    unsigned int coincidenceId;
                    edmNew::DetSet<SiPixelCluster>::const_iterator coincidenceCluster;
                    bool found = this->findCoincidence2x(detId, globalPosClu, true, coincidenceId, coincidenceCluster);
                    if (found) {
                        m_total2xcoincidences++;
                        x2Counter[hist_id][ring_id]++;
                        //I have found a coincidence hit, so now I need to check the sim Track ID of each of the two clusters
                        //so first, let's get the simTrackID of the original cluster-for this I need to loop over all pixels
                        //and check if multiple SimTracks have caused it
                        //if that is not the case, I use the SimLink DSViter and the DSiter (cluit) to get a simTrackID
                        //I need to use the DetId of the coincidence cluster to get another SimLinkDSViter and the coincidence 
                        //cluster DSiter for the second simTrackID

                        //now get the simlink detset
                        edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter = findSimLinkDetSet(rawid);
                        std::set<unsigned int> simTrackId = this->getSimTrackId(simLinkDSViter, cluit, false);
                        //now get the simlink detset based on the coincidence hit detid
                        simLinkDSViter = findSimLinkDetSet(coincidenceId);
                        std::set<unsigned int> coincidencesimTrackId = this->getSimTrackId(simLinkDSViter, coincidenceCluster, false);
                        std::set<unsigned int> intersection;
                        bool areSame = areSameSimTrackId(simTrackId, coincidencesimTrackId, intersection);

                        if (areSame) {
                            x2Counterreal[hist_id][ring_id]++;
                        } else {
                            m_fake2xcoincidences++;
                        }

                        m_trackerLayout2xZR->Fill(globalPosClu.z(), globalPosClu.perp());
                        m_trackerLayout2xYX->Fill(globalPosClu.x(), globalPosClu.y());

                        //done with 2 fold coincidences, now 3 fold
                        //only if we have a 2 fold coincidence we can search for a third one in another ring
                        found = false;
                        unsigned int coincidenceId3x;
                        edmNew::DetSet<SiPixelCluster>::const_iterator coincidenceCluster3x;
                        found = this->findCoincidence3x(detId, globalPosClu, true, coincidenceId3x, coincidenceCluster3x);
                        if (found) {
                            m_total3xcoincidences++;
                            x3Counter[hist_id][ring_id]++;

                            //now get the simlink detset based on the coincidence hit detid
                            edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter3x = findSimLinkDetSet(coincidenceId3x);
                            std::set<unsigned int> coincidencesimTrackId3x = this->getSimTrackId(simLinkDSViter3x, coincidenceCluster3x, false);
                            std::set<unsigned int> intersection3x;
                            //std::cout << "third track ID: ";
                            //for (auto it : coincidencesimTrackId3x)
                            //std::cout << it;
                            //std::cout << std::endl;
                            areSame = areSameSimTrackId(intersection, coincidencesimTrackId3x, intersection3x);
                            //std::cout << "common track ID: ";
                            //for (auto it : intersection3x)
                            //std::cout << it;
                            //std::cout << std::endl;

                            if (areSame) {
                                x3Counterreal[hist_id][ring_id]++;
                            } else {
                                m_fake3xcoincidences++;
                            }

                            m_trackerLayout3xZR->Fill(globalPosClu.z(), globalPosClu.perp());
                            m_trackerLayout3xYX->Fill(globalPosClu.x(), globalPosClu.y());
                        }
                    }
                    //-----------------------------------------
                    std::vector<unsigned int> coincidenceIdInR;
                    std::vector<edmNew::DetSet<SiPixelCluster>::const_iterator> coincidenceClusterInR;
                    std::vector<float> coincidenceClusterdr;
                    bool found2xinR = this-> findCoincidenceInR2x(detId, globalPosClu, true, coincidenceIdInR, coincidenceClusterInR, coincidenceClusterdr);
                    if (found2xinR) {
                        m_total2xcoincidencesInR++;
                        x2CounterInR[hist_id][ring_id]++;

                        //debugging...
                        //std::cout << "I have " << coincidenceClusterInR.size() << " overlapping clusters" << std::endl;
                        //for (unsigned int i=0;i<coincidenceClusterInR.size();i++) {
                        //    std::cout << "dr of cluster " << coincidenceClusterInR.at(i) << "in module " << coincidenceIdInR.at(i) 
                        //              << " is: " << coincidenceClusterdr.at(i) << std::endl;
                        //}
                        //std::cout << "min dr is: " << *std::min_element(coincidenceClusterdr.begin(), coincidenceClusterdr.end()) << " and is located in entry " 
                        //          << std::min_element(coincidenceClusterdr.begin(), coincidenceClusterdr.end()) - coincidenceClusterdr.begin() << std::endl; 

                        int mindrIndex = std::min_element(coincidenceClusterdr.begin(), coincidenceClusterdr.end()) - coincidenceClusterdr.begin();

                        edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter = findSimLinkDetSet(rawid);
                        std::set<unsigned int> simTrackId = this->getSimTrackId(simLinkDSViter,cluit,false);
                        simLinkDSViter = findSimLinkDetSet(coincidenceIdInR.at(mindrIndex));
                        std::set<unsigned int> coincidencesimTrackId = this->getSimTrackId(simLinkDSViter, coincidenceClusterInR.at(mindrIndex), false);
                        std::set<unsigned int> intersection;
                        bool areSame = areSameSimTrackId(simTrackId, coincidencesimTrackId, intersection);

                        if (areSame) {
                            x2CounterrealInR[hist_id][ring_id]++;
                        } else {
                            m_fake2xcoincidencesInR++;
                        }

                        m_trackerLayout2xZR_InR->Fill(globalPosClu.z(), globalPosClu.perp());
                        m_trackerLayout2xYX_InR->Fill(globalPosClu.x(), globalPosClu.y());
                    }
                    //----------------------------------------- 
                }       
            }  //end of cluster loop

        } else { // TFPX modules

            //the index in my histogram map
            int hist_id = -1;
            unsigned int ring_id = ring - 1;
            if (side == 1) {
                //this is a TFPX- hit on side 1
                hist_id = layer - 1;
            } else if (side == 2) {
                //this is a TFPX+ hit on side 2
                hist_id = 8 + layer - 1;
            }

            // Get the geomdet
            const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(detId));
            if (!geomDetUnit)
                continue;

            unsigned int nClu = 0;

            //fill the number of clusters for this module
            m_nClusters_TFPX->Fill(DSVit->size());   

            //now loop the clusters for each detector
            for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = DSVit->begin(); cluit != DSVit->end(); cluit++) {
                //increment the counters
                nClu++;
                cluCounter_TFPX[hist_id][ring_id]++;

                // determine the position
                MeasurementPoint mpClu(cluit->x(), cluit->y());
                Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
                Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

                //fill TkLayout histos
                m_trackerLayoutClustersZR_TFPX->Fill(globalPosClu.z(), globalPosClu.perp());
                m_trackerLayoutClustersYX_TFPX->Fill(globalPosClu.x(), globalPosClu.y());

                if (m_docoincidence) {
                    unsigned int coincidenceId;
                    edmNew::DetSet<SiPixelCluster>::const_iterator coincidenceCluster;
                    bool found = this->findCoincidence2x(detId, globalPosClu, false, coincidenceId, coincidenceCluster);
                    if (found) {
                        m_total2xcoincidences_TFPX++;
                        x2Counter_TFPX[hist_id][ring_id]++;

                        //now get the simlink detset
                        edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter = findSimLinkDetSet(rawid);
                        std::set<unsigned int> simTrackId = this->getSimTrackId(simLinkDSViter, cluit, false);
                        //now get the simlink detset based on the coincidence hit detid
                        simLinkDSViter = findSimLinkDetSet(coincidenceId);
                        std::set<unsigned int> coincidencesimTrackId = this->getSimTrackId(simLinkDSViter, coincidenceCluster, false);
                        std::set<unsigned int> intersection;
                        bool areSame = areSameSimTrackId(simTrackId, coincidencesimTrackId, intersection);

                        if (areSame) {
                            x2Counterreal_TFPX[hist_id][ring_id]++;
                        } else {
                            m_fake2xcoincidences_TFPX++;
                        }

                        m_trackerLayout2xZR_TFPX->Fill(globalPosClu.z(), globalPosClu.perp());
                        m_trackerLayout2xYX_TFPX->Fill(globalPosClu.x(), globalPosClu.y());

                        //done with 2 fold coincidences, now 3 fold
                        //only if we have a 2 fold coincidence we can search for a third one in another ring
                        found = false;
                        unsigned int coincidenceId3x;
                        edmNew::DetSet<SiPixelCluster>::const_iterator coincidenceCluster3x;
                        found = this->findCoincidence3x(detId, globalPosClu, false, coincidenceId3x, coincidenceCluster3x);
                        if (found) {
                            m_total3xcoincidences_TFPX++;
                            x3Counter_TFPX[hist_id][ring_id]++;

                            //now get the simlink detset based on the coincidence hit detid
                            edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter3x = findSimLinkDetSet(coincidenceId3x);
                            std::set<unsigned int> coincidencesimTrackId3x = this->getSimTrackId(simLinkDSViter3x, 
                                                                                                 coincidenceCluster3x, false);
                            std::set<unsigned int> intersection3x;
                            areSame = areSameSimTrackId(intersection, coincidencesimTrackId3x, intersection3x);

                            if (areSame) {
                                x3Counterreal_TFPX[hist_id][ring_id]++;
                            } else {
                                m_fake3xcoincidences_TFPX++;
                            }

                            m_trackerLayout3xZR_TFPX->Fill(globalPosClu.z(), globalPosClu.perp());
                            m_trackerLayout3xYX_TFPX->Fill(globalPosClu.x(), globalPosClu.y());
                        }

                    }
                    //-----------------------------------------
                    std::vector<unsigned int> coincidenceIdInR;
                    std::vector<edmNew::DetSet<SiPixelCluster>::const_iterator> coincidenceClusterInR;
                    std::vector<float> coincidenceClusterdr;
                    bool found2xinR = this-> findCoincidenceInR2x(detId, globalPosClu, false, coincidenceIdInR, coincidenceClusterInR, coincidenceClusterdr);
                    if (found2xinR) {
                        m_total2xcoincidencesInR_TFPX++;
                        x2CounterInR_TFPX[hist_id][ring_id]++;

                        int mindrIndex = std::min_element(coincidenceClusterdr.begin(), coincidenceClusterdr.end()) - coincidenceClusterdr.begin();

                        edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter = findSimLinkDetSet(rawid);
                        std::set<unsigned int> simTrackId = this->getSimTrackId(simLinkDSViter,cluit,false);
                        simLinkDSViter = findSimLinkDetSet(coincidenceIdInR.at(mindrIndex));
                        std::set<unsigned int> coincidencesimTrackId = this->getSimTrackId(simLinkDSViter, coincidenceClusterInR.at(mindrIndex), false);
                        std::set<unsigned int> intersection;
                        bool areSame = areSameSimTrackId(simTrackId, coincidencesimTrackId, intersection);

                        if (areSame) {
                            x2CounterrealInR_TFPX[hist_id][ring_id]++;
                        } else {
                            m_fake2xcoincidencesInR_TFPX++;
                        }

                        m_trackerLayout2xZR_InR_TFPX->Fill(globalPosClu.z(), globalPosClu.perp());
                        m_trackerLayout2xYX_InR_TFPX->Fill(globalPosClu.x(), globalPosClu.y());       
                    }
                    //-----------------------------------------
                }

            }      

        }

    } //end of module loop

    //ok, now I know the number of clusters/hits per ring per disk and should fill the histogram once for this event
    for (unsigned int i = 0; i < 8; i++) {  // TEPX
        //loop the disks
        for (unsigned int j = 0; j < 5; j++) {
            //and the rings
            m_diskHistosCluster[i]->Fill(j + 1, cluCounter[i][j]);
            m_diskHistosHits[i]->Fill(j + 1, hitCounter[i][j]);
            if (m_docoincidence) {
                m_diskHistos2x[i]->Fill(j + 1, x2Counter[i][j]);
                m_diskHistos2xreal[i]->Fill(j + 1, x2Counterreal[i][j]);
                m_diskHistos2xInR[i]->Fill(j + 1, x2CounterInR[i][j]);
                m_diskHistos2xrealInR[i]->Fill(j + 1, x2CounterrealInR[i][j]);
                m_diskHistos3x[i]->Fill(j + 1, x3Counter[i][j]);
                m_diskHistos3xreal[i]->Fill(j + 1, x3Counterreal[i][j]);
            }
        }
    }

    for (unsigned int i = 0; i < 16; i++) {  // TFPX
        //loop the disks
        for (unsigned int j = 0; j < 4; j++) {
            //and the rings
            m_diskHistosCluster_TFPX[i]->Fill(j + 1, cluCounter_TFPX[i][j]);
            m_diskHistosHits_TFPX[i]->Fill(j + 1, hitCounter_TFPX[i][j]);
            if (m_docoincidence) {
                m_diskHistos2x_TFPX[i]->Fill(j + 1, x2Counter_TFPX[i][j]);
                m_diskHistos2xreal_TFPX[i]->Fill(j + 1, x2Counterreal_TFPX[i][j]);
                m_diskHistos2xInR_TFPX[i]->Fill(j + 1, x2CounterInR_TFPX[i][j]);
                m_diskHistos2xrealInR_TFPX[i]->Fill(j + 1, x2CounterrealInR_TFPX[i][j]);
                m_diskHistos3x_TFPX[i]->Fill(j + 1, x3Counter_TFPX[i][j]);
                m_diskHistos3xreal_TFPX[i]->Fill(j + 1, x3Counterreal_TFPX[i][j]);
            }
        }
    }


    m_nevents++;
}

// ------------ method called once each job just after ending the event loop  ------------
void ITclusterAnalyzer::endJob() {

    if (m_storeClusterTree) {
        outFileCluster->Write("",TObject::kOverwrite);
        outFileCluster->Close();
    }

    std::cout << "IT cluster Analyzer processed " << m_nevents << " events!" << std::endl;
    if (m_docoincidence) {
        std::cout << "IT cluster Analyzer found " << m_fake2xcoincidences / (double)m_total2xcoincidences * 100 
                  << "\% fake double coincidences in TEPX modules." << std::endl;
        std::cout << "IT cluster Analyzer found " << m_fake3xcoincidences / (double)m_total3xcoincidences * 100 
                  << "\% fake triple coincidences in TEPX modules." << std::endl;
        std::cout << "IT cluster Analyzer found " << m_fake2xcoincidences_TFPX / (double)m_total2xcoincidences_TFPX * 100 
                  << "\% fake double coincidences in TFPX modules." << std::endl;
        std::cout << "IT cluster Analyzer found " << m_fake3xcoincidences_TFPX / (double)m_total3xcoincidences_TFPX * 100 
                  << "\% fake triple coincidences in TFPX modules." << std::endl;
        std::cout << "IT cluster Analyzer found " << m_fake2xcoincidencesInR / (double)m_total2xcoincidencesInR * 100
                  << "\% fake double coincidences in R in TEPX modules." << std::endl;
        std::cout << "IT cluster Analyzer found " << m_fake2xcoincidencesInR_TFPX / (double)m_total2xcoincidencesInR_TFPX * 100
                  << "\% fake double coincidences in R in TFPX modules." << std::endl;
    }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void ITclusterAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    edm::ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);

    //Specify that only 'tracks' is allowed
    //To use, remove the default given above and uncomment below
    //ParameterSetDescription desc;
    //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
    //descriptions.addDefault(desc);
}

//----------
//Adding function to find 2x coincidences in R
//COB - 21.May.2019
bool ITclusterAnalyzer::findCoincidenceInR2x(DetId thedetid, Global3DPoint theglobalPosClu, bool isTEPX, std::vector<unsigned int>& ovModIds, std::vector<edmNew::DetSet<SiPixelCluster>::const_iterator>& ovClusIds, std::vector<float>& ovDr) {

    bool found = false;

    //getting ring and module for initial cluster
    unsigned int thering = (tTopo->pxfBlade(thedetid));
    unsigned int thedisk = (tTopo->pxfDisk(thedetid));
    unsigned int theside = (tTopo->pxfSide(thedetid));

    //debugging...
    //std::string dumpId = tTopo->print(thedetid);
    //std::cout << "=> Initial detector Id: " << dumpId << std::endl;

    //go to the next ring
    uint32_t newring = thering + 1;
    if (isTEPX) {
        if (thering == 5) 
            return false;  //can't search for coincidences in R in the last ring
    } else {  //TFPX
        if (thering == 4)
            return false;  //can't search for coincidences in R in the last ring
    }

    //get the geomdet
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(thedetid));
    std::pair<float,float> phiSpan = geomDetUnit->surface().phiSpan();
    //std::pair<float,float> zSpan = geomDetUnit->surface().zSpan();
    //std::pair<float,float> rSpan = geomDetUnit->surface().rSpan();

    //define number of modules in each ring
    int nmod = -99;
    if (isTEPX) {
        if (newring==2) {
            nmod = 28;
        } else if (newring==3) {
            nmod = 36;
        } else if (newring==4) {
            nmod = 44;
        } else if (newring==5) {
            nmod = 48;
        }
    } else {  //TFPX
        if (newring==2) {
            nmod = 32;
        } else if (newring==3) {
            nmod = 24;
        } else if (newring==4) {
            nmod = 32;
        }
    }

    //debugging...
    //std::cout << "phiSpan " << phiSpan.first << "," << phiSpan.second << std::endl;
    //std::cout << "zSpan " << zSpan.first << "," << zSpan.second << std::endl;
    //std::cout << "rSpan " << rSpan.first << "," << rSpan.second << std::endl;

    //loop over modules in new ring
    //find first the id of the first module in the new ring
    uint32_t tmpid = getModuleID(isTEPX, theside, thedisk, newring);
    for (int i=1; i<=nmod; i++) {

        found = false;

        DetId tmp(tmpid);
       
        //debugging... 
        //std::cout << "tmp " << tTopo->print(tmp) << std::endl;

        const GeomDetUnit* geomDetUnit_tmp(tkGeom->idToDetUnit(tmpid));
        std::pair<float,float> phiSpan_tmp = geomDetUnit_tmp->surface().phiSpan();
        //std::pair<float,float> rSpan_tmp = geomDetUnit_tmp->surface().rSpan();   

        //debugging...
        //std::cout << "phiSpan (new) " << phiSpan_tmp.first << "," << phiSpan_tmp.second << std::endl;
        //std::cout << "rSpan (new) " << rSpan_tmp.first << "," << rSpan_tmp.second << std::endl;

        unsigned int foundDetId;
        edmNew::DetSet<SiPixelCluster>::const_iterator foundCluster;

        //checking if module is in the phi sign transition (pi --> -pi)
        //not pretty... 
        bool isInPhiTransition = false;
        if ( ((phiSpan_tmp.first > 0) && (phiSpan_tmp.second < 0)) ) {
            if ( (phiSpan.first > 0) && (phiSpan.second < 0) ) {
                if ( ((phiSpan_tmp.first > phiSpan.first) && (phiSpan_tmp.first <= TMath::Pi())) ||
                     ((phiSpan_tmp.second < phiSpan.second) && (phiSpan_tmp.second >= -TMath::Pi())) ) {
                    isInPhiTransition = true;
                } 
            } else if ( (phiSpan.first < 0) && (phiSpan.second < 0) ) {
                if ( (phiSpan_tmp.second > phiSpan.first) && (phiSpan_tmp.second < phiSpan.second) ) {
                    isInPhiTransition = true;
                }
            } else if ( (phiSpan.first > 0) && (phiSpan.second > 0) ) {       
                if ( (phiSpan_tmp.first > phiSpan.first) && (phiSpan_tmp.first < phiSpan.second) ) {
                    isInPhiTransition = true;
                }
            }
        } else if ( ((phiSpan_tmp.first > 0) && (phiSpan_tmp.second > 0)) ) {
            if ( (phiSpan.first > 0) && (phiSpan.second < 0) ) {
                if ( (phiSpan_tmp.second > phiSpan.first) && (phiSpan_tmp.second <= TMath::Pi()) ) {
                    isInPhiTransition = true;
                }
            }
        } else if ( ((phiSpan_tmp.first < 0) && (phiSpan_tmp.second < 0)) ) {
            if ( (phiSpan.first > 0) && (phiSpan.second < 0) ) {
                if ( (phiSpan_tmp.first < phiSpan.second) && (phiSpan_tmp.first >= -TMath::Pi()) ) {
                    isInPhiTransition = true;
                }
            }
        }

        //all modules in new ring have the same r, so we need to look for clusters only in the vicinity of the original module.
        //check if module in new ring is within the same phi region as original module
        if ( ((phiSpan_tmp.first > phiSpan.first) && (phiSpan_tmp.first < phiSpan.second)) || 
             ((phiSpan_tmp.second > phiSpan.first) && (phiSpan_tmp.second < phiSpan.second)) ||
             isInPhiTransition ) {

            //debugging...
            //std::cout << "module " << tTopo->pxfModule(tmpid) << " overlaps in phi with " << tTopo->pxfModule(thedetid) << std::endl;
   
            //check if there are clusters in this new module
            edmNew::DetSetVector<SiPixelCluster>::const_iterator theit = clusters->find(tmpid);
            if (theit == clusters->end()) {
                tmpid = (tmpid & 0xFFFFFC03) | (((i+1) & 0xFF) << 2);
                continue;
            }

            //debugging...
            //std::cout << "there are clusters in module " << tTopo->pxfModule(tmpid) << ". checking for overlaps..." << std::endl;

            unsigned int nClu = 0;
            double r_min = 1000.;
            std::vector<Residual> r_vec;

            foundCluster = theit->end();

            //loop over clusters in module and check if they overlap with the original cluster
            for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = theit->begin(); cluit != theit->end(); cluit++) {

                //determine the position
                MeasurementPoint mpClu(cluit->x(), cluit->y());
                Local3DPoint localPosClu = geomDetUnit_tmp->topology().localPosition(mpClu);
                Global3DPoint globalPosClu = geomDetUnit_tmp->surface().toGlobal(localPosClu);

                //now check that the global position is within the cuts
                if (fabs(globalPosClu.x() - theglobalPosClu.x()) < m_dx
                    && fabs(globalPosClu.y() - theglobalPosClu.y()) < m_dy
                    && fabs(globalPosClu.z() - theglobalPosClu.z()) < m_dz) {

                    nClu++;
                    //debugging...
                    //std::cout << "found a cluster that overlaps within dx " << m_dx << " dy " << m_dy << " dz " << m_dz << std::endl;

                    double delta_x = fabs(globalPosClu.x() - theglobalPosClu.x());
                    double delta_y = fabs(globalPosClu.y() - theglobalPosClu.y());
                    Residual r(delta_x, delta_y);
                    r_vec.push_back(r);

                    //debugging...
                    //std::cout << "and has dr of " << r.dr << std::endl;

                    if (r.dr < r_min) {
                        r_min = r.dr;
                        found = true;
                        foundCluster = cluit;
                        foundDetId = tmpid;
                    }

                }

            }

            if (nClu > 1) {
                for (auto r : r_vec) {
                    if (r.dr == r_min) {
                        if (isTEPX) {
                            m_residualX_InR->Fill(r.dx);
                            m_residualY_InR->Fill(r.dy);
                            m_residualR_InR->Fill(r.dr); 
                        } else {  //TFPX
                            m_residualX_TFPX_InR->Fill(r.dx);
                            m_residualY_TFPX_InR->Fill(r.dy);
                            m_residualR_TFPX_InR->Fill(r.dr);
                        }
                    }
                }
            }

            //debugging...
            //std::cout << "closest overlapping cluster for module " << foundDetId << " is " << foundCluster 
            //          << " with dr of " << r_min << std::endl;

            //store info in vectors in case there is more than one module overlapping
            if (found) {
                ovModIds.push_back(foundDetId);
                ovClusIds.push_back(foundCluster);    
                ovDr.push_back(r_min);
            }

        }

        //go to the next module in the new ring
        tmpid = (tmpid & 0xFFFFFC03) | (((i+1) & 0xFF) << 2);

    }

    //debugging...
    //std::cout << "Found overlapping clusters in " << ovModIds.size() << " modules." << std::endl;

    bool foundClusters;
    if (ovClusIds.size() > 0) {
        foundClusters = true;
    } else {
        foundClusters = false;
    }

    return foundClusters;   

}

//---------

bool ITclusterAnalyzer::findCoincidence2x(DetId thedetid, Global3DPoint theglobalPosClu, bool isTEPX, unsigned int& foundDetId, edmNew::DetSet<SiPixelCluster>::const_iterator& foundCluster) {

    bool found = false;
    uint32_t rawid = thedetid.rawId();
    uint32_t newid = rawid;
    //now I have the raw ID and can mess with the bits
    //the side, layer and ring are the same and I just have to increment or decrement the module number
    unsigned int themodule = (tTopo->pxfModule(thedetid));
    unsigned int thering = (tTopo->pxfBlade(thedetid));

    //in order to avoid duplicates, only look in the next module clockwise
    //depending on the ring, if I am already on the module with the highest module id in the ring, I need to go to the first one
    uint32_t newmodule = themodule + 1;
    if (isTEPX) {
        if (thering == 1 && themodule == 20)
            newmodule = 1;
        else if (thering == 2 && themodule == 28)
            newmodule = 1;
        else if (thering == 3 && themodule == 36)
            newmodule = 1;
        else if (thering == 4 && themodule == 44)
            newmodule = 1;
        else if (thering == 5 && themodule == 48)
            newmodule = 1;
    } else {  //TFPX
        if (thering == 1 && themodule == 20)
            newmodule = 1;
        else if (thering == 2 && themodule == 32)
            newmodule = 1;
        else if (thering == 3 && themodule == 24)
            newmodule = 1;
        else if (thering == 4 && themodule == 32)
            newmodule = 1;
    }

    //now encode
    newid = (newid & 0xFFFFFC03) | ((newmodule & 0xFF) << 2);

    //now I have a raw id of the module I want to use
    DetId id(newid);
    //unsigned int ring = (tTopo->pxfBlade(id));
    //unsigned int module = (tTopo->pxfModule(id));

    edmNew::DetSetVector<SiPixelCluster>::const_iterator theit = clusters->find(id);
    if (theit == clusters->end()) {
        return false;
    }

    // Get the geomdet
    const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(id));

    unsigned int nClu = 0;
    //at the end of the day, need to find the closest coincidence hit, so store the minimum 2D distance in a temporary variable and a vector for all values
    double r_min = 1000.;
    std::vector<Residual> r_vec;

    //make the return value end();
    foundCluster = theit->end();

    for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = theit->begin(); cluit != theit->end(); cluit++) {

        // determine the position
        MeasurementPoint mpClu(cluit->x(), cluit->y());
        Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
        Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

        //now check that the global position is within the cuts
        if (fabs(globalPosClu.x() - theglobalPosClu.x()) < m_dx
            && fabs(globalPosClu.y() - theglobalPosClu.y()) < m_dy
            && fabs(globalPosClu.z() - theglobalPosClu.z()) < m_dz) {
            nClu++;

            double delta_x = fabs(globalPosClu.x() - theglobalPosClu.x());
            double delta_y = fabs(globalPosClu.y() - theglobalPosClu.y());
            Residual r(delta_x, delta_y);
            r_vec.push_back(r);

            if (r.dr < r_min) {
                r_min = r.dr;
                found = true;
                // I assign this here to always have the closest cluster
                foundCluster = cluit;
                foundDetId = newid;
            }

            //std::cout << "Found matching cluster # " << nClu << std::endl;

            //std::cout << "Original x: " << theglobalPosClu.x() << " y: " << theglobalPosClu.y() << " z: " << theglobalPosClu.z() << " ring: " << thering << " module: " << themodule << std::endl;
            //std::cout << "New      x: " << globalPosClu.x() << " y: " << globalPosClu.y() << " z: " << globalPosClu.z() << " ring: " << ring << " module: " << module << std::endl;
        }
    }

    if (nClu > 1) {
        //std::cout << "Warning, found " << nClu << "Clusters within the cuts - the minimum distance is " << r_min << "!" << std::endl;
        //std::cout << "All distances: ";
        for (auto r : r_vec) {
            //r.print();
            if (r.dr == r_min) {
                if (isTEPX) {
                    m_residualX->Fill(r.dx);
                    m_residualY->Fill(r.dy);
                    m_residualR->Fill(r.dr);
                } else {
                    m_residualX_TFPX->Fill(r.dx);
                    m_residualY_TFPX->Fill(r.dy);
                    m_residualR_TFPX->Fill(r.dr);
                }
            }
        }
        //std::cout << std::endl;
    }

    return found;

}

bool ITclusterAnalyzer::findCoincidence3x(DetId thedetid, Global3DPoint theglobalPosClu, bool isTEPX, unsigned int& foundDetId, edmNew::DetSet<SiPixelCluster>::const_iterator& foundCluster) {

    bool found = false;
    uint32_t rawid = thedetid.rawId();
    uint32_t newid = rawid;
    //now I have the raw ID and can mess with the bits
    //the side and layer are the same and I just have to look in a lower ring
    //unsigned int themodule = (tTopo->pxfModule(thedetid));
    unsigned int thering = (tTopo->pxfBlade(thedetid));

    unsigned int maxmodule = 0;
    unsigned int newring = 0;

    if (isTEPX) {
        if (thering > 1 && thering <= 5) {
            newring = thering - 1; //look in a lower ring
            //now get the max range of modules for the respective ring
            if (newring == 1)
                maxmodule = 20;
            else if (newring == 2)
                maxmodule = 28;
            else if (newring == 3)
                maxmodule = 36;
            else if (newring == 4)
                maxmodule = 44;
        } else
            return false;
    } else {
        if (thering > 1 && thering <= 4) {
            newring = thering - 1; //look in a lower ring
            //now get the max range of modules for the respective ring
            if (newring == 1)
                maxmodule = 20;
            else if (newring == 2)
                maxmodule = 32;
            else if (newring == 3)
                maxmodule = 24;
            //else if (newring == 4)
            //    maxmodule = 32;
        } else
            return false;
    }

    unsigned int nClu = 0;
    //to make sure we only use the closest hit
    double r_min = 1000.;
    std::vector<Residual> r_vec;

    //make the return value end();
    //foundCluster = theit->end();

    for (uint32_t newmodule = 1; newmodule <= maxmodule; newmodule++) {
        newid = (newid & 0xFFFC0C03) | ((newring & 0x3F) << 12) | ((newmodule & 0xFF) << 2);

        DetId id(newid);
        //unsigned int ring = (tTopo->pxfBlade(id));
        //unsigned int module = (tTopo->pxfModule(id));

        edmNew::DetSetVector<SiPixelCluster>::const_iterator theit = clusters->find(id);
        if (theit == clusters->end()) {
            return false;
        }
        // Get the geomdet
        const GeomDetUnit* geomDetUnit(tkGeom->idToDetUnit(id));
        if (!geomDetUnit)
            continue;

        for (edmNew::DetSet<SiPixelCluster>::const_iterator cluit = theit->begin(); cluit != theit->end(); cluit++) {

            // determine the position
            MeasurementPoint mpClu(cluit->x(), cluit->y());
            Local3DPoint localPosClu = geomDetUnit->topology().localPosition(mpClu);
            Global3DPoint globalPosClu = geomDetUnit->surface().toGlobal(localPosClu);

            //now check that the global position is within the cuts
            if (fabs(globalPosClu.x() - theglobalPosClu.x()) < m_dx
                && fabs(globalPosClu.y() - theglobalPosClu.y()) < m_dy
                && fabs(globalPosClu.z() - theglobalPosClu.z()) < m_dz) {
                nClu++;
                //found = true;
                //foundCluster = cluit;
                //foundDetId = newid;

                double delta_x = fabs(globalPosClu.x() - theglobalPosClu.x());
                double delta_y = fabs(globalPosClu.y() - theglobalPosClu.y());
                Residual r(delta_x, delta_y);
                r_vec.push_back(r);

                if (r.dr < r_min) {
                    r_min = r.dr;
                    found = true;
                    // i assign this here to be sure to always have the closest cluster
                    foundCluster = cluit;
                    foundDetId = newid;
                    //std::cout << "New det ID: " << newid << std::endl;
                }
                //std::cout << "Found matching cluster # " << nClu << " which is a 3x coincidence" << std::endl;
                //std::cout << "Original x: " << theglobalPosClu.x() << " y: " << theglobalPosClu.y() << " z: " << theglobalPosClu.z() << " ring: " << thering << " module: " << themodule << std::endl;
                //std::cout << "New      x: " << globalPosClu.x() << " y: " << globalPosClu.y() << " z: " << globalPosClu.z() << " ring: " << ring << " module: " << module << std::endl;
            }
        }
    }
    //if (found && nClu > 1) {
    //std::cout << "3X COINCIDENCE - Warning, found " << nClu << "Clusters within the cuts - the minimum distance is " << r_min << "!" << std::endl;
    //std::cout << "All distances: ";
    //for (auto r : r_vec) {
    //r.print();
    //}
    //std::cout << std::endl;
    //}
    return found;
}

edm::DetSetVector<PixelDigiSimLink>::const_iterator ITclusterAnalyzer::findSimLinkDetSet(unsigned int thedetid) {
    ////basic template
    edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDS = simlinks->find(thedetid);
    return simLinkDS;
}

std::set<unsigned int> ITclusterAnalyzer::getSimTrackId(edm::DetSetVector<PixelDigiSimLink>::const_iterator simLinkDSViter, edmNew::DetSet<SiPixelCluster>::const_iterator cluster, bool print) {
    int size = cluster->size();
    std::set<unsigned int> simTrackIds;

    for (int i = 0; i < size; i++) {

        SiPixelCluster::Pixel pix = cluster->pixel(i);
        unsigned int clusterChannel = PixelDigi::pixelToChannel(pix.x, pix.y);

        if (simLinkDSViter != simlinks->end()) {
            for (edm::DetSet<PixelDigiSimLink>::const_iterator it = simLinkDSViter->data.begin(); it != simLinkDSViter->data.end(); it++) {
                if (clusterChannel == it->channel()) {
                    simTrackIds.insert(it->SimTrackId());
                    if (print)
                        std::cout << "Channel: " << clusterChannel << " SimTrack ID: " << it->SimTrackId() << std::endl;
                }
            }
        }
    }
    //if(simTrackIds.size() != 1){
    //std::cout << "WARNING: have more than 1 simTrackId for this cluster! " << std::endl;
    //return 0;
    //}
    return simTrackIds;
}

bool ITclusterAnalyzer::areSameSimTrackId(std::set<unsigned int> first, std::set<unsigned int> second, std::set<unsigned int>& intersection) {
    //method to check if the sim Track id is present in both sets
    //std::set<unsigned int> intersection;
    std::set_intersection(first.begin(), first.end(), second.begin(), second.end(), std::inserter(intersection, intersection.begin()));
    if (!intersection.size()) {
        //std::cout << "WARNING, these clusters have not been caused by the same SimTrackID" << std::endl;
        return false;
    } else if (intersection.size() == 1) {
        return true;
    } else {
        //std::cout << "WARNING: both clusters caused by multiple tracks!" << std::endl;
        return true;
    }
}

//----------
//Adding function to construct DetId from PXFDetId
//COB - 22.May.2019
uint32_t ITclusterAnalyzer::getModuleID(bool isTEPX, unsigned int side, unsigned int disk, unsigned int ring) {

    //std::cout << "isTEPX " << isTEPX << " side " << side << " disk " << disk << " ring " << ring << std::endl;

    uint32_t modid = -999;

    if (isTEPX) {
        if (side==1) {
            if (disk==9) {
                if (ring==2) {
                    modid = 346301444;
                } else if (ring==3) {
                    modid = 346305540;
                } else if (ring==4) {
                    modid = 346309636;
                } else if (ring==5) {
                    modid = 346313732;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << " side " << side << "!" << std::endl;
                    return modid;
                }
            } else if (disk==10) {
                if (ring==2) {
                    modid = 346563588;
                } else if (ring==3) {
                    modid = 346567684;
                } else if (ring==4) {
                    modid = 346571780;
                } else if (ring==5) {
                    modid = 346575876;
                } else {
                   std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                   return modid;
                }
            } else if (disk==11) {
                if (ring==2) {
                    modid = 346825732;
                } else if (ring==3) {
                    modid = 346829828;
                } else if (ring==4) {
                    modid = 346833924;
                } else if (ring==5) {
                    modid = 346838020;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==12) {
                if (ring==2) {
                    modid = 347087876;
                } else if (ring==3) {
                    modid = 347091972;
                } else if (ring==4) {
                    modid = 347096068;
                } else if (ring==5) {
                    modid = 347100164;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else {
                std::cout << "Non-existent disk for TEPX!" << std::endl;
                return modid;
            }
        } else if (side==2) {
            if (disk==9) {
                if (ring==2) {
                    modid = 354690052;
                } else if (ring==3) {
                    modid = 354694148;
                } else if (ring==4) {
                    modid = 354698244;
                } else if (ring==5) {
                    modid = 354702340;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==10) {
                if (ring==2) {
                    modid = 354952196;
                } else if (ring==3) {
                    modid = 354956292;
                } else if (ring==4) {
                    modid = 354960388;
                } else if (ring==5) {
                    modid = 354964484;
                } else {
                   std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                   return modid;
                }
            } else if (disk==11) {
                if (ring==2) {
                    modid = 355214340;
                } else if (ring==3) {
                    modid = 355218436;
                } else if (ring==4) {
                    modid = 355222532;
                } else if (ring==5) {
                    modid = 355226628;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==12) {
                if (ring==2) {
                    modid = 355476484;
                } else if (ring==3) {
                    modid = 355480580;
                } else if (ring==4) {
                    modid = 355484676;
                } else if (ring==5) {
                    modid = 355488772;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else {
                std::cout << "Non-existent disk for TEPX!" << std::endl;
                return modid;
            }
        } else {
            std::cout << "Non-existent side!" << std::endl;
            return modid;
        }
    } else {  //TFPX

        if (side==1) {
            if (disk==1) {
                if (ring==2) {
                    modid = 344204292;
                } else if (ring==3) {
                    modid = 344208388;
                } else if (ring==4) {
                    modid = 344212484;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==2) {
                if (ring==2) {
                    modid = 344466436;
                } else if (ring==3) {
                    modid = 344470532;
                } else if (ring==4) {
                    modid = 344474628;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==3) {
                if (ring==2) {
                    modid = 344728580;
                } else if (ring==3) {
                    modid = 344732676;
                } else if (ring==4) {
                    modid = 344736772;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==4) {
                if (ring==2) {
                    modid = 344990724;
                } else if (ring==3) {
                    modid = 344994820;
                } else if (ring==4) {
                    modid = 344998916;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==5) {
                if (ring==2) {
                    modid = 345252868;
                } else if (ring==3) {
                    modid = 345256964;
                } else if (ring==4) {
                    modid = 345261060;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==6) {
                if (ring==2) {
                    modid = 345515012;
                } else if (ring==3) {
                    modid = 345519108;
                } else if (ring==4) {
                    modid = 345523204;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==7) {
                if (ring==2) {
                    modid = 345777156;
                } else if (ring==3) {
                    modid = 345781252;
                } else if (ring==4) {
                    modid = 345785348;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==8) {
                if (ring==2) {
                    modid = 346039300;
                } else if (ring==3) {
                    modid = 346043396;
                } else if (ring==4) {
                    modid = 346047492;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else {
                std::cout << "Non-existent disk for TFPX!" << std::endl;
                return modid;
            }
        } else if (side==2) {
            if (disk==1) {
                if (ring==2) {
                    modid = 352592900;
                } else if (ring==3) { 
                    modid = 352596996;
                } else if (ring==4) {
                    modid = 352601092;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==2) {
                if (ring==2) {
                    modid = 352855044;
                } else if (ring==3) {
                    modid = 352859140;
                } else if (ring==4) {
                    modid = 352863236;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==3) {
                if (ring==2) {
                    modid = 353117188;
                } else if (ring==3) {
                    modid = 353121284;
                } else if (ring==4) {
                    modid = 353125380;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==4) {
                if (ring==2) {
                    modid = 353379332;
                } else if (ring==3) {
                    modid = 353383428;
                } else if (ring==4) {
                    modid = 353387524;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==5) {
                if (ring==2) {
                    modid = 353641476;
                } else if (ring==3) {
                    modid = 353645572;
                } else if (ring==4) {
                    modid = 353649668;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==6) {
                if (ring==2) {
                    modid = 353903620;
                } else if (ring==3) {
                    modid = 353907716;
                } else if (ring==4) {
                    modid = 353911812;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==7) {
                if (ring==2) {
                    modid = 354165764;
                } else if (ring==3) {
                    modid = 354169860;
                } else if (ring==4) {
                    modid = 354173956;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else if (disk==8) {
                if (ring==2) {
                    modid = 354427908;
                } else if (ring==3) {
                    modid = 354432004;
                } else if (ring==4) {
                    modid = 354436100;
                } else {
                    std::cout << "Non-existent ring number for disk " << disk << "!" << std::endl;
                    return modid;
                }
            } else {
                std::cout << "Non-existent disk for TFPX!" << std::endl;
                return modid;
            }                            
        } else {
            std::cout << "Non-existent side!" << std::endl;
            return modid;
        }

    }

    return modid;

}

DEFINE_FWK_MODULE(ITclusterAnalyzer);
