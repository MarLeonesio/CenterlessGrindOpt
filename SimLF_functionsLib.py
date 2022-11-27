# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:01:28 2022

@author: leonesio
"""
#----------------------------------------------------------------------------------
def CenterlessLF(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,**opt):
    """
Created on Thu Apr 21 09:01:28 2022
It launches centerless grinding simulation and returns the FFT of the
profile. By default, the initial workpiece profile is given by uniform
noise.
INPUT:
 - dw: workpiece diameter in mm
 - ds: operating wheel diameter in mm
 - dcw: regulating wheel diameter
 - KK: stiffness factor
 - lc: contact length in mm
 - fz: feed per revolute in mm/rev
 - gamma: blade angle in deg
 - Hw: workpiece heigth in mm
 - NRev: revolutions number
 
OPTIONAL arguments
 -InitProf=filename: it loads the initial profile from the file filename. 
  If filename is '', then a GUI appears to browse for the file. Measurement
  in mm.
 -Nd: dimension of workpiece array (default 7200)
 -NoisAmpl: amplitude of the white noise used to initialize the
             workpiece profile if a user profile is not available in mm (default
                0.1)
 -NLobes: band lobes limit of the FIR filter for contact filtering
              (defaul 100)

Output:
    tuple: RD, DetInd, RDToT, Aval
    
    -RD: workpiece profile [mm]
    -DetInd: detachindex measuring the workpiece-wheel detachment level
    -RDToT: RD along with the whole simulation [mm]
    -Aval: depth of cut [mm]
    -ProxyAE: depth of cut 
        
@author: Marco Leonesio
"""
# Libraries
    import numpy as np
    from scipy import signal as sigs    
    from tkinter import filedialog
    import matplotlib.pyplot as plt

    #Initializations of default parametes
    Init_Prof_Flag=False
    Nd=7200 # Size of workpiece array
    NoisAmpl=5e-3 # Initial Noise on workpiece surface (if profile is not defined)
    NLobes=100 # References number of lobes to get zero at bandLobes: it determines the filter order
    ctTouch=0 # becomes 1 if wheel and workpiece touches at the current angular step
    ctDetach=0 # becomes 1 if wheel and workpiece do not touch at the current angular step
    
    #Pre-allocating output variables
    AvalNCl=np.empty(int(Nd*Nrev))
    Aval=np.empty(int(Nd*Nrev))
    RDToT=np.empty(int(Nd*Nrev))
    ProxyAE=np.empty(int(Nd*Nrev))
    DetInd=np.empty(int(Nd*Nrev))
    
    #Handle the options
    for key, value in opt.items():
        if key=="InitProf":
           ProfFilename=opt[key]
           Init_Prof_Flag=True
        elif  key=="Nd":
              Nd=opt[key]
        elif  key=="NoisAmpl":
              RDInit=opt[key]
        elif  key=="NLobes":
              NLobes=opt[key]
        else:
              raise Exception("Undefined optional argument ",key)
              
    #Contact filtering computation
    BF, FilterOrder, CentralIndex=ContactFilt(lc,dw,Nd,NLobes)          
    
    # lambda function for transparent circular indexing
    CI=lambda x: (x-1)%Nd
    
    betaS=np.arcsin(2*hw/(ds+dw))
    betaCW=np.arcsin(2*hw/(dcw+dw))
    beta=betaS+betaCW
    alpha=np.pi/2-gamma*np.pi/180-(betaS)
    k1=np.sin(beta)/np.sin((alpha)+(beta))
    k2=np.sin(alpha)/np.sin((alpha)+(beta))
       
    
    #Loading external WP profile, if selected
    if Init_Prof_Flag:
        if not(ProfFilename):
            ProfFilename=filedialog.askopenfilename()
        DataProf = np.loadtxt(ProfFilename)
        # Interpolation by 0 padding in frequency domain
        if len(DataProf)%2:
            raise Exception('Odd profile points not supported yet')
        FFTInitPro=np.fft.fft(DataProf)
        ZRs=np.zeros(((Nd-(len(DataProf)))//2))
        FFTInitProDense=np.concatenate((FFTInitPro[:(len(FFTInitPro)//2)],ZRs,np.array(FFTInitPro[(len(FFTInitPro))//2,None]),ZRs,FFTInitPro[(len(FFTInitPro))//2+1:]))
        RD=abs(np.fft.ifft(FFTInitProDense)*Nd/len(DataProf))
    else:
        RD=np.copy(RDInit)
             
    # Creation buffer for filter
    BufferXRD=RD[CI(np.arange(-FilterOrder,0))+1]
    BoolRD=np.zeros_like(RD, dtype=bool)
    
    # Computation of the indeces corresponding to the delayed positions at
    # rubbing wheel and work rest
    ct1=int(np.round(alpha/2/np.pi*Nd)) # Index associted to K1
    ct2=int(np.round((np.pi-beta)/2/np.pi*Nd)) # Index associted to K2

    if ct1<FilterOrder+1: # Verifying that the filter does not interfere with neckrest and/or rubbing wheel
        raise Exception('Too large wave filtering') 
    ShiftFilter=CentralIndex+1;

    # Recuperate the initial position skipping air cutting
    
    X0=-k1*RD[CI(0-ct1)]+k2*RD[CI(0-ct2)]+RD[CI(0)]
    for theta in range(int(Nd*Nrev)):
        X=fz*(theta)/Nd+X0*0.9 # Feed position (corrected with 0.9*X0) 
        RDBuff=X+k1*RD[CI(theta-ct1)]-k2*RD[CI(theta-ct2)] # Compute workpiece radius reduction
        AvalNCl[theta]=(RDBuff-RD[CI(theta)])*KK # Compute intersection
        Aval[theta]=0.5*(AvalNCl[theta]+abs(AvalNCl[theta])) # Clipping intersection
        if Aval[theta]>0:
            RD[CI(theta)]=Aval[theta]+RD[CI(theta)]
            ctTouch+=1
            BoolRD[CI(theta)]=True
        else:
            ctDetach+=1
            BoolRD[CI(theta)]=False
            
        RDToT[theta]=RD[CI(theta)] # Storing the cumulated wear
        ProxyAE[theta]=RDBuff-RD[CI(theta)] # Proxy of Acoustic Sensor (currently not used)
        DetInd[theta]=ctDetach/(ctTouch+1)
          
        #apply filter
        BufferXRD[:len(BufferXRD)-1]=BufferXRD[1:]
        BufferXRD[-1]=RD[CI(theta-ShiftFilter+int(np.floor(FilterOrder/2)))]
        if BoolRD[CI(theta-ShiftFilter)]:
            RD[CI(theta-ShiftFilter)]=np.matmul(BF,BufferXRD)
    
    return RD, DetInd, RDToT, Aval, ProxyAE


#------------------------------------------------------------------------------------

def ContactFilt(lc,dw,Nd,NLobes):
    
    import numpy as np
    from scipy import signal as sigs
    RefPointNum = Nd/NLobes; # reference number of points to get a zero at BandLobes GB: NLObes == BandLobes?
                             # for a non-rectangular window, a higher number of points is 
                             # required
    FilterOrder = np.floor(10*RefPointNum)
    FilterOrder = int(np.floor(FilterOrder/2)*2)+1 #it must be odd, to get an 
                                                 #odd number of coefficients
                                                 #REM: firwin2 does not can change the filter order
                                                 #as FIR in Matlab!!!
    RelCuttFreq = NLobes/Nd
    FreqNyq = np.floor(Nd/2) # Nyquist frequency
    CuttFreqR = np.pi/(lc/dw); # 0 gain frequency
    if CuttFreqR>=FreqNyq:
         # raise Exception("Wave filter cutting frequency too high! CuttFreqR= %f", CuttFreqR)  --> Previuos version, halting the optimization
         CuttFreqR=FreqNyq*0.99
    FiltFreq = np.concatenate(([0], np.logspace(np.log10(0.05*CuttFreqR),np.log10(CuttFreqR),100)))
    FiltAmpl = 0.5*(1+(np.cos((lc/dw)*FiltFreq)));
    FiltFre01 = FiltFreq/FreqNyq
    FiltFre01[-1]=1 # normalized to 0:1
    BF = sigs.firwin2(FilterOrder,FiltFre01,FiltAmpl) # Generate filter
    ZG = np.sum(BF) 
    BF=BF/ZG
    FilterOrder = len(BF) # REM: fir2 can change the filter order
    CentralIndex = int(np.floor(FilterOrder/2)+1); #  B coefficients are symmetric respect to B(CentralIndex)
                                           # a buffer of CentraIndex elements must be used to avoid writing on
                                           # points that will be involved in subsequent filter calculations
    return BF, FilterOrder, CentralIndex

#------------------------------------------------------------------------------------

def RoundnessComp(Profile):
    
    import numpy as np
    
    NP = len(Profile)
    Ang = np.linspace(0, 2*np.pi,NP)
    MatrFitt = np.column_stack((np.ones((NP,1)), np.sin(Ang), np.cos(Ang)))
    Coeff = np.linalg.lstsq(MatrFitt,Profile,rcond=None)
    ProfileOff = Profile-np.matmul(MatrFitt,Coeff[0])
    return max(ProfileOff)-min(ProfileOff)
 
#------------------------------------------------------------------------------------   
    
class WP:
    
    def __init__(self, dw, lnght, profile, lastupdate):
        
        self.dw=dw
        self.lnght=lnght
        self.profile=profile
        self.lastupdate=lastupdate
        self.roundness=RoundnessComp(profile)
                
    def RoundnessComp(Profile):    
        import numpy as np    
        NP = len(Profile)
        Ang = np.linspace(0, 2*np.pi,NP)
        MatrFitt = np.column_stack((np.ones((NP,1)), np.sin(Ang), np.cos(Ang)))
        Coeff = np.linalg.lstsq(MatrFitt,Profile,rcond=None)
        ProfileOff = Profile-np.matmul(MatrFitt,Coeff[0])
        return max(ProfileOff)-min(ProfileOff)
    
    def __eq__(WP1,WP2):        
        if ((WP1.dw==WP2.dw) & (WP1.lnght==WP2.lnght) & (WP1.profile==WP2.profile)).any():
            return True
        else:
            return False       
            
#------------------------------------------------------------------------------------     

# In the first impementation **par must be fixed, as they correspond to the parameters used to generate HI-FI dataset (except for cat threshold, as the NN produces only continuous roundness)    
def h_NN1(x,NN,NNFacts,**par):
        
    """   
This is the function that predicts the WP roundness provided the process parameters vector x.
x elements are:

    x[0]   : WP height [mm]
    x[1]   : gamma ang [deg]
    x[2]   : feed velocity [mm/s]
    x[3]   : revolutions number
    x[4]   : WP lenght [mm]
    x[5]   : workpiece diameter [mm]
    x[6]   : grinding wheel diameter [mm]
    x[7]   : control wheel diameter [mm]
    x[8]   : control wheel angular velocity [rpm]
    x[9]   : Volumetric specific energy Ec [N/mm^2]
    x[10]  : Edge component force range for unit length [N/mm]
    x[11]  : Grit stiffness [N/mm]
    
Fixed parameters:
    
    OmegaS: Operating wheel angular velocity [rpm]
    Nd:     Number of radial elements
    NoisAmpl: Amplitude of random noise on initial profile [mm]
    dg: grit diameter [mm]
    GbondR: Fraction of abrasive over the wheel material    
    
"""
# Libraries
    import numpy as np
    
    # Initialization
    FixedParN=7

    # reassignement of inputs for readibility

    hw=x[0]
    gamma=x[1]
    vf=x[2]
    Nrev=x[3]
    WPlength=x[4]
    dw=x[5]
    ds=x[6]
    dcw=x[7]
    OmegaCw=x[8]
    Ec=x[9]
    Eb=x[10]
    Kgrit=x[11]   
    
    #Handle parameters
    if len(par.items())<5:
       raise Exception("Not enough fixed parameters") 
       
    for key, value in par.items():
        if key=="OmegaS":
           OmegaS=par[key]*2*np.pi/60 #Transformation in [rad/s]
        elif  key=="Nd":
              Nd=par[key]
        elif  key=="NoisAmpl":
              NoisAmpl=par[key]
        elif  key=="dg":
              dg=par[key]
        elif  key=="GbondR":
              GbondR=par[key]
        elif  key=="Rf":
              Rf=par[key]
        elif  key=="Eq":
              Eq=par[key]
        elif  key=="Kq":
              Kq=par[key]
        elif  key=="NLobes":
              NLobes=par[key]
        elif  key=="CatThr":
              CatThr=par[key]      
        else:
              raise Exception("Undefined parameters ",key)
    
    # Computation of intermediate parameters          
    Dq=1/(1/dw+1/ds)          
    Ke=WPlength*Eb
    
       #Kc computation
    Sg=(GbondR**(1/3))/dg #linear grit density
    Ng=Sg**2 #surface grit density
    AreaEl=dw*np.pi/Nd*WPlength #Element Area
    Kc=Kgrit*Ng*AreaEl #Contact stiffness for element area
    
       #Normalized specific enery
    Vs=OmegaS*ds/2   
    Ev=Ec*WPlength/Vs
    
       #Grinding stiffness
    OmegaW=OmegaCw*dcw/dw/60;   
    Kg=Ev*dw*OmegaW*np.pi
     
       #Feed per revolute
    fz=vf/OmegaW
    
    lc=(np.sqrt(2)*np.sqrt(32*Dq**2*Ke**2*Rf**4 + 
    8*Eq*fz*Kg*np.pi*Dq*Rf**2*WPlength \
    + Eq*dw*fz*np.pi*WPlength**2) \
    + 8*Dq*Ke*Rf**2)/(2*Eq*WPlength*np.pi)
     
        #Computation of stiffness factor    
    Ktot=1/(1/Kq+1/(Kc/AreaEl*lc*WPlength))
    KK=Ktot/(Ktot+Kg)*1.1 #Coefficient provided by an optimization
    
    #print(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd,NoisAmpl,NLobes) #for debug
    ProcessOut=CenterlessLF(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd=Nd,NoisAmpl=NoisAmpl,NLobes=NLobes)
    
    LoFiFeatures_Cont=RoundnessComp(ProcessOut[0]*1000) #From mm to micron
    
    # Continuous roundness classification
    if LoFiFeatures_Cont<CatThr:
        LoFiFeatures_Cat=0
    else:
        LoFiFeatures_Cat=1    
    #Normalization (the NN requires normalized inputs)   
    diagInStdInv=np.linalg.inv(np.diag(NNFacts['InStd']))
    xN=np.matmul((x-NNFacts['InMean']),diagInStdInv)
    
    #Creation of the input set
    Inset=np.append(np.append(xN,LoFiFeatures_Cat),LoFiFeatures_Cont)
    Inset=Inset[np.newaxis,:] # from (14,) to (1, 14)
    roundnessN=NN(Inset)
    roundnessN=roundnessN.numpy().item()
    roundness=roundnessN*NNFacts['OutStd']+NNFacts['OutMean']
    roundness=roundness.item()
    
    if roundness<CatThr:
        cat=0
    else:
       cat=1
    
    return roundness, cat
#------------------------------------------------------------------------------------

# In the first impementation **par must be fixed, as they correspond to the parameters used to generate HI-FI dataset (except for cat threshold, as the NN produces only continuous roundness)    
def h_NN1_v2(x,NN,NNFacts,**par):
        
    """   
This is the function that predicts the WP roundness provided the process parameters vector x.
x elements are:

    x[0]   : WP height [mm]
    x[1]   : gamma ang [deg]
    x[2]   : feed velocity [mm/s]
    x[3]   : Diametral material removal [mm]
    x[4]   : WP lenght [mm]
    x[5]   : workpiece diameter [mm]
    x[6]   : grinding wheel diameter [mm]
    x[7]   : control wheel diameter [mm]
    x[8]   : control wheel angular velocity [rpm]
    x[9]   : Volumetric specific energy Ec [N/mm^2]
    x[10]  : Edge component force range for unit length [N/mm]
    x[11]  : Grit stiffness [N/mm]
    
Fixed parameters:
    
    OmegaS: Operating wheel angular velocity [rpm]
    Nd:     Number of radial elements
    NoisAmpl: Amplitude of random noise on initial profile [mm]
    dg: grit diameter [mm]
    GbondR: Fraction of abrasive over the wheel material    
    
"""
# Libraries
    import numpy as np
    
    # Initialization
    FixedParN=7

    # reassignement of inputs for readibility

    hw=x[0]
    gamma=x[1]
    vf=x[2]
    Drem=x[3]
    WPlength=x[4]
    dw=x[5]
    ds=x[6]
    dcw=x[7]
    OmegaCw=x[8]
    Ec=x[9]
    Eb=x[10]
    Kgrit=x[11]   
    
    #Handle parameters
    if len(par.items())<5:
       raise Exception("Not enough fixed parameters") 
       
    for key, value in par.items():
        if key=="OmegaS":
           OmegaS=par[key]*2*np.pi/60 #Transformation in [rad/s]
        elif  key=="Nd":
              Nd=par[key]
        elif  key=="NoisAmpl":
              NoisAmpl=par[key]
        elif  key=="dg":
              dg=par[key]
        elif  key=="GbondR":
              GbondR=par[key]
        elif  key=="Rf":
              Rf=par[key]
        elif  key=="Eq":
              Eq=par[key]
        elif  key=="Kq":
              Kq=par[key]
        elif  key=="NLobes":
              NLobes=par[key]
        elif  key=="CatThr":
              CatThr=par[key]      
        else:
              raise Exception("Undefined parameters ",key)
    
    # Computation of intermediate parameters          
    Dq=1/(1/dw+1/ds)          
    Ke=WPlength*Eb
    
       #Kc computation
    Sg=(GbondR**(1/3))/dg #linear grit density
    Ng=Sg**2 #surface grit density
    AreaEl=dw*np.pi/Nd*WPlength #Element Area
    Kc=Kgrit*Ng*AreaEl #Contact stiffness for element area
    
       #Normalized specific enery
    Vs=OmegaS*ds/2   
    Ev=Ec*WPlength/Vs
    
       #Grinding stiffness
    OmegaW=OmegaCw*dcw/dw/60;   
    Kg=Ev*dw*OmegaW*np.pi
     
       #Feed per revolute
    fz=vf/OmegaW
    Nrev=Drem/fz
    
    lc=(np.sqrt(2)*np.sqrt(32*Dq**2*Ke**2*Rf**4 + 
    8*Eq*fz*Kg*np.pi*Dq*Rf**2*WPlength \
    + Eq*dw*fz*np.pi*WPlength**2) \
    + 8*Dq*Ke*Rf**2)/(2*Eq*WPlength*np.pi)
     
        #Computation of stiffness factor    
    Ktot=1/(1/Kq+1/(Kc/AreaEl*lc*WPlength))
    KK=Ktot/(Ktot+Kg)*1.1 #Coefficient provided by an optimization
    
    #print(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd,NoisAmpl,NLobes) #for debug
    ProcessOut=CenterlessLF(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd=Nd,NoisAmpl=NoisAmpl,NLobes=NLobes)
    
    LoFiFeatures_Cont=RoundnessComp(ProcessOut[0]*1000) #From mm to micron
    
    # Continuous roundness classification
    if LoFiFeatures_Cont<CatThr:
        LoFiFeatures_Cat=0
    else:
        LoFiFeatures_Cat=1    
   
    #Normalization (the NN requires normalized inputs)   
    diagInStdInv=np.linalg.inv(np.diag(NNFacts['InStd']))
    xN=np.matmul((x-NNFacts['InMean']),diagInStdInv)
    
    #Creation of the input set
    Inset=np.append(np.append(xN,LoFiFeatures_Cat),LoFiFeatures_Cont)
    Inset=Inset[np.newaxis,:] # from (14,) to (1, 14)
    roundnessN=NN(Inset)
    roundnessN=roundnessN.numpy().item()
    roundness=roundnessN*NNFacts['OutStd']+NNFacts['OutMean']
    roundness=roundness.item()
    
    if roundness<CatThr:
        cat=0
    else:
       cat=1
    
    return roundness, cat
#------------------------------------------------------------------------------------

# In the first impementation **par must be fixed, as they correspond to the parameters used to generate HI-FI dataset (except for cat threshold, as the NN produces only continuous roundness)    
def h_NN_LF(x,**par):
        
    """   
This is the function that predicts the WP roundness provided the process parameters vector x.
x elements are:

    x[0]   : WP height [mm]
    x[1]   : gamma ang [deg]
    x[2]   : feed velocity [mm/s]
    x[3]   : revolutions number
    x[4]   : WP lenght [mm]
    x[5]   : workpiece diameter [mm]
    x[6]   : grinding wheel diameter [mm]
    x[7]   : control wheel diameter [mm]
    x[8]   : control wheel angular velocity [rpm]
    x[9]   : Volumetric specific energy Ec [N/mm^2]
    x[10]  : Edge component force range for unit length [N/mm]
    x[11]  : Grit stiffness [N/mm]
    
Fixed parameters:
    
    OmegaS: Operating wheel angular velocity [rpm]
    Nd:     Number of radial elements
    NoisAmpl: Amplitude of random noise on initial profile [mm]
    dg: grit diameter [mm]
    GbondR: Fraction of abrasive over the wheel material    
    
"""
# Libraries
    import numpy as np
    
    # Initialization
    FixedParN=7

    # reassignement of inputs for readibility

    hw=x[0]
    gamma=x[1]
    vf=x[2]
    Nrev=x[3]
    WPlength=x[4]
    dw=x[5]
    ds=x[6]
    dcw=x[7]
    OmegaCw=x[8]
    Ec=x[9]
    Eb=x[10]
    Kgrit=x[11]   
    
    #Handle parameters
    if len(par.items())<5:
       raise Exception("Not enough fixed parameters") 
       
    for key, value in par.items():
        if key=="OmegaS":
           OmegaS=par[key]*2*np.pi/60 #Transformation in [rad/s]
        elif  key=="Nd":
              Nd=par[key]
        elif  key=="NoisAmpl":
              NoisAmpl=par[key]
        elif  key=="dg":
              dg=par[key]
        elif  key=="GbondR":
              GbondR=par[key]
        elif  key=="Rf":
              Rf=par[key]
        elif  key=="Eq":
              Eq=par[key]
        elif  key=="Kq":
              Kq=par[key]
        elif  key=="NLobes":
              NLobes=par[key]
        elif  key=="CatThr":
              CatThr=par[key]      
        else:
              raise Exception("Undefined parameters ",key)
    
    # Computation of intermediate parameters          
    Dq=1/(1/dw+1/ds)          
    Ke=WPlength*Eb
    
       #Kc computation
    Sg=(GbondR**(1/3))/dg #linear grit density
    Ng=Sg**2 #surface grit density
    AreaEl=dw*np.pi/Nd*WPlength #Element Area
    Kc=Kgrit*Ng*AreaEl #Contact stiffness for element area
    
       #Normalized specific enery
    Vs=OmegaS*ds/2   
    Ev=Ec*WPlength/Vs
    
       #Grinding stiffness
    OmegaW=OmegaCw*dcw/dw/60;   
    Kg=Ev*dw*OmegaW*np.pi
     
       #Feed per revolute
    fz=vf/OmegaW
    
    lc=(np.sqrt(2)*np.sqrt(32*Dq**2*Ke**2*Rf**4 + 
    8*Eq*fz*Kg*np.pi*Dq*Rf**2*WPlength \
    + Eq*dw*fz*np.pi*WPlength**2) \
    + 8*Dq*Ke*Rf**2)/(2*Eq*WPlength*np.pi)
     
        #Computation of stiffness factor    
    Ktot=1/(1/Kq+1/(Kc/AreaEl*lc*WPlength))
    KK=Ktot/(Ktot+Kg)*1.1 #Coefficient provided by an optimization
    
    #print(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd,NoisAmpl,NLobes) #for debug
    ProcessOut=CenterlessLF(dw,ds,dcw,KK,lc,fz,gamma,hw,Nrev,Nd=Nd,NoisAmpl=NoisAmpl,NLobes=NLobes)
    
    LoFiFeatures_Cont=RoundnessComp(ProcessOut[0]*1000) #From mm to micron
    
    # Continuous roundness classification
    roundness=LoFiFeatures_Cont
        
    if roundness<CatThr:
        cat=0
    else:
       cat=1
    
    return roundness, cat

#------------------------------------------------------------------------------------  

# In the first impementation **par must be fixed, as they correspond to the parameters used to generate HI-FI dataset (except for cat threshold, as the NN produces only continuous roundness)    
# h_NN2 is the version that does not use PB features
def h_NN2(x,NN,NNFacts,**par):
        
    """   
This is the function that predicts the WP roundness provided the process parameters vector x.
x elements are:

    x[0]   : WP height [mm]
    x[1]   : gamma ang [deg]
    x[2]   : feed velocity [mm/s]
    x[3]   : revolutions number
    x[4]   : WP lenght [mm]
    x[5]   : workpiece diameter [mm]
    x[6]   : grinding wheel diameter [mm]
    x[7]   : control wheel diameter [mm]
    x[8]   : control wheel angular velocity [rpm]
    x[9]   : Volumetric specific energy Ec [N/mm^2]
    x[10]  : Edge component force range for unit length [N/mm]
    x[11]  : Grit stiffness [N/mm]
    
Fixed parameters:
    
    OmegaS: Operating wheel angular velocity [rpm]
    Nd:     Number of radial elements
    NoisAmpl: Amplitude of random noise on initial profile [mm]
    dg: grit diameter [mm]
    GbondR: Fraction of abrasive over the wheel material    
    
"""
# Libraries
    import numpy as np
    
    # Initialization
    FixedParN=7

    # reassignement of inputs for readibility

    hw=x[0]
    gamma=x[1]
    vf=x[2]
    Nrev=x[3]
    WPlength=x[4]
    dw=x[5]
    ds=x[6]
    dcw=x[7]
    OmegaCw=x[8]
    Ec=x[9]
    Eb=x[10]
    Kgrit=x[11]   
    
    #Handle parameters
    if len(par.items())<5:
       raise Exception("Not enough fixed parameters") 
       
    for key, value in par.items():
        if key=="OmegaS":
           OmegaS=par[key]*2*np.pi/60 #Transformation in [rad/s]
        elif  key=="Nd":
              Nd=par[key]
        elif  key=="NoisAmpl":
              NoisAmpl=par[key]
        elif  key=="dg":
              dg=par[key]
        elif  key=="GbondR":
              GbondR=par[key]
        elif  key=="Rf":
              Rf=par[key]
        elif  key=="Eq":
              Eq=par[key]
        elif  key=="Kq":
              Kq=par[key]
        elif  key=="NLobes":
              NLobes=par[key]
        elif  key=="CatThr":
              CatThr=par[key]      
        else:
              raise Exception("Undefined parameters ",key)
    
  
    #Normalization (the NN requires normalized inputs)   
    diagInStdInv=np.linalg.inv(np.diag(NNFacts['InStd']))
    xN=np.matmul((x-NNFacts['InMean']),diagInStdInv)
    xN=xN[np.newaxis,:] # from (14,) to (1, 14)
    roundnessN=NN(xN)
    roundnessN=roundnessN.numpy().item()
    roundness=roundnessN*NNFacts['OutStd']+NNFacts['OutMean']
    roundness=roundness.item()
    
    if roundness<CatThr:
        cat=0
    else:
       cat=1
    
    return roundness, cat

#------------------------------------------------------------------------------------ 
#Definition of loss function
def LossFunCent(x,NN,NNFacts,h_NN, FixedPar, BlockVar, SigSlope, BarrWeight):
    
    import numpy as np
    
    # OverallInputN=12;
    # InputParLabels=["hw","gamma","vf","Nrev","WPLength","WPDiameter",
    #                 "OPWheelDiameter","CWheelDiameter","CwheelAngularVelocity",
    #                 "VolSpecificEnergy","EdgeComponentF","GritStiffness"]
    # xTot=np.empty(12)
       
    #BarrWeight (new parameter): Weight for the barrier definition
    
    for key, value in FixedPar.items():
        if key=="OmegaS":
              OmegaS=FixedPar[key]
        elif  key=="Nd":
              Nd=FixedPar[key]
        elif  key=="NoisAmpl":
              NoisAmpl=FixedPar[key]
        elif  key=="dg":
              dg=FixedPar[key]
        elif  key=="GbondR":
              GbondR=FixedPar[key]
        elif  key=="Rf":
              Rf=FixedPar[key]
        elif  key=="Eq":
              Eq=FixedPar[key]
        elif  key=="Kq":
              Kq=FixedPar[key]
        elif  key=="NLobes":
              NLobes=FixedPar[key]
        elif  key=="CatThr":
              CatThr=FixedPar[key]
        elif  key!="DataSet_X":              
              raise Exception("Undefined fixed parameters ",key)
    
    # Ordered meaning of input vector x entries:
    #0: hw
    #1: gamma
    #2: vf
    #3: Nrev
    #4: WPLength
    #5: WPDiameter
    #6: OPWheelDiameter
    #7: CWheelDiameter
    #8: CwheelAngularVelocity
    #9:VolSpecificEnergy
    #10:EdgeComponentF
    #11:GritStiffness
    
    xTot=[]
    ct=0      
    for key, value in BlockVar.items():
           if BlockVar[key]["Blocked"]:
               xTot=np.append(xTot,BlockVar[key]["value"])
           else:
               xTot=np.append(xTot,x[ct])
               ct+=1             
              
    Res=h_NN(xTot,NN,NNFacts,OmegaS=OmegaS,Nd=Nd,NoisAmpl=NoisAmpl,dg=dg,GbondR=GbondR,
    Rf=Rf,Eq=Eq,Kq=Kq,NLobes=NLobes,CatThr=CatThr)
     
    #Definition of barrier
    Barr=lambda y: .5 * (np.tanh(SigSlope * (y-CatThr)) + 1)
    Cost=Res[0]+BarrWeight*Barr(Res[0])
    return Cost

#Definition of loss function (unconstrained)

#------------------------------------------------------------------------------------  
#Definition of loss function unconstrained (without barrier)
def LossFunCentUnconst(x,NN,NNFacts,h_NN, FixedPar, BlockVar, SigSlope, BarrWeight):
    
    import numpy as np
    
    # OverallInputN=12;
    # InputParLabels=["hw","gamma","vf","Nrev","WPLength","WPDiameter",
    #                 "OPWheelDiameter","CWheelDiameter","CwheelAngularVelocity",
    #                 "VolSpecificEnergy","EdgeComponentF","GritStiffness"]
    # xTot=np.empty(12)
       
    #BarrWeight (new parameter): Weight for the barrier definition
    
    for key, value in FixedPar.items():
        if key=="OmegaS":
              OmegaS=FixedPar[key]
        elif  key=="Nd":
              Nd=FixedPar[key]
        elif  key=="NoisAmpl":
              NoisAmpl=FixedPar[key]
        elif  key=="dg":
              dg=FixedPar[key]
        elif  key=="GbondR":
              GbondR=FixedPar[key]
        elif  key=="Rf":
              Rf=FixedPar[key]
        elif  key=="Eq":
              Eq=FixedPar[key]
        elif  key=="Kq":
              Kq=FixedPar[key]
        elif  key=="NLobes":
              NLobes=FixedPar[key]
        elif  key=="CatThr":
              CatThr=FixedPar[key]
        elif  key!="DataSet_X":              
              raise Exception("Undefined fixed parameters ",key)
    
    # Ordered meaning of input vector x entries:
    #0: hw
    #1: gamma
    #2: vf
    #3: Nrev
    #4: WPLength
    #5: WPDiameter
    #6: OPWheelDiameter
    #7: CWheelDiameter
    #8: CwheelAngularVelocity
    #9:VolSpecificEnergy
    #10:EdgeComponentF
    #11:GritStiffness
    
    xTot=[]
    ct=0      
    for key, value in BlockVar.items():
           if BlockVar[key]["Blocked"]:
               xTot=np.append(xTot,BlockVar[key]["value"])
           else:
               xTot=np.append(xTot,x[ct])
               ct+=1             
              
    Res=h_NN(xTot,NN,NNFacts,OmegaS=OmegaS,Nd=Nd,NoisAmpl=NoisAmpl,dg=dg,GbondR=GbondR,
    Rf=Rf,Eq=Eq,Kq=Kq,NLobes=NLobes,CatThr=CatThr)
     
    Cost=Res[0]
    return Cost
#------------------------------------------------------------------------------------
#Definition of loss function (unconstrained) (without barrier) only LF predictions
def LossFunCentUnconstSolLF(x, h_NN, FixedPar, BlockVar, SigSlope, BarrWeight):
    
    import numpy as np
    
    # OverallInputN=12;
    # InputParLabels=["hw","gamma","vf","Nrev","WPLength","WPDiameter",
    #                 "OPWheelDiameter","CWheelDiameter","CwheelAngularVelocity",
    #                 "VolSpecificEnergy","EdgeComponentF","GritStiffness"]
    # xTot=np.empty(12)
       
    #BarrWeight (new parameter): Weight for the barrier definition
    
    for key, value in FixedPar.items():
        if key=="OmegaS":
              OmegaS=FixedPar[key]
        elif  key=="Nd":
              Nd=FixedPar[key]
        elif  key=="NoisAmpl":
              NoisAmpl=FixedPar[key]
        elif  key=="dg":
              dg=FixedPar[key]
        elif  key=="GbondR":
              GbondR=FixedPar[key]
        elif  key=="Rf":
              Rf=FixedPar[key]
        elif  key=="Eq":
              Eq=FixedPar[key]
        elif  key=="Kq":
              Kq=FixedPar[key]
        elif  key=="NLobes":
              NLobes=FixedPar[key]
        elif  key=="CatThr":
              CatThr=FixedPar[key]
        elif  key!="DataSet_X":              
              raise Exception("Undefined fixed parameters ",key)
    
    # Ordered meaning of input vector x entries:
    #0: hw
    #1: gamma
    #2: vf
    #3: Nrev
    #4: WPLength
    #5: WPDiameter
    #6: OPWheelDiameter
    #7: CWheelDiameter
    #8: CwheelAngularVelocity
    #9:VolSpecificEnergy
    #10:EdgeComponentF
    #11:GritStiffness
    
    xTot=[]
    ct=0      
    for key, value in BlockVar.items():
           if BlockVar[key]["Blocked"]:
               xTot=np.append(xTot,BlockVar[key]["value"])
           else:
               xTot=np.append(xTot,x[ct])
               ct+=1             
   
                
    Res=h_NN(xTot,OmegaS=OmegaS,Nd=Nd,NoisAmpl=NoisAmpl,dg=dg,GbondR=GbondR,
    Rf=Rf,Eq=Eq,Kq=Kq,NLobes=NLobes,CatThr=CatThr)
    #Definition of barrier
    Barr=lambda y: .5 * (np.tanh(SigSlope * (y-CatThr)) + 1)
    Cost=Res[0]+BarrWeight*Barr(Res[0])
    return Res[0], Cost
#------------------------------------------------------------------------------------  
#Definition of loss function penalizing distance fron dataset
def LossFunCentDist(x,NN,NNFacts,h_NN, FixedPar, BlockVar, SigSlope, BarrWeight, Lconst):
    
    import numpy as np
    
    # OverallInputN=12;
    # InputParLabels=["hw","gamma","vf","Nrev","WPLength","WPDiameter",
    #                 "OPWheelDiameter","CWheelDiameter","CwheelAngularVelocity",
    #                 "VolSpecificEnergy","EdgeComponentF","GritStiffness"]
    # xTot=np.empty(12)
       
    #BarrWeight (new parameter): Weight for the barrier definition
    
    for key, value in FixedPar.items():
        if key=="OmegaS":
              OmegaS=FixedPar[key]
        elif  key=="Nd":
              Nd=FixedPar[key]
        elif  key=="NoisAmpl":
              NoisAmpl=FixedPar[key]
        elif  key=="dg":
              dg=FixedPar[key]
        elif  key=="GbondR":
              GbondR=FixedPar[key]
        elif  key=="Rf":
              Rf=FixedPar[key]
        elif  key=="Eq":
              Eq=FixedPar[key]
        elif  key=="Kq":
              Kq=FixedPar[key]
        elif  key=="NLobes":
              NLobes=FixedPar[key]
        elif  key=="CatThr":
              CatThr=FixedPar[key]
        elif  key=="DataSet_X":
              DataSet_X=FixedPar[key]
        else:
              raise Exception("Undefined fixed parameters ",key)
    
    # Ordered meaning of input vector x entries:
    #0: hw
    #1: gamma
    #2: vf
    #3: Nrev
    #4: WPLength
    #5: WPDiameter
    #6: OPWheelDiameter
    #7: CWheelDiameter
    #8: CwheelAngularVelocity
    #9:VolSpecificEnergy
    #10:EdgeComponentF
    #11:GritStiffness
    
    xTot=[]
    ct=0      
    for key, value in BlockVar.items():
           if BlockVar[key]["Blocked"]:
               xTot=np.append(xTot,BlockVar[key]["value"])
           else:
               xTot=np.append(xTot,x[ct])
               ct+=1             
              
    Res=h_NN(xTot,NN,NNFacts,OmegaS=OmegaS,Nd=Nd,NoisAmpl=NoisAmpl,dg=dg,GbondR=GbondR,
    Rf=Rf,Eq=Eq,Kq=Kq,NLobes=NLobes,CatThr=CatThr)
     
    #Definition of barrier
    Barr=lambda y: .5 * (np.tanh(SigSlope * (y-CatThr)) + 1)
    DistFactor=np.amin(np.linalg.norm(DataSet_X-xTot,axis=1))
    
    Cost=Res[0]+BarrWeight*Barr(Res[0])+Lconst*DistFactor
    return Cost

#------------------------------------------------------------------------------------
def OptimCenterless(TrainedNN_model_filename, ScaleFactors_filename, FixedPar, BlockVar, OptPar):
    #Libraries
    import SimLF_functionsLib 
    import scipy.optimize as opt
    import numpy as np
    import pickle
    from tensorflow import keras
    import my_loss_def 
    
    NN=keras.models.load_model(TrainedNN_model_filename, custom_objects={'my_loss_fn7': my_loss_def.my_loss_fn7})
    f=open(ScaleFactors_filename,'rb')
    NNFacts=pickle.load(f)
    
    x0=[]; lb=[]; ub=[]   
        #Definition of Intial values and Bounds
    for key in BlockVar:
        #breakpoint()
        if not(BlockVar[key]["Blocked"]):
                x0=np.append(x0,BlockVar[key]["value"])
                lb=np.append(lb,BlockVar[key]["lb"])
                ub=np.append(ub,BlockVar[key]["ub"])
            
    Bounds=opt.Bounds(lb, ub, keep_feasible=False)
    if NN.input.shape[1]==len(BlockVar):
        H_Model=SimLF_functionsLib.h_NN2
    elif 'Nrev' in BlockVar:
        H_Model=SimLF_functionsLib.h_NN1
    else:
        H_Model=SimLF_functionsLib.h_NN1_v2
    
    #Sustituting a fixed profile to the noise value
    FixedPar["NoisAmpl"]=FixedPar["NoisAmpl"]*np.random.uniform(size=FixedPar["Nd"]);
   
    #h_NN1: Hybrid model (LF+NN) in SimLF_functionsLib library
    #ObjFun= lambda x: SimLF_functionsLib.LossFunCent(x,NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight'])
    ObjFun= lambda x: SimLF_functionsLib.LossFunCent(x,NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight'])               
    Solution=opt.minimize(ObjFun, x0, args=(), method='L-BFGS-B', tol=OptPar['tol'], bounds=Bounds, options={'disp':OptPar['Display'], 'gtol':OptPar['gradient tol'],
                                                                                                             'eps':OptPar['eps']})
    FinalRoundness=SimLF_functionsLib.LossFunCentUnconst(Solution['x'],NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight']) 
    ct=0
    SolDict={}
    for key in BlockVar:
        if not(BlockVar[key]["Blocked"]):
             SolDict[key]=Solution['x'][ct]
             ct+=1
    SolDict['FinalRoundness']=FinalRoundness
    SolDict['verbose']=Solution
    return SolDict

#------------------------------------------------------------------------------------  

def OptimCenterlessGlob(TrainedNN_model_filename, ScaleFactors_filename, FixedPar, BlockVar, OptPar):
    #Libraries
    import SimLF_functionsLib 
    import scipy.optimize as opt
    import numpy as np
    import pickle
    from tensorflow import keras
    import my_loss_def 
    
    NN=keras.models.load_model(TrainedNN_model_filename, custom_objects={'my_loss_fn7': my_loss_def.my_loss_fn7})
    f=open(ScaleFactors_filename,'rb')
    NNFacts=pickle.load(f)
    
    x0=[]; lb=[]; ub=[]; RangesTuple=()   
        #Definition of Intial values and Bounds
    for key in BlockVar:
        #breakpoint()
        if not(BlockVar[key]["Blocked"]):
                lb=BlockVar[key]["lb"]
                ub=BlockVar[key]["ub"]
                RangesTuple+=(slice(lb,ub,(ub-lb)/(OptPar['Ns'])),)
            
    Bounds=opt.Bounds(lb, ub, keep_feasible=False)
    if NN.input.shape[1]==len(BlockVar):
        H_Model=SimLF_functionsLib.h_NN2
    elif 'Nrev' in BlockVar:
        H_Model=SimLF_functionsLib.h_NN1
    else:
        H_Model=SimLF_functionsLib.h_NN1_v2
    
    #Sustituting a fixed profile to the noise value
    FixedPar["NoisAmpl"]=FixedPar["NoisAmpl"]*np.random.uniform(size=FixedPar["Nd"]);
   
    #h_NN1: Hybrid model (LF+NN) in SimLF_functionsLib library
    #ObjFun= lambda x: SimLF_functionsLib.LossFunCent(x,NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight'])
    ObjFun= lambda x: SimLF_functionsLib.LossFunCent(x,NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight'])  
    #Generation of the ranges
    Solution=opt.brute(ObjFun, RangesTuple, args=(), full_output=True, finish=None)
    FinalRoundness=SimLF_functionsLib.LossFunCentUnconst(Solution[0],NN,NNFacts, H_Model, FixedPar, BlockVar, OptPar['SigmaSlope'], OptPar['BarrierWeight'])
    ct=0
    SolDict={}
    for key in BlockVar:
        if not(BlockVar[key]["Blocked"]):
             SolDict[key]=Solution[0][ct]
             ct+=1
    SolDict['FinalRoundness']=FinalRoundness
    SolDict['verbose']=Solution
    return SolDict

#------------------------------------------------------------------------------------  

def SMBounds(x,gammaSM,epsilonSM,xtilde,ytilde):
    #Set Membership bounds computation
    #x: input vector to be boundend (n,)
    #gammaSM: Lipschitz constant
    #epsilonSM: observation uncertainty
    #xtilde: N observations independent variables (Nxn) array
    #ytilde: N observations dependent variables (N,)
    
    import numpy as np
    
    N,n = xtilde.shape
    x_matr=np.zeros((N,n))
    
    for k in range(n):
        x_matr[:,k]=np.ones((N,1))*x[k]
    
    gamma_norm_vec=gammaSM*np.linalg.norm((x_matr-xtilde),axis=1)
    UB=min(ytilde+gamma_norm_vec+epsilonSM/2);
    LB=max(ytilde-gamma_norm_vec-epsilonSM/2);
    return LB, UB

#------------------------------------------------------------------------------------  

def NN_InputsNumber(TrainedNN_model_filename):
    from tensorflow import keras
    import my_loss_def
    
    NN=keras.models.load_model(TrainedNN_model_filename, custom_objects={'my_loss_fn7': my_loss_def.my_loss_fn7})
    return NN.input.shape[1]
    