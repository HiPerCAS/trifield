* ============================================================================
* sdevice2_des.cmd — Vth extraction via backward Vg sweep
*
* Second sdevice run (TEMPLATES[2] in src/simulation/simulation.py). Its only
* purpose is the threshold-voltage current curve consumed by
* pltfile.PLTFile.find_vth("g_outer_voltage", "d_total_current", 1.0). The
* resulting node2_des.tdr is NOT consumed by the graph builder — only the
* .plt file is parsed — so the Plot{} block below is stripped to the bare
* minimum needed for current logging.
*
* Pipeline:
*   1. Warmup: ramp Vd 0 -> 12 V, then Vg 0 -> 5 V (.plt log muted)
*   2. Measurement: backward Vg 5 -> 0 V (.plt log re-enabled)
*
* find_vth() interpolates Vg at the point where d_total_current crosses
* 1 uA, so it MUST see only the backward sweep. The warmup ramps are
* redirected to warmup_des.plt via NewCurrentFile= "warmup_des", then
* NewCurrentFile= "" before the backward sweep restores the default
* node2_des.plt from the File{} block.
*
* @AreaFactor@ is a literal float computed by simulation.calculate_area_factor.
* ============================================================================

File{
	Grid     = "node_msh.tdr"
	Parameter= "parameter.par"
	Plot     = "node2_des.tdr"
	Current  = "node2_des.plt"
	Output   = "node2_des.log"
}

Electrode{
	{ Name="S" Voltage= 0.0 }
	{ Name="D" Voltage= 0.0 }
	{ Name="G" Voltage= 0.0 WorkFunction= 5.2 }
}

Physics{
	AreaFactor= @AreaFactor@
	Temperature= 300
	*Thermionic
	*IncompleteIonization
	*Piezoelectric_Polarization(strain)
	Mobility(
		DopingDependence
		HighFieldSaturation
	)
	EffectiveIntrinsicDensity(NoBandGapNarrowing)
	Fermi
	Recombination(
		SRH
	)
}

Physics(Region="Buffer"){
	Mobility(
		DopingDependence
		HighFieldSaturation
	)
}

Physics(MaterialInterface="GaN/AlGaN"){
	Piezoelectric_Polarization(Activation= 1.0)
	Traps(
		(Donor Level fromMidbandgap EnergyMid= 1.25 Conc= 3e13 eXsection= 1e-14 hXsection= 1e-14)
	)
}

Physics(MaterialInterface="AlGaN/HfO2"){
	Traps(
		(Donor Level Conc= 5e13 EnergyMid= 0.6 FromMidBandGap)
	)
}

Physics(MaterialInterface="GaN/Nitride"){
	Traps(
		(Donor Level Conc= 5e13 EnergyMid= 0.6 FromMidBandGap)
	)
}

Physics(MaterialInterface="AlGaN/Nitride"){
	Traps(
		(Donor Level Conc= 5e13 EnergyMid= 0.6 FromMidBandGap)
	)
}

* ----- Solver options -----
Math{
	*Number_of_Threads= 4
	*Wallclock
	Transient= BE
	Traps(Damping= 0)
	*CurrentPlot (IntegrationUnit = cm)*
	*ExtendedPrecision
	RHSMin= 1e-15
	Extrapolate
	Derivatives
	* Avalderivatives
	RelErrControl
	Digits= 6
	ErRef(electron)= 1e8
	ErRef(hole)=     1e8
	Method= Blocked
	*Notdamped= 50
	Iterations= 1000
	*Newdiscretization
	DirectCurrent
	Parallel= 1
	*-VoronoiFaceBoxMethod
	NaturalBoxMethod
}

Plot{
	* Full field set: the transfer-curve snapshots (snap2_vgNNNN_vd12, saved
	* during the backward Vg sweep below) feed the reference FoM extractor
	* (src/gnn/data/reference_metrics.py), which integrates TotalCurrentDensity
	* and may also use the potentials.
	Potential eQuasiFermi hQuasiFermi
	TotalCurrent/Vector eCurrent/Vector hCurrent/Vector CurrentPotential
}

Solve{
	* --- Warmup ramps (current redirected to warmup_des.plt) ---
	* Redirect the current log so that node2_des.plt only contains the
	* meaningful backward Vg sweep (parsed by pltfile.PLTFile.find_vth).
	NewCurrentFile= "warmup_des"
	Coupled(Iterations= 1000 LineSearchDamping= 1e-4){ Poisson }
	Coupled(Iterations= 1000 LineSearchDamping= 1e-4){ Poisson Electron Hole }
	Coupled                                          { Poisson Electron Hole }

	Quasistationary(
		InitialStep= 0.05 MinStep= 1e-12 MaxStep= 0.05
		Goal{ Name="D" Voltage= 12 }
	){ Coupled{ Poisson Electron Hole } }

	Quasistationary(
		InitialStep= 0.05 MinStep= 1e-12 MaxStep= 0.05
		Goal{ Name="G" Voltage= 5 }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg5000_vd12")

	* --- Measurement: backward Vg 5 -> 0 V (.plt re-enabled) ---
	* find_vth() parses this backward sweep (continuous .plt) for Vth. It is
	* stepped into 0.5 V intervals so a TDR field snapshot (snap2_vgNNNN_vd12,
	* NNNN = Vg in mV) is saved at each Vg: the Id-Vg transfer curve at Vd=12
	* that the reference extractor turns into vth (1 uA crossing) and gm (max
	* dId/dVg). Internal sub-stepping (MaxStep 0.05) keeps the .plt fine.
	NewCurrentFile= ""
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 4.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg4500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 4.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg4000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 3.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg3500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 3.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg3000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 2.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg2500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 2.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg2000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 1.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg1500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 1.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg1000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 0.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg0500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= 0.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vg0000_vd12")
	* --- Continue Vg 0 -> -6 V so D-mode (normally-on) devices have their
	* 1 uA crossing SAMPLED, not extrapolated (snap2_vgnNNNN = negative Vg). ---
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -0.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn0500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -1.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn1000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -1.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn1500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -2.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn2000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -2.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn2500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -3.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn3000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -3.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn3500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -4.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn4000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -4.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn4500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -5.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn5000_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -5.5 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn5500_vd12")
	Quasistationary( InitialStep= 0.02 MinStep= 1e-12 MaxStep= 0.05 Goal{ Name="G" Voltage= -6.0 } ){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap2_vgn6000_vd12")
}
