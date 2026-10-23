* ============================================================================
* sdevice1_des.cmd — Id-Vd corner sweep + TDR snapshots grid
*
* First sdevice run (TEMPLATES[1] in src/simulation/simulation.py). Ramps Vg
* 0 -> @Vg11@ then Vd 0 -> 12 V, writing one TDR snapshot per intermediate
* (Vg, Vd) corner. The final solution (Vg=@Vg11@, Vd=12) is plotted to
* node1_des.tdr; src/simulation/pltfile.py parses node1_des.plt for Id_max.
*
* Snapshot grid: 5 Vg points (20/40/60/80/100% of @Vg11@) at Vd=0, then 4 Vd
* points (3/6/9/12 V) at full Vg. Snapshots are consumed by Simulation._build_graphs()
* to produce 9 PyG .pt files per simulation.
*
* The Plot{} block below is minimal — it only lists the fields required by
* src/gnn/data/graph_builder.py targets (Potential, eQuasiFermi, hQuasiFermi)
* plus the vector/current fields needed by src/simulation/pltfile.py. Mesh
* fields (Doping, xMoleFraction) come from node_msh.tdr at mesh-build time.
*
* @AreaFactor@ is a literal float computed by simulation.calculate_area_factor:
*   1e6 / (8*(GaN+AlGaN) + Fw + Sfin)
* ============================================================================

File{
	Grid     = "node_msh.tdr"
	Parameter= "parameter.par"
	Plot     = "node1_des.tdr"
	Current  = "node1_des.plt"
	Output   = "node1_des.log"
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

* ----- Solver options -----
Math{
	*Number_of_Threads= 4
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
	Iterations= 200
	*Newdiscretization
	DirectCurrent
	Parallel= 1
	*-VoronoiFaceBoxMethod
	NaturalBoxMethod
}

Plot{
	* --- Required by graph_builder targets (data.y columns 0..2) ---
	Potential                    * -> ElectrostaticPotential
	eQuasiFermi hQuasiFermi      * -> eQuasiFermiPotential, hQuasiFermiPotential

	* --- Required by pltfile.py Current parser (node1_des.plt -> Id_max) ---
	TotalCurrent/Vector eCurrent/Vector hCurrent/Vector CurrentPotential
}

Solve{
	NewCurrentFile= ""
	Coupled(Iterations= 1000){ Poisson }
	Coupled(Iterations= 1000){ Poisson Electron Hole }
	Coupled                  { Poisson Electron Hole }

	* --- Vg sweep 0 -> @Vg11@ at Vd=0 (TDR snapshots at 20/40/60/80/100%) ---
	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="G" Voltage= @<Vg11*0.2>@ }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg02_vd00")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="G" Voltage= @<Vg11*0.4>@ }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg04_vd00")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="G" Voltage= @<Vg11*0.6>@ }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg06_vd00")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="G" Voltage= @<Vg11*0.8>@ }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg08_vd00")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="G" Voltage= @Vg11@ }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg10_vd00")

	* --- Vd sweep 0 -> 12 V at full Vg (TDR snapshots at 3/6/9 V) ---
	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="D" Voltage= 3 }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg10_vd03")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="D" Voltage= 6 }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg10_vd06")

	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="D" Voltage= 9 }
	){ Coupled{ Poisson Electron Hole } }
	Plot(FilePrefix= "snap_vg10_vd09")

	* --- Final corner: Vg=@Vg11@, Vd=12 -> written to node1_des.tdr ---
	Quasistationary(
		InitialStep= 0.001 MinStep= 1e-6 MaxStep= 0.02
		Goal{ Name="D" Voltage= 12 }
	){ Coupled{ Poisson Electron Hole } }
}
