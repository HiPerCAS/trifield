# Sentaurus TCAD files

Everything under `trifield/tcad/` needs a licensed Synopsys Sentaurus
installation (tested with V-2024.03). Put `sde`, `sdevice` and `tdx` on your
`PATH`, or point `SENTAURUS_BIN` at the installation's `bin/` directory.

## What is included

| File | Role | Origin |
|---|---|---|
| `templates/sde_dvs.cmd` | Sentaurus Structure Editor script: multi-fin GaN/AlGaN tri-gate structure, contacts, meshing. Parameters are substituted at `@name@` tokens. | Written by the authors |
| `templates/sdevice1_des.cmd` | Sentaurus Device run 1: Vg ramp, Vd ramp to 12 V, one TDR field snapshot per bias corner (the nine training snapshots) and the Id-Vd current file. | Written by the authors |
| `templates/sdevice2_des.cmd` | Sentaurus Device run 2: backward Vg sweep for the threshold voltage. | Written by the authors |
| `simulation.py` | Renders the templates, runs the tools, converts snapshots to graphs. | Written by the authors |
| `campaign.py` | Parallel design sampling + simulation (training data or mesh-only candidate pools). | Written by the authors |

## What is *not* included: `parameter.par`

The device runs read material models from `parameter.par`. The campaign used
the GaN, AlN and AlGaN parameter sets of the Synopsys Sentaurus material
database, which are proprietary to Synopsys and may not be redistributed.
Assemble the file from your own installation:

```bash
DB=$STROOT/tcad/$STRELEASE/lib/sdevice/MaterialDB      # adjust to your install
cat $DB/GaN.par $DB/AlN.par $DB/AlGaN.par > parameter.par
```

and pass it with `--par parameter.par`. The templates reference the material
names `GaN`, `AlGaN`, `HfO2`, `Gold` and `TiN`; dielectrics and metals use
the Sentaurus defaults and need no entry. The physics selections (Fermi
statistics, doping- and field-dependent mobility, SRH recombination,
thermionic emission and polarisation at the GaN/AlGaN interface) live in the
`sdevice*_des.cmd` templates, not in the parameter file.

## Running one design

```bash
python -m trifield.tcad.simulation --design examples/design.json --par parameter.par \
    --work_dir runs --out_dir data/mine            # full solve: FoMs + 9 graphs
python -m trifield.tcad.simulation --design examples/design.json --par parameter.par \
    --out_dir runs/candidates --mesh_only          # sde only: 9 mesh-only graphs
```

## Converting your own TDR snapshots

If you already have a finished Sentaurus run laid out like the templates
(`node_msh.tdr`, `snap_vg*_vd*_des.tdr`, `node1_des.tdr`):

```python
from trifield.tcad import build_snapshot_graphs
build_snapshot_graphs("runs/sim_000001", params, "data/mine", design_name="my_design")
```

`tdx` converts each TDR to the DF-ISE text format; `trifield.data.tdr_extract`
parses it (no Synopsys Python API needed).
