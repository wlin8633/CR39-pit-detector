# Local benchmark data

This directory is a landing area for private calibration inputs. Raw contents
are ignored by Git. Either copy the Geant4 and experimental files here or keep
them in their experiment archives and point `../local_paths.json` to their
absolute paths.

Recommended local-only layout:

```text
data/
  raw/geant4/       eDep and trackLen files
  results/          pit_size_vs_energy.sav and related detection outputs
  states/           pit-detection state files used for the calibration
```
