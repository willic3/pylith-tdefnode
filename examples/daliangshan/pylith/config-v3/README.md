### This directory contains the files needed to generate PyLith Green's functions using PyLith v3+.
The only contents are the `.cfg` files. There are `.cfg` file for each type of GF needed. Note that it is not necessary to generate fault-normal GF for any fault other than Anninghe-Zemuhe, since this is the only fault being used in 3D mode within TDefnode. As an example, the following commands will generate all homogeneous GF for the Anninghe-Zemuhe fault:

```
pylith gf_anning_zem_ll_heter.cfg --nodes=10
# This will generate the left-lateral Green's functions.
pylith gf_anning_zem_ud_heter.cfg --nodes=10
# This will generate the updip Green's functions.
pylith gf_anning_zem_nm_heter.cfg --nodes=10
# This will generate the fault-normal Green's functions (not needed for all faults).
```

If running on a workstation, it is generally a good idea to save stderr and stdout to a log file. For example, under bash, you would do something like:

```
pylith gf_anning_zem_ll_heter.cfg --nodes=10 > run_az_ll_homog_workstation.log 2>&1 &
```

It is unlikely you would be able to run more than one or two jobs at a time if running on a workstation, and each one will take some time to run.

### NOTE: For the current version of PyLith (v 3.0), running on more than 3 or 4 cores will be very slow or will not converge.
