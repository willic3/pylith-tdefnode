### This directory contains the files needed to generate PyLith Green's functions using PyLith v2.x.
The main contents are the `.cfg` files. Shell scripts are also included to demonstrate the commands necessary to run the code. These may be modified as desired. There are `.cfg` and shell scripts for each type of GF needed. Note that it is not necessary to generate fault-normal GF for any fault other than Anninghe-Zemuhe, since this is the only fault being used in 3D mode within TDefnode. As an example, the following commands will generate all homogeneous GF for the Anninghe-Zemuhe fault:

```
./run_az_ll_homog_workstation.sh
# This will generate the left-lateral Green's functions.
./run_az_ud_homog_workstation.sh
# This will generate the updip Green's functions.
./run_az_nm_homog_workstation.sh
# This will generate the fault-normal Green's functions (not needed for all faults).
```

Note that there are versions of the scripts available for both workstations and clusters. The cluster scripts are just examples, and would have to be modified for the particular cluster/queue system being used. The scripts are presently set up for the PBS system.

If running on a workstation, it is generally a good idea to save stderr and stdout to a log file. For example, under bash, you would do something like:

```
./run_az_ll_homog_workstation.sh > run_az_ll_homog_workstation.log 2>&1 &
```

It is unlikely you would be able to run more than one or two jobs at a time if running on a workstation, and each one will take some time to run. Note that I have not created all of the workstation scripts, since I ran everything on a cluster.
