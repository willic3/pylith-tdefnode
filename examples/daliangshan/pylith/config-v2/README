This directory contains the .cfg files and some shell scripts needed to
generate PyLith Green's functions. There are .cfg and shell scripts for
each type of GF needed. Note that it is not necessary to generate normal
GF for any fault other than Anning-Zem, since this is the only fault being
used in 3D mode within TDefnode.

run_az_ll_homog_workstation.sh: Left-lateral Green's functions.
run_az_ud_homog_workstation.sh: Updip Green's functions.
run_az_nm_homog_workstation.sh: Fault-normal Green's functions (not needed
                                for all faults).

Note that there are versions of the scripts available for both workstations
and clusters. The cluster scripts are just examples, and would have to be
modified for the particular cluster/queue system being used. The scripts
are presently set up for the PBS system.

If running on a workstation, it is generally a good idea to save stderr and
stdout to a log file. For example, under bash, you would do something like:

./run_az_ll_homog_workstation.sh > run_az_ll_homog_workstation.log 2>&1 &

It is unlikely you would be able to run more than one job at a time, and each
one will take some time to run.

Note that I have not created all of the workstation scripts, since I am running
everything on a cluster.
