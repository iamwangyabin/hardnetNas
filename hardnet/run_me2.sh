#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/"
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"

mkdir -p "$DATASETS"
mkdir -p "$DATALOGS"

( # Run the code
    cd "$RUNPATH"
    python ./HardNet.py --fliprot=False --experiment-name=liberty_train/ $@ | tee -a "$DATALOGS/log_HardNet_Lib.log"
    #python ./HardNet.py --fliprot=True --experiment-name=liberty_train_with_aug/  $@ | tee -a "$DATALOGS/log_HardNetPlus_Lib.log"
)

