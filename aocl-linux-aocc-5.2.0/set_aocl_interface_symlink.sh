#!/bin/bash
################################################################################
# Copyright(c) 2021-2023 Advanced Micro Devices
# Name:     set_aocl_interface_symlink.sh
# Purpose: Helps to switch over AOCL package installation from LP64 to ILP64 or
#          vice versa
# Usage: bash set_aocl_interface_symlink.sh <lp64 / ilp64>
################################################################################

opt=$1;

# Condition to check current working directory before sourcing set_aocl_interface_symlink.sh
if [[ $PWD = *"/usr"* || $PWD = *"/root" ]]; then
    if [[ $PWD = *"/root" ]]; then
        printf "##############################################\n"
        printf "User should be in AOCL installed path. user in /root/* directory, cannot source set_aocl_interface_symlink.sh in this directory. Exiting.\n"
        printf "##############################################\n"
        exit 1
    elif [[ $PWD = *"/usr"* ]]; then
        printf "##############################################\n"
        printf "User should be in AOCL installed path. user in /usr/* directory, cannot source set_aocl_interface_symlink.sh in this directory. Exiting.\n"
        printf "##############################################\n"
        exit 1
    fi
fi

printf "Setting $opt as default...\n";

mydir=$(dirname "$0")
echo "Installed directory is $mydir"
cd "$mydir"


# if symlink exists, delete it, and..
if [ -L ./lib ]; then
    rm -rf ./lib;
    rm -rf ./include;
fi

# ... based on the option, create it
if [ "${opt}" = "lp64" ]; then
    ln -s ./lib_LP64 ./lib;
    ln -s ./include_LP64 ./include;
elif [ "${opt}" = "ilp64" ]; then
    ln -s ./lib_ILP64 ./lib;
    ln -s ./include_ILP64 ./include;
else
    ln -s ./lib_LP64 ./lib;
    ln -s ./include_LP64 ./include;
fi

printf "Setting $opt as default is completed\n";
