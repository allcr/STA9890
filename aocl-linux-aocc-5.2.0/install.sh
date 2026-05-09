#!/bin/bash

#  install.sh - installs the AOCL Libraries package

#===----------------------------------------------------------------------===#
#==== Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Developed by: Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimers.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimers in the documentation
# and/or other materials provided with the distribution.
#
# Neither the names of Advanced Micro Devices, Inc., nor the names of its
# contributors may be used to endorse or promote products derived from this
# Software without specific prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
# THE SOFTWARE.
#===----------------------------------------------------------------------===#

#Set initial variables
CUSTOM_PATH=
SELECTED_LIBS=
MASTER_INSTALL=
LIBMAP=(blis libflame rng fftw securerng libm scalapack sparse libmem crypto compression utils da dlp fftz) #List of AOCL Libraries
SELECTED_INT_TYPE=

error=0
rc=
ow=1 #overwrite
dir_name=
message=
lib=

#AMD Specific Variables
HOME_LOC=$HOME
AMD_LIBS_ROOT="$HOME/aocl"
# Default library set uses fftw (not fftz).
# Users can still install fftz explicitly using: -l fftz
AMD_LIBS=(blis libflame rng fftw securerng libm scalapack sparse libmem crypto compression utils da dlp)
AMD_LIBS_LOG_FILE="amd-libs.log"
AMD_LIBS_CFG_FILE="amd-libs.cfg"
AMD_LIBS_REL="5.2.0"
AMD_LIBM_REL="5.2.0"
AMD_REL="5.2.0/aocc"
export MODULE_SCRIPT="${AMD_REL}/aocl-linux-aocc-${AMD_LIBS_REL}_module"
export G_LOC_LIB_PATH=""

#Storing output to log
rm -rf $AMD_LIBS_LOG_FILE #cleaning config and log file

help()
{
   #Display Help
   echo
    echo "Usage  : ./install.sh [-h] [-l <libname> ...] [-t <custom_dir_path>] [-i <lp64|ilp64>]"
   echo "Example: ./install.sh -l libm -t <custom_dir_path>"
    echo "Options:"
    echo "  -h     Print this help message"
   echo "-t     Custom target directory to install libraries"
    echo "  -l     One or more libraries to install (space separated)."
    echo "  -i     Choose integer interface (lp64 or ilp64) for default symlink"
   echo
    echo "Default behavior:"
    echo "  * fftw is installed by default."
    echo "  * fftz will ONLY be installed if explicitly requested via -l fftz."
    echo "  * fftw and fftz cannot be installed together."
    echo "    If you try to install both, you will get:"
    echo "    Error: fftw and fftz cannot be installed simultaneously. Install only one of them, and when installing each individually, choose a different target directory."
    echo
    echo "Examples:"
    echo "  ./install.sh                        # installs default set (includes fftw)"
    echo "  ./install.sh -l fftw fftz           # Error: cannot install both fftw and fftz together"
    echo "  ./install.sh -l blis libflame fftz -i ilp64"
   echo
}

################################################################################
#Quit nicely with messages as appropriate                                      #
################################################################################
quit()
{
   if [ $error != 0 ]
   then
      echo "Program terminated with error ID $error" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
      rc=$error
   else
      if [ $error = 0 ]
      then
         echo "Program terminated normally" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
         rc=0
      fi
   fi

   exit $rc
}

################################################################################
#Install in custom directory                                                   #
################################################################################
#Get the path from user
custom_dir()
{
    #read the user given path and check if its already exists or not
    if [[ ${CUSTOM_PATH:0:1} == "/" ]]; then
        #its an absolute path
        AMD_LIBS_ROOT=${CUSTOM_PATH}
    elif  [[ ${CUSTOM_PATH:0:2} == "./" ]]; then
        AMD_LIBS_ROOT=`pwd`
    else
        AMD_LIBS_ROOT=`pwd`/${CUSTOM_PATH}
    fi
}

################################################################################
#Install specific libraries                                                    #
################################################################################
#Get the list of libraries user wants to install
custom_libs()
{
    AMD_LIBS=()
    #need to split the string of inputs given and install them
    LOWER_SELECTED_LIBS="$(echo ${SELECTED_LIBS[@]} | tr '[A-Z]' '[a-z]')"
    IFS=' '
    read -ra ADDR <<< "$LOWER_SELECTED_LIBS"
    # Check for fftw and fftz together
    found_fftw=0
    found_fftz=0
    for i in "${ADDR[@]}"; do
        if [[ "$i" == "fftw" ]]; then
            found_fftw=1
        fi
        if [[ "$i" == "fftz" ]]; then
            found_fftz=1
        fi
    done
    if [[ $found_fftw -eq 1 && $found_fftz -eq 1 ]]; then
        echo "Error: fftw and fftz cannot be installed simultaneously. Install only one of them, and when installing each individually, choose a different target directory." 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        error=3
        quit $error
    fi
    for i in "${ADDR[@]}"; do # access each element of array
        lib_in=\\b$i\\b
        if [[ ${LIBMAP[*]} =~ $lib_in  ]]; then
            AMD_LIBS+=($i)
        else
            echo "$i is not a library, Please use ./install.sh -h to know the usage" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
            error=2
            quit $error
        fi
    done
}

################################################################################
#Process the input options                                                     #
################################################################################
#Get options
OPTSPEC="t:T:l:L:i:I:h:H"
while getopts "$OPTSPEC" options; do
   case "${options}" in
        [Hh]* )
            #print help
            help;
            exit;
        ;;
        [Tt]* ) # install libraries to specific directory
            CUSTOM_PATH="${OPTARG}";
            custom_dir;
        ;;
        [Ll]* )  # Selected libraries
            SELECTED_LIBS=("$OPTARG")
            until [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
                SELECTED_LIBS+=($(eval "echo \${$OPTIND}"))
                OPTIND=$((OPTIND + 1))
            done
            custom_libs;
        ;;
        [Ii]* )
            SELECTED_INT_TYPE="${OPTARG}";
        ;;
        * ) # incorrect option
            echo "usage: ./install.sh [-h] [-l <libname>] [-t <custom_dir_path>] [-i <lp64/ilp64>]"  2>&1 | tee -a $AMD_LIBS_LOG_FILE
            exit ;;
   esac
done

################################################################################
#Gets simple (Y)es (N)o (Q)uit response from user. Loops until                 #
#one of those responses is detected.                                           #
################################################################################
message_temp="WARNING!!! Do you wish to continue to overwrite the package --> "
ynq()
{
   # loop until we get a good answer and break out
   while true
   do
      # Print the message
      echo -n "$message (yes | no | quit):"
      # Now get some input
      read input
      # Test the input
      if [ -z $input ]
      then
         # Invalid input - null
         echo "INPUT ERROR: Must be yes or no or quit in lowercase. Please try again." 2>&1 | tee -a $AMD_LIBS_LOG_FILE
      elif [ $input = "yes" ] || [ $input = "y" ]
      then
         ow=1
         break
      elif [ $input = "no" ] || [ $input = "n" ]
      then
         ow=0
         break
      elif [ $input = "Help" ] || [ $input = "h" ]
      then
         help
         break
      elif [ $input = "q" ] || [ $input = "quit" ]
      then
         quit
      else
         # Invalid input
         echo "INPUT ERROR: Must be y or n or q in lowercase. Please try again."  2>&1 | tee -a $AMD_LIBS_LOG_FILE
      fi
   done
}

##############################################################################
#Create directories and install libraries                                    #
##############################################################################
#Implementation of installtion

InitModuleFile()
{
retVal=0

       echo '#%Module1.0#####################################################################'>$MODULE_SCRIPT
       echo " " >>$MODULE_SCRIPT
       echo 'proc ModulesHelp { } {' >>$MODULE_SCRIPT
       echo '    global version AOCLhome' >>$MODULE_SCRIPT
       echo '    puts stderr "\tAOCL \n"' >>$MODULE_SCRIPT
       echo '    puts stderr "\tloads AMD Optimizing CPU Libraries (AOCL) setup \n"' >>$MODULE_SCRIPT
       echo '}' >>$MODULE_SCRIPT
       echo " " >>$MODULE_SCRIPT
       echo 'module-whatis "loads AOCL Libraries setup "' >>$MODULE_SCRIPT
       echo " " >>$MODULE_SCRIPT
       retVal=$?

       return $retVal
}

AddToLOC()
{
    fldr=$1
    if [[ $G_LOC_LIB_PATH != "" ]];then
        G_LOC_LIB_PATH="$fldr:$G_LOC_LIB_PATH"
    else
        G_LOC_LIB_PATH="$fldr"
    fi
}

HandleLibEnvSetting()
{
    if [[ -d '/usr/lib32' ]];then
        AddToLOC '/usr/lib32'
    fi

    if [[ -d '/usr/lib' ]];then
        AddToLOC '/usr/lib'
    fi

    if [[ -d '/usr/lib/x86_64-linux-gnu' ]];then
        AddToLOC '/usr/lib/x86_64-linux-gnu'
    fi

    if [[ -d '/usr/lib64' ]];then
        AddToLOC '/usr/lib64'
    fi

    if [[ $G_LOC_LIB_PATH != "" ]];then
        ModuleFile "prepend-path    LIBRARY_PATH    ${G_LOC_LIB_PATH}"
        ModuleFile "prepend-path    LD_LIBRARY_PATH    ${G_LOC_LIB_PATH}"
    fi
}

ModuleFile()
{
msg=$1

echo $msg >>$MODULE_SCRIPT

}

do_cmd()
{
    eval $1
    echo -e $1 >> $AMD_LIBS_CFG_FILE
}

init_env()
{
    libs_root="$AMD_LIBS_ROOT/$AMD_REL"
    libs_dir="$libs_root"
    libs_dir_lp64="$libs_dir/lib_LP64"
    libs_dir_ilp64="$libs_dir/lib_ILP64"
    include_dir_lp64="$libs_root/include_LP64";
    include_dir_ilp64="$libs_root/include_ILP64";

    #check if directories are avaible
    if [ ! -d "$libs_root" ]; then
        #create directories
        eval "mkdir -p $libs_root"
        if [[ $? -ne 0 ]]; then
            echo "Unable to create Directory $libs_root. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
            error=13
            quit $error
        fi
    eval "mkdir -p $libs_dir"
        if [[ $? -ne 0 ]]; then
            echo "Unable to create Directory $libs_dir. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
            error=13
            quit $error
        fi

    eval "mkdir -p $libs_dir_lp64"
        if [[ $? -ne 0 ]]; then
            echo "Unable to create Directory $libs_dir_lp64. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
            error=13
            quit $error
        fi

    eval "mkdir -p $libs_dir_ilp64"
        if [[ $? -ne 0 ]]; then
            echo "Unable to create Directory $libs_dir_ilp64. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
            error=13
            quit $error
        fi

    #check if include dirs (LP64, ILP64 exists)
    eval "mkdir $include_dir_lp64";
    if [[ $? -ne 0 ]]; then
        echo "Unable to create Directory $include_dir_lp64. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        error 13
        quit $error
    fi

    eval "mkdir $include_dir_ilp64";
    if [[ $? -ne 0 ]]; then
        echo "Unable to create Directory $include_dir_ilp64. Try with sudo " 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        error 13
        quit $error
    fi

    #check if the libs path is same or different
    if [[ -f $AMD_LIBS_CFG_FILE ]]; then
        line1=$(head -n 1 $AMD_LIBS_CFG_FILE)
        if [[ $line1 != *"${libs_dir}"* ]]; then
            echo "You have choosen different directory from previous installation, Only current target directory will be added to the config file"
        fi
    fi

    for lib in ${AMD_LIBS[@]}; do
        untar_libs
    done

    else
       #read input libs to be installed, ask user if he wants to overwrite
       #yes means, do overwrite
       for lib in ${AMD_LIBS[@]}; do
           dir_name="$libs_root/amd-$lib"
           if [ -d $dir_name ]; then
               message="$message_temp $lib ?"
               ynq
           fi
           untar_libs
       done
    fi

    if [[ -f $AMD_LIBS_CFG_FILE ]]; then
        rm $AMD_LIBS_CFG_FILE
    fi

    #set env variables if libs dir is not empty
    if [ -d "${libs_dir}" ]; then
        MODULE_SCRIPT="${libs_dir}/aocl-linux-aocc-${AMD_LIBS_REL}_module"
        InitModuleFile
        HandleLibEnvSetting
        ModuleFile "set    AOCLhome    ${libs_root}"
        ModuleFile "prepend-path    AOCLhome    \$AOCLhome"
        ModuleFile "prepend-path    AOCL_ROOT    \$AOCLhome"
        ModuleFile "prepend-path    LIBRARY_PATH    \$AOCLhome/lib"
        ModuleFile "prepend-path    LD_LIBRARY_PATH    \$AOCLhome/lib"

        #updating environment variables
        do_cmd "export AOCL_ROOT=${libs_root};"
        do_cmd "export C_INCLUDE_PATH=${libs_dir}/include:\$C_INCLUDE_PATH"
        ModuleFile "prepend-path    C_INCLUDE_PATH     \$AOCLhome/include"
        do_cmd "export CPLUS_INCLUDE_PATH=${libs_dir}/include:\$CPLUS_INCLUDE_PATH"
        ModuleFile "prepend-path    CPLUS_INCLUDE_PATH    \$AOCLhome/include"
        do_cmd "export LD_LIBRARY_PATH=${libs_dir}/lib:\$LD_LIBRARY_PATH"
        ModuleFile "prepend-path    LD_LIBRARY_PATH    \$AOCLhome/lib"
        do_cmd "export LIBRARY_PATH=${libs_dir}/lib:\$LIBRARY_PATH"
        ModuleFile "prepend-path    LIBRARY_PATH    \$AOCLhome/lib"

        cp -rv $AMD_LIBS_CFG_FILE $libs_root/$AMD_LIBS_CFG_FILE
	cp -r set_aocl_interface_symlink.sh $libs_root/
    fi

}

untar_libs()
{
    pkg=""
    error=0

    #installations
    if [ "$lib" = "libm" ]; then
        pkg="aocl-libm-linux-aocc-$AMD_LIBM_REL.tar.gz"
    else
        pkg="aocl-$lib-linux-aocc-$AMD_LIBS_REL.tar.gz"
    fi

    if [[ ! -f $pkg ]];then
        echo "Error: No package found for ${pkg}" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    fi

    if [[ -d $dir_name ]] && [[ "$ow" -eq 1 ]]; then
        echo -e Untarring $lib 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        echo "writing the library: $lib" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        tar -xvzf ${pkg} -C $AMD_LIBS_ROOT/$AMD_REL 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    elif [[ -d $dir_name ]] && [[ "$ow" -eq 0 ]]; then
        echo "Library already exists, not overwriting" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    else
        #directory will be created and links will setup
        echo -e Untarring $lib 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        echo "writing the library: $lib" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
        tar -xvzf ${pkg} -C $AMD_LIBS_ROOT/$AMD_REL 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    fi

    master_install
}

master_install ()
{
    libs_root="$AMD_LIBS_ROOT/$AMD_REL"
    libs_dir="$libs_root"
    # extra inc directories
    include_dir_lp64="$libs_root/include_LP64";
    include_dir_ilp64="$libs_root/include_ILP64";

    # Helper: ensure fftz symlinks use relative targets before copying
    relink_fftz_symlinks()
    {
        local base="$AMD_LIBS_ROOT/$AMD_REL/amd-fftz"
        local inc_dir="$base/include"
        local lib_dir="$base/lib"
        local w_inc="$base/wrappers/fftw/include"
        local w_lib="$base/wrappers/fftw/lib"

        # Recreate include symlinks with relative paths if source exists
        if [ -d "$inc_dir" ] && [ -d "$w_inc" ]; then
            # fftw3.h -> ../wrappers/fftw/include/fftw_wrapper.h
            if [ -L "$inc_dir/fftw3.h" ] || [ -f "$inc_dir/fftw3.h" ]; then
                rm -f "$inc_dir/fftw3.h"
            fi
            ln -s "../amd-fftz/wrappers/fftw/include/fftw_wrapper.h" "$inc_dir/fftw3.h" 2>/dev/null || true

            # fftw3.f -> ../wrappers/fftw/include/fftw_wrapper.f
            if [ -L "$inc_dir/fftw3.f" ] || [ -f "$inc_dir/fftw3.f" ]; then
                rm -f "$inc_dir/fftw3.f"
            fi
            ln -s "../amd-fftz/wrappers/fftw/include/fftw_wrapper.f" "$inc_dir/fftw3.f" 2>/dev/null || true
        fi

        # Recreate lib symlink with relative path if source exists
        if [ -d "$lib_dir" ] && [ -d "$w_lib" ]; then
            if [ -L "$lib_dir/libfftw3xc_wrapper_fftz.so" ] || [ -f "$lib_dir/libfftw3xc_wrapper_fftz.so" ]; then
                rm -f "$lib_dir/libfftw3xc_wrapper_fftz.so"
        fi
            ln -s "../amd-fftz/wrappers/fftw/lib/libfftw3xc_wrapper_fftz.so" "$lib_dir/libfftw3xc_wrapper_fftz.so" 2>/dev/null || true
        fi
    }

    #check if lib and include directories present
    if [ -d $libs_dir ] && [ -d $include_dir_lp64 ] && [ -d $include_dir_ilp64 ]; then
        if [ "$lib" = "fftz" ]; then
            relink_fftz_symlinks
        fi
        collect_input
        collect_lib
        
	#TODO: Update pkgconfig files based on text instead of line number
	if [ "$lib" == "libflame" ]; then
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/flame.pc
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/static/flame.pc
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/flame.pc
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/static/flame.pc
        fi

	if [ "$lib" == "blis" ]; then
	   # Pkgconfig for BLIS Multithread LP64 support
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/blis-mt.pc
            sed -i "2 c\exec_prefix=$libs_root" $libs_dir_lp64/pkgconfig/blis-mt.pc
            sed -i "3 c\libdir=$libs_root/lib" $libs_dir_lp64/pkgconfig/blis-mt.pc
            sed -i "4 c\includedir=$libs_root/include" $libs_dir_lp64/pkgconfig/blis-mt.pc
            sed -i "11 c\Cflags: -I$libs_root/include" $libs_dir_lp64/pkgconfig/blis-mt.pc

             # Pkgconfig for BLIS Singlethread LP64 support
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/blis.pc
            sed -i "2 c\exec_prefix=$libs_root" $libs_dir_lp64/pkgconfig/blis.pc
            sed -i "3 c\libdir=$libs_root/lib" $libs_dir_lp64/pkgconfig/blis.pc
            sed -i "4 c\includedir=$libs_root/include" $libs_dir_lp64/pkgconfig/blis.pc
            sed -i "11 c\Cflags: -I$libs_root/include" $libs_dir_lp64/pkgconfig/blis.pc


            #  Pkgconfig for BLIS Multithread ILP64 support
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/blis-mt.pc
            sed -i "2 c\exec_prefix=$libs_root" $libs_dir_ilp64/pkgconfig/blis-mt.pc
            sed -i "3 c\libdir=$libs_root/lib" $libs_dir_ilp64/pkgconfig/blis-mt.pc
            sed -i "4 c\includedir=$libs_root/include" $libs_dir_ilp64/pkgconfig/blis-mt.pc
            sed -i "11 c\Cflags: -I$libs_root/include" $libs_dir_ilp64/pkgconfig/blis-mt.pc

            #  Pkgconfig for BLIS Singlethread ILP64 support
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/blis.pc
            sed -i "2 c\exec_prefix=$libs_root" $libs_dir_ilp64/pkgconfig/blis.pc
            sed -i "3 c\libdir=$libs_root/lib" $libs_dir_ilp64/pkgconfig/blis.pc
            sed -i "4 c\includedir=$libs_root/include" $libs_dir_ilp64/pkgconfig/blis.pc
            sed -i "11 c\Cflags: -I$libs_root/include" $libs_dir_ilp64/pkgconfig/blis.pc

	fi

	if [ "$lib" == "utils" ]; then
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/aocl-utils.pc
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/aocl-utils.pc
        fi

        if [ "$lib" == "fftw" ]; then
            sed -i "1 c\prefix=$libs_root" $libs_dir_lp64/pkgconfig/fftw*
            sed -i "1 c\prefix=$libs_root" $libs_dir_ilp64/pkgconfig/fftw*
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_lp64/cmake/fftw3/FFTW3Config.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_lp64/cmake/fftw3/FFTW3fConfig.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_lp64/cmake/fftw3/FFTW3lConfig.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_lp64/cmake/fftw3/FFTW3qConfig.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_ilp64/cmake/fftw3/FFTW3Config.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_ilp64/cmake/fftw3/FFTW3fConfig.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_ilp64/cmake/fftw3/FFTW3lConfig.cmake
            sed -i "10 c\set (FFTW3_LIBRARY_DIRS $libs_root/lib)" $libs_dir_ilp64/cmake/fftw3/FFTW3qConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_lp64/cmake/fftw3/FFTW3Config.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_lp64/cmake/fftw3/FFTW3fConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_lp64/cmake/fftw3/FFTW3lConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_lp64/cmake/fftw3/FFTW3qConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_ilp64/cmake/fftw3/FFTW3Config.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_ilp64/cmake/fftw3/FFTW3fConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_ilp64/cmake/fftw3/FFTW3lConfig.cmake
            sed -i "11 c\set (FFTW3_INCLUDE_DIRS $libs_root/include)" $libs_dir_ilp64/cmake/fftw3/FFTW3qConfig.cmake
        fi
    fi
}

collect_input ()
{
    libs_root="$AMD_LIBS_ROOT/$AMD_REL"

    #return if lib are scalapack
    if [ "$lib" = "scalapack" ]; then
           return
    fi
    #copy all includes in include directory
    key=amd-$lib/include
	key_wrapper=amd-$lib/wrappers/fftw/include

    for filepath in $libs_root/$key/*
    do
        filename=$(basename $filepath)
        if [ "$filename" = "*" ]; then
                echo  "WARNING: $key not found proceeding with next  library "
                return
        fi

        if [[ "$lib" = "blis" || "$lib" = "libflame" || "$lib" = "da" ]]; then
            cp -r $libs_root/$key/LP64/* $libs_root/include_LP64/;
            cp -r $libs_root/$key/ILP64/* $libs_root/include_ILP64/;
        fi

	    if [ "$lib" = "sparse" ]; then
            cp -r $libs_root/$key/* $libs_root/include_LP64/;
            cp -r $libs_root/$key/* $libs_root/include_ILP64/;
        fi
	    if [[ "$lib" = "dlp" ]]; then
  	    	cp -r $libs_root/$key/MT/* $libs_root/include_LP64/;
	    	cp -r $libs_root/$key/MT/* $libs_root/include_ILP64/;
		fi

        #copy only if its a header file
        if [[ -f $libs_root/$key/$filename ]]; then
            if [ "$lib" = "compression" ]; then
                cp $libs_root/$key/$filename $libs_root/include_LP64/$filename;
            else
                cp $libs_root/$key/$filename $libs_root/include_LP64/$filename;
                cp $libs_root/$key/$filename $libs_root/include_ILP64/$filename;
            fi
        fi

				#copy only if its a header file
        if [ "$lib" = "fftz" ]; then
            cp -r $libs_root/$key/$filename $libs_root/include_LP64/$filename;
            cp -r $libs_root/$key/$filename $libs_root/include_ILP64/$filename;
        fi

        # copy include/alcp dir for crypto
        if [ "$lib" = "crypto" ]; then
            cp -r $libs_root/$key/alcp/ $libs_root/include_LP64/;
            cp -r $libs_root/$key/alcp/ $libs_root/include_ILP64/;
        fi

        if [ "$lib" = "utils" ]; then
           cp -r $libs_root/$key/*/ $libs_root/include_LP64/;
           cp -r $libs_root/$key/*/ $libs_root/include_ILP64/;
        fi

    done

    rm -rf $libs_root/$key
    echo -e "collected includes  in the common include directory" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    echo  2>&1 | tee -a $AMD_LIBS_LOG_FILE
}



collect_lib()
{
    libs_root="$AMD_LIBS_ROOT/$AMD_REL"
    libs_dir="$libs_root"

    libs_dir_lp64="$libs_dir/lib_LP64"
    libs_dir_ilp64="$libs_dir/lib_ILP64"

    #collect all libraries at one place
    key=amd-$lib/lib
    key_wrapper=amd-$lib/wrappers/fftw/lib
    
    for filepath in $libs_root/$key/*
    do
        filename=$(basename $filepath)
            if [ "$filename" = "*" ]; then
                echo  "**********WARNING: $key not found proceeding with next library********* "
                return
            fi

        if [ "$filename" = "LP64" ]; then
            cp -r $libs_root/$key/$filename/* $libs_dir_lp64/
        elif [ "$filename" = "ILP64" ]; then
            cp -r $libs_root/$key/$filename/* $libs_dir_ilp64/
        else
            if [ "$lib" = "compression" ]; then
                cp -r $libs_root/$key/$filename/* $libs_dir_lp64/
            elif [ "$lib" = "fftz" ]; then
                cp -r $libs_root/$key/$filename $libs_dir_ilp64/
                cp -r $libs_root/$key/$filename $libs_dir_lp64/
                cp -r $libs_root/$key_wrapper/*.a $libs_dir_ilp64/
                cp -r $libs_root/$key_wrapper/*.a $libs_dir_lp64/
            else
                #for non lp64, ilp64 variants (like fftw, libm)
                if [ "$lib" = "libmem" ]; then
                    cp -r $libs_root/$key/$filename $libs_dir_ilp64/
                    cp -r $libs_root/$key/$filename $libs_dir_lp64/
		elif [ "$lib" = "dlp" ]; then
		    cp -r $libs_root/$key/$filename/* $libs_dir_ilp64/
		    cp -r $libs_root/$key/$filename/* $libs_dir_lp64/
                else
                    cp -r $libs_root/$key/$filename $libs_dir_ilp64/
                    cp -r $libs_root/$key/$filename $libs_dir_lp64/
		fi
            fi
        fi
    done
    rm -rf $libs_root/$key
    echo -e "collected libs for $lib in the common lib directory" 2>&1 | tee -a $AMD_LIBS_LOG_FILE
    echo  2>&1 | tee -a $AMD_LIBS_LOG_FILE
}

#Format the messages
print_result() {
    install_path=$1
    libs_root=$2
    printf -- '-%.0s' {1..100};
    printf "\nAOCL INSTALLED SUCCESSFULLY AT: $install_path\n";
    printf -- '-%.0s' {1..100};
    printf "\nSteps to setup:\n";
    printf -- '-%.0s' {1..100};
    printf "\nSetup AOCL environment by executing the below command\n";
    printf "\tsource $libs_root/amd-libs.cfg\n";
    printf -- '-%.0s' {1..100};
    printf "\nTo set AOCL environment via module system\n";
    printf "\tcd $libs_root\n";
    printf "\tmodule load ./aocl-linux-aocc-${AMD_LIBS_REL}_module\n";
    printf "\nTo unset AOCL environment via module system\n";
    printf "\tcd $libs_root\n";
    printf "\tmodule unload ./aocl-linux-aocc-${AMD_LIBS_REL}_module\n";
    printf -- '-%.0s' {1..100};
    printf "\n";
}

#Chose to create symlinks for lp64
create_symlinks_lp64() {
    lib_path=$1
    lib_path_lp64=${lib_path}/lib_LP64;

    # if folder isnt empty,
    if ! [ -z "$(ls -A ${lib_path_lp64})" ]; then
        cd $lib_path;

        if [ -L ./lib ]; then
            rm -rf ./lib;
        fi
        ln -s ./lib_LP64 ./lib;
        cd ../;
        print_result "$lib_path/lib" "$lib_path"
    fi
}

#Chose to create symlinks for ilp64
create_symlinks_ilp64() {
    lib_path=$1
    lib_path_ilp64=${lib_path}/lib_ILP64;
    # if folder isnt empty,
    if ! [ -z "$(ls -A ${lib_path_ilp64})" ]; then
        cd $lib_path;

        if [ -L ./lib ]; then
            rm -rf ./lib;
        fi
        ln -s ./lib_ILP64 ./lib;
        cd ../;
        print_result "$lib_path/lib" "$lib_path"
    fi
}

#Chose to create symlinks for include ilp64
create_symlinks_include_ilp64() {
    inc_path=$1
    inc_path_ilp64=${inc_path}/include_ILP64;
    # if folder isnt empty,
    if ! [ -z "$(ls -A ${inc_path_ilp64})" ]; then
        cd $inc_path;

        if [ -L ./include ]; then
            rm -rf ./include;
        fi
        ln -s ./include_ILP64/ ./include;

        cd ../;
    fi
}

#Chose to create symlinks for include lp64
create_symlinks_include_lp64() {
    inc_path=$1
    inc_path_lp64=${inc_path}/include_LP64;
    # if folder isnt empty,

    if ! [ -z "$(ls -A ${inc_path_lp64})" ]; then
        cd $inc_path;

        if [ -L ./include ]; then
            rm -rf ./include;
        fi
        ln -s ./include_LP64/ ./include;

        cd ../;
    fi
}

#Choose to install lp64/ilp64
install_special() {
    #ask if we need to install lp64, ilp64 symlinks
    lib_root=$1
    lib_path=${lib_root};

    while true; do
    printf -- '-%.0s' {1..125}
    printf "\nDo you want to set LP64 or ILP64 libraries as default libraries? (Enter 1 for LP64 / 2 for ILP64 / Default option: 1)\n"
    printf -- '-%.0s' {1..125}
    printf "\n"
    read -p "" opt
    case $opt in
        [1]* ) echo "Setting LP64 libs as default";  create_symlinks_lp64 ${lib_path}; create_symlinks_include_lp64 ${libs_root};  break;;
        [2]* ) echo "Setting ILP64 libs as default"; create_symlinks_ilp64 ${lib_path}; create_symlinks_include_ilp64 ${libs_root};
               if [ "$lib" = "compression" ]; then
                   echo "AOCL-Compression is not validated for ILP64";
               else
                   echo "Setting ILP64 libs as default"; create_symlinks_ilp64 ${lib_path}; create_symlinks_include_ilp64 ${libs_root};
               fi
               break;;
         * )    echo "Setting LP64 libs as default"; create_symlinks_lp64 ${lib_path}; create_symlinks_include_lp64 ${libs_root};break;;
    esac
    done
}

install() {
    init_env
    echo "AOCL available at: $libs_root" 2>&1 | tee -a $AMD_LIBS_LOG_FILE

    #here check the value of $SELECTED_INT_TYPE and accordingly set sym links
    if [ "$SELECTED_INT_TYPE" = "lp64" ]; then
        printf "Setting LP64 libs as default\n";
        create_symlinks_lp64 ${libs_root};
        create_symlinks_include_lp64 ${libs_root};

    elif [ "$SELECTED_INT_TYPE" = "ilp64" ]; then
        printf "Setting ILP64 libs as default\n";
        create_symlinks_ilp64 ${libs_root};
        create_symlinks_include_ilp64 ${libs_root};

    else
        install_special $libs_root; # call default option where user will be prompted to choose ilp64/lp64
    fi

    echo
}


install
