#!/bin/bash
read -p "Do you have an existing installation of mamba (community made anaconda reimplementation)? [y/n]" existing_mamba

case $existing_mamba in
    y)
        alias mamba_exec=mamba
        ;;
    n) 
        read -p "mamba/conda are required for this installation script. Do you want to install mamba? If you already have conda, please choose 'n'.  mamba is highly recommended, conda may not be able to resolve dependencies in a reasonable amount of time. See https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html if you'd like to install mamba into an existing conda install. [y/n]" install_mamba
        case $install_mamba in
            y) 
                read -p "Are you using a Linux x86-64 system? [y/n]" linux_flag
                case $linux_flag in
                    y) 
                        echo "Installing mamba. Please follow the prompts."
                        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
                        bash Miniforge3-Linux-x86_64.sh
                        alias mamba_exec=mamba
                        ;;
                    n) 
                        echo "Please navigate to https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html and follow the instructions there to install mamba. Then rerun this setup script. Exiting."
                        return 0
                        ;;
                    *) 
                        echo invalid response
			return 1
                        ;;
                 esac
                 ;;
            n) 
                echo "Not installing mamba. We will assume that you have conda installed. Note, this may result in dependency resolution issues."
                alias mamba_exec=conda
                ;;
            *) 
                echo invalid response
		return 1
                ;;
            esac
            ;;
    *) 
        echo invalid response
	return 1
        ;;
esac

source ~/.bashrc

echo "The remainder of this script will assume that you will want to install OpenMPI, Boost, Eigen, PETSc, and SLEPc in your local mamba/conda environment. If you have an existing installation of any of these packages, you may instead want to do a custom install described at: https://emopt.readthedocs.io/en/latest/installation.html."
read -p "Continue? [y/n]" continue_var

case $continue_var in
    y) ;;
    n) return 0;;
    *) echo invalid response; return 1;;
esac

read -p "Please provide a name for your EMopt environment, or hit enter for default name (emopt):" emopt_name
emopt_name=${emopt_name:=emopt}

read -p "Do you want to install PyTorch? This is needed for experimental EMopt features such as AutoDiff-accelerated feature-mapping and free-form topology optimization. It is not required for use with the base EMopt package. [y/n]" install_torch

case $install_torch in
    y)
        read -p "Do you have an NVIDIA GPU for PyTorch? [y/n]" have_gpu
        case $have_gpu in
             y)
                 mamba_exec create --name $emopt_name -c conda-forge -y python pip numpy scipy matplotlib requests h5py future eigen=3.3.7 boost=1.73.0 mpi4py openmpi petsc=*=*complex* petsc4py slepc slepc4py pytorch
		 ;;
             n)
                 mamba_exec create --name $emopt_name -c conda-forge -y python pip numpy scipy matplotlib requests h5py future eigen=3.3.7 boost=1.73.0 mpi4py openmpi petsc=*=*complex* petsc4py slepc slepc4py pytorch-cpu
		 ;;
             *) echo invalid response; return 1;;
        esac
	;;
    n)
        mamba_exec create --name $emopt_name -c conda-forge -y python pip numpy scipy matplotlib requests h5py future eigen=3.3.7 boost=1.73.0 mpi4py openmpi petsc=*=*complex* petsc4py slepc slepc4py
	;;
    *) echo invalid response; return 1;;
esac

mamba_exec activate $emopt_name

if [ $? -eq 0 ]
then
    echo "Successfully installed dependencies! Installing EMopt..."
else
    echo "Depedencies failed to install."
    return 1
fi

echo export OMP_NUM_THREADS=1 >> ~/.bashrc
rm ~/.emopt_deps
touch ~/.emopt_deps
echo export EIGEN_DIR=$CONDA_PREFIX/include/eigen3 >> ~/.emopt_deps
echo export BOOST_DIR=$CONDA_PREFIX/include/ >> ~/.emopt_deps
echo export PETSC_DIR=$CONDA_PREFIX/ >> ~/.emopt_deps
echo export PETSC_ARCH="" >> ~/.emopt_deps
echo export SLEPC_DIR=$CONDA_PREFIX/ >> ~/.emopt_deps

source ~/.emopt_deps

pip install -vv --no-deps --force-reinstall .
pip install -vv --no-deps --force-reinstall .

if [ $? -eq 0 ]
then
    echo "Succesfully installed EMopt!. Exiting."
    return 0
else
    echo "There was a problem in pip install. Exiting."
    return 1
fi
