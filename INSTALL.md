## Getting Started

This is a guide on how to install EMopt and its dependencies

---

### Prerequisites

**Optional:** Install miniconda to create an environment dedicated for EMopt.

* If using miniconda or anaconda, replace all instances of ``pip`` with ``conda``

```sh 
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
sh Miniconda3-py39_4.9.2-Linux-x86_64.sh
# follow instructions
 ```

Create and activate an environment for EMopt

 ```sh
conda create --name emopt
conda activate emopt
 ```


**Required:** Install (or update) required packages

CentosOS:
  ```sh
sudo yum install epel-release
yum install http://opensource.wandisco.com/centos/7/git/x86_64/wandisco-git-release-7-2.noarch.rpm
sudo yum groupinstall "Development Tools"
sudo yum install openmpi openmpi-devel git python-pip python-devel tkinter
module load mpi/openmpi-x86_64
  ```

Ubuntu/Debian:
```sh
sudo apt-get install build-essential gfortran openmpi-bin libopenmpi-dev python python-dev python-pip git python-tk
```


**Required:** Then, install required python modules

```sh
pip install requests matplotlib numpy scipy mpi4py h5py
```

---

### Installation
1. Clone the repo and switch directory into it
   ```sh
   git clone https://github.com/JosueCom/emopt.git
   cd emopt
   ```
2. Install required libraries and modules
   ```sh
   python install.py
   ```
4. Install EMopt
   ```sh
   python setup.py install
   ```
