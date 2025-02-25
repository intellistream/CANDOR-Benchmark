import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob
import multiprocessing

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is installed
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # Set environment variables
        os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
        if sys.platform == 'linux':
            os.environ['LD_LIBRARY_PATH'] = '/path/to/custom/libs:' + os.environ.get('LD_LIBRARY_PATH', '')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        os.system("python3 -c 'import torch;print(torch.utils.cmake_prefix_path)' >> 1.txt")
        with open('1.txt', 'r') as file:
            torchCmake = file.read().rstrip('\n')
        os.system('rm 1.txt')
        os.system('nproc >> 1.txt')
        with open('1.txt', 'r') as file:
            threads = file.read().rstrip('\n')
        threads = str(multiprocessing.cpu_count())
        os.system('rm 1.txt')
        #os.system('cd thirdparty&&./makeClean.sh&&./installPAPI.sh')
        print(threads)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_PREFIX_PATH='+torchCmake,
                      '-DENABLE_HDF5=ON', 
                      '-DENABLE_PYBIND=ON',
                      '-DCMAKE_INSTALL_PREFIX=/usr/local/lib',
                      '-DENABLE_PAPI=OFF',
                      '-DENABLE_SPTAG=OFF',
                      '-DENABLE_PUCK=ON',
                      '-DENABLE_DiskANN=ON',
                      '-DPYBIND=ON',
                      f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
                      f"-DPYTHON_EXECUTABLE={sys.executable}",
                      f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
                      f"-DVERSION_INFO={self.distribution.get_version()}"  # commented out, we want this set in the CMake file
                   ]
        
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        build_args = ['--config', cfg]
        build_args +=  ['--', '-j'+ threads]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.run(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp,check=True)
        subprocess.run(['cmake', '--build', '.'] + build_args, cwd=self.build_temp,check=True)
        # Now copy all *.so files from the build directory to the final installation directory
        so_files = glob.glob(os.path.join(self.build_temp, '*.so'))
        print("so_files:")
        print(so_files)
        for file in so_files:
            shutil.copy(file, extdir)
setup(
    name='PyCANDYAlgo',
    version='0.1',
    author='Your Name',
    description='A simple python version of CANDY benchmark built with Pybind11 and CMake',
    long_description='',
    ext_modules=[CMakeExtension('.')],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
)
