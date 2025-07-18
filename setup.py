import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob
import platform

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

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Get PyTorch CMake path
        try:
            torch_cmake_prefix = os.environ.get('Torch_DIR')
            if not torch_cmake_prefix:
                torch_cmake_prefix = subprocess.check_output(
                    [sys.executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
                    text=True,
                    encoding='utf-8'
                ).strip()
            print(f"DEBUG: Using torch_cmake_prefix: {torch_cmake_prefix}")
        except subprocess.CalledProcessError as e:
            print(f"Error getting torch cmake path: {e}")
            sys.exit(1)

        # Get CPU core count
        try:
            threads_output = subprocess.check_output(["nproc"], text=True, encoding='utf-8').strip()
            threads = threads_output if threads_output.isdigit() else "2"
        except subprocess.CalledProcessError:
            threads = "2"

        print(f"Using {threads} build threads.")

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={torch_cmake_prefix}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DVERSION_INFO={self.distribution.get_version()}',

            '-DENABLE_HDF5=OFF',
            '-DENABLE_PYBIND=ON',
            '-DCMAKE_INSTALL_PREFIX=/usr/local/lib',
            '-DENABLE_PAPI=OFF',
            '-DENABLE_SPTAG=ON',
            '-DENABLE_DiskANN=ON',
            '-DPYBIND=ON',
        ]

        # MKL path
        mkl_base_path = os.environ.get('MKLROOT')
        if not mkl_base_path:
            raise RuntimeError("MKLROOT environment variable is not set. Please ensure Intel oneAPI MKL is installed and MKLROOT is configured in Dockerfile.")

        cmake_args.append(f'-DMKL_PATH={mkl_base_path}')
        cmake_args.append(f'-DMKL_INCLUDE_PATH={mkl_base_path}/include')

        print(f"DEBUG: Setting -DMKL_PATH={mkl_base_path}")
        print(f"DEBUG: Setting -DMKL_INCLUDE_PATH={mkl_base_path}/include")

        build_args = ['--config', cfg]
        build_args += ['--', '-j' + threads]

        self.build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(self.build_temp, exist_ok=True)
        print(f"DEBUG: Using build directory: {self.build_temp}")

        print(f"DEBUG: CMake configure command: {['cmake', ext.sourcedir] + cmake_args}")
        try:
            subprocess.run(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, check=True)
        except subprocess.CalledProcessError as e:
            print(f"CMake configuration failed with error: {e}")
            print(f"STDOUT:\n{e.stdout.decode() if e.stdout else ''}")
            print(f"STDERR:\n{e.stderr.decode() if e.stderr else ''}")
            raise

        print(f"DEBUG: CMake build command: {['cmake', '--build', '.'] + build_args}")
        try:
            result = subprocess.run(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, capture_output=True, text=True)
            print("--- CMake Build STDOUT ---")
            print(result.stdout)
            print("--- CMake Build STDERR ---")
            print(result.stderr)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed with error: {e}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            raise

        lib_name = "libCANDYBENCH.so"
        py_lib_name = "PyCANDYAlgo.so"

        so_files = glob.glob(os.path.join(self.build_temp, py_lib_name))
        if not so_files:
            so_files = glob.glob(os.path.join(self.build_temp, lib_name))

        print(f"Discovered .so files in {self.build_temp}:")
        print(so_files)

        if not so_files:
            print(f"WARNING: No .so files found in {self.build_temp}. Checking common subdirectories...")
            so_files = glob.glob(os.path.join(self.build_temp, 'lib', '*.so'))
            if not so_files:
                so_files = glob.glob(os.path.join(self.build_temp, 'bin', '*.so'))
            if not so_files:
                print("WARNING: Still no .so files found in common subdirectories. Build might have failed or output path is unusual.")


        for file in so_files:
            target_path = os.path.join(extdir, os.path.basename(file))
            print(f"Copying {file} to {target_path}")
            shutil.copy(file, extdir)

setup(
    name='PyCANDYAlgo',
    version='0.1',
    author='Your Name',
    description='A simple python version of CANDY benchmark built with Pybind11 and CMake',
    long_description='',
    ext_modules=[CMakeExtension('PyCANDYAlgo', sourcedir=".")],
    cmdclass={
        'build_ext': CMakeBuild,
    },
    zip_safe=False,
)