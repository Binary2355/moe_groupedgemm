import glob
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def get_gpu_arch():
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"GPU Architecture: compute_{major}{minor}")
        return f"compute_{major}{minor},code=sm_{major}{minor}"
    return None

def get_cpu_parallel_jobs():
    import psutil

    try:
        num_cpu = psutil.cpu_count(logical=False)
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4
    return str(cpu_use)


def get_extensions():
    os.environ.setdefault('MAX_JOBS', get_cpu_parallel_jobs())
    define_macros = []

    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    extra_compile_args = {'cxx': ['-std=c++17', '-Wl,--no-undefined'],}

    base_dir = os.path.abspath(os.path.dirname(__file__))
    cutlass_include = os.path.join(base_dir, "third_party", "cutlass", "include")
    if os.path.isdir(cutlass_include):
        print(f"Using CUTLASS from: {cutlass_include}")
        define_macros.append(('USE_CUTLASS', None))
    else:
        print("未找到 third_party/cutlass/include 目录")
        cutlass_include = None

    include_dirs = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, 'mlperf', 'ops', 'csrc')
    import torch
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        gpu_arch = get_gpu_arch()
        print(f'当前GPU架构为[{gpu_arch}]')
        if gpu_arch:
            extra_compile_args['nvcc'] = [f"-gencode=arch={gpu_arch}"]
        else:
            extra_compile_args['nvcc'] = []
        op_files = [os.path.join(project_dir, 'common/*.cpp'),
                    os.path.join(project_dir, 'host/*.cpp'),
                    os.path.join(project_dir, 'host/cuda/*.cu')]
        op_files = [y for x in op_files for y in glob.glob(x)]
        extension = CUDAExtension
        include_dirs.append(os.path.abspath(os.path.join(project_dir, 'include')))
        include_dirs.append(os.path.abspath(os.path.join(project_dir, 'host/include')))
        include_dirs.append(os.path.abspath(os.path.join(project_dir, 'kernel/cuda')))
        include_dirs.append(os.path.join(cuda_home, 'include'))
        if cutlass_include:
            include_dirs.append(cutlass_include)
    else:
        op_files = [os.path.join(project_dir, 'common/*.cpp'),
                    os.path.join(project_dir, 'host/*.cpp')]
        op_files = [y for x in op_files for y in glob.glob(x)]
        extension = CppExtension
        include_dirs.append(os.path.abspath(os.path.join(project_dir, 'include')))
        include_dirs.append(os.path.abspath(os.path.join(project_dir, 'host/include')))
        if cutlass_include:
            include_dirs.append(cutlass_include)

    if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
        extra_compile_args['nvcc'] += ['-O3', '--expt-relaxed-constexpr', '-std=c++17']

    print(f"Compiling with sources: {op_files}")
    print(f"Include dirs: {include_dirs}")
    print(f"Extra compile args: {extra_compile_args}")
    print(f"Defined macros: {define_macros}")

    return [extension(
        name='mlperf.ext',
        sources=op_files,
        include_dirs=include_dirs,
        include_package_data=True,
        library_dirs=[os.path.join(cuda_home, 'lib64')],
        libraries=['cudart'],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)]

def main():
    setup(
        name='mlperf',
        version='0.0.1',
        packages=find_packages(),
        include_package_data=True,
        author='zekali',
        author_email='zekali.007@bytedance.com',
        description='moe groupedgemm with CUTLASS support',
        long_description='A detailed description of the mlperf package using CUTLASS for accelerated GPU operations',
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        setup_requires=['torch', 'psutil', 'pybind11>=2.6', 'wheel', 'setuptools>=60.0'],
        install_requires=['torch', 'psutil'],
        options={'install': {'prefix': '/usr/local'}}
    )

if __name__ == "__main__":
    main()