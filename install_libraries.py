import subprocess

libraries = ['psutil', 'pandas', 'llama_cpp']

for library in libraries:
    subprocess.call(['pip', 'install', library])

# Additional installation for llama_cpp
# Note: Make sure to replace 'llama_cpp_package.whl' with the actual package name and path.
subprocess.call(['pip', 'install', 'llama_cpp_package.whl'])
