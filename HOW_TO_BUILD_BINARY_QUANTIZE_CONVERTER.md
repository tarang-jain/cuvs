# How to Build the Binary Quantization Converter

This guide explains how to build and use the binary quantization converter tool that has been integrated into the cuVS build system.

## Method 1: Using the Integrated CMake Build (Recommended)

The binary quantization converter has been integrated into the main cuVS CMake build system.

### Step 1: Configure the Build

Navigate to your cuVS repository and create/enter the build directory:

```bash
cd /raid/tarangj/cuvs
mkdir -p cpp/build
cd cpp/build
```

### Step 2: Run CMake with the Converter Enabled

Configure the project with the converter enabled (it's ON by default):

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;90" \
  -DBUILD_CUVS_BENCH_ANN=ON \
  -DCUVS_BUILD_BINARY_QUANTIZE_CONVERTER=ON
```

Or if you only want to build the converter without other benchmarks:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;90" \
  -DBUILD_TESTS=OFF \
  -DBUILD_CUVS_BENCH_ANN=ON \
  -DCUVS_BUILD_BINARY_QUANTIZE_CONVERTER=ON
```

### Step 3: Build the Converter

Build only the converter target:

```bash
make convert_binary_quantize -j
```

Or build everything:

```bash
make -j
```

### Step 4: Find and Use the Executable

The executable will be located at:
```
cpp/build/cpp/bench/ann/convert_binary_quantize
```

You can run it directly:

```bash
./cpp/bench/ann/convert_binary_quantize <input.fbin> <output.u8bin> [threshold_type]
```

Or install it to your system:

```bash
sudo make install
# This installs to /usr/local/bin by default
convert_binary_quantize <input.fbin> <output.u8bin> [threshold_type]
```

## Method 2: Using the Standalone Build Script

If you want a quick build without modifying the main CMake configuration:

```bash
cd /raid/tarangj/cuvs/cpp/bench/ann/src
./build_convert_binary.sh
```

This will create the executable in the build directory.

## Method 3: Manual Compilation

If you need to compile manually without CMake:

```bash
cd /raid/tarangj/cuvs

nvcc -std=c++17 \
     -I"cpp/include" \
     -I"_deps/raft-src/cpp/include" \
     -I"/usr/local/cuda/include" \
     -arch=sm_75 \
     -O3 \
     -o convert_binary_quantize \
     cpp/bench/ann/src/convert_binary_quantize.cpp \
     -lcuvs -lcudart -lcublas
```

## Disabling the Converter Build

If you don't want to build the converter, you can disable it:

```bash
cmake .. -DCUVS_BUILD_BINARY_QUANTIZE_CONVERTER=OFF
```

## Usage Examples

Once built, use the converter as follows:

### Basic Usage

```bash
# Convert with zero threshold (default)
./convert_binary_quantize input.fbin output.u8bin

# Convert with mean threshold
./convert_binary_quantize input.fbin output.u8bin 1

# Convert with sampling median threshold
./convert_binary_quantize input.fbin output.u8bin 2
```

### Batch Processing

Create a script to process multiple files:

```bash
#!/bin/bash
CONVERTER="./cpp/build/cpp/bench/ann/convert_binary_quantize"

for dataset in sift gist deep1b mnist; do
    echo "Converting $dataset..."
    $CONVERTER ${dataset}.fbin ${dataset}_binary.u8bin 1
done
```

## Integration with CI/CD

To integrate into your CI/CD pipeline, ensure the option is enabled in your build configuration:

```yaml
# .github/workflows/build.yml or similar
- name: Configure CMake
  run: |
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CUVS_BENCH_ANN=ON \
      -DCUVS_BUILD_BINARY_QUANTIZE_CONVERTER=ON

- name: Build Converter
  run: cmake --build build --target convert_binary_quantize
```

## Troubleshooting

### CMake Can't Find Target

If you get an error like "No rule to make target 'convert_binary_quantize'":
1. Make sure you've updated the CMakeLists.txt as shown above
2. Re-run CMake configuration: `cmake ..`
3. Check that `CUVS_BUILD_BINARY_QUANTIZE_CONVERTER` is ON

### Linking Errors

If you get linking errors:
1. Ensure cuVS is properly built with: `make cuvs`
2. Check that CUDA toolkit is properly installed
3. Verify raft dependencies are available

### Runtime Errors

If the converter fails at runtime:
1. Check CUDA device availability: `nvidia-smi`
2. Verify input file format (must be .fbin)
3. Ensure sufficient GPU memory

## Notes

- The CMakeLists_convert_binary.txt file is now obsolete since the configuration has been integrated into the main CMakeLists.txt
- The converter is built by default when `BUILD_CUVS_BENCH_ANN=ON`
- The converter will be installed to the `bin` directory when you run `make install`



