# Installing dlib on Windows - Step by Step Guide

## Problem
dlib requires CMake executable (not the Python package) to build from source. On Windows, this can be challenging.

## ⚠️ IMPORTANT: Install CMake Executable (Not Python Package)

The `pip install cmake` installs a Python wrapper, NOT the actual CMake executable needed by dlib.

### Option 1: Install CMake Executable (Recommended)

1. **Download CMake:**
   - Go to https://cmake.org/download/
   - Download "Windows x64 Installer" (cmake-x.x.x-windows-x86_64.msi)
   - Or use direct link: https://cmake.org/files/v3.28/cmake-3.28.0-windows-x86_64.msi

2. **Install CMake:**
   - Run the installer
   - **CRITICAL:** Check "Add CMake to system PATH for all users" during installation
   - Complete the installation

3. **Close and reopen your terminal/PowerShell** (to refresh PATH)

4. **Verify CMake installation:**
   ```bash
   cmake --version
   ```
   Should show: `cmake version 3.x.x`

5. **Install Visual Studio Build Tools (Required):**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Build Tools for Visual Studio 2022"
   - Select "Desktop development with C++" workload
   - This installs the C++ compiler needed to build dlib

6. **Install dlib:**
   ```bash
   pip install dlib
   pip install face-recognition
   ```

### Option 2: Use Conda (Easier if you have Anaconda/Miniconda)

```bash
conda install -c conda-forge dlib
pip install face-recognition
```

### Option 3: Use Pre-built Wheel (Alternative)

Try installing from an unofficial source (use at your own risk):

```bash
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp312-cp312-win_amd64.whl
pip install face-recognition
```

**Note:** Replace `cp312` with your Python version if different (cp311, cp310, etc.)

### Option 4: Alternative Library (No dlib required)

If dlib installation continues to fail, we can modify the code to use `deepface` or `opencv` for face recognition instead, which don't require dlib.

## After Installation

1. Restart your terminal/PowerShell
2. Verify installation:
   ```bash
   python -c "import face_recognition; print('Success!')"
   ```
3. Restart your Flask application
4. Try uploading a face image again
