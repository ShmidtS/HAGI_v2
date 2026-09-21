$ErrorActionPreference = "Continue"
$cmake = "C:\HAGI_v2\.venv\Lib\site-packages\cmake\data\bin\cmake.exe"
Set-Location C:\HAGI_v2\llama-glm5
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && $cmake -B build-vulkan -S . -G `"NMake Makefiles`" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON" 2>&1 | Select-Object -Last 3
