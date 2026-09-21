$ErrorActionPreference = "Continue"
$cmake = "C:\HAGI_v2\.venv\Lib\site-packages\cmake\data\bin\cmake.exe"
Set-Location C:\HAGI_v2\llama-glm5
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && $cmake --build build-vulkan -j 6" 2>&1 | Select-String -Pattern "error|LLVM|fatal|warning C|lld-link|llama-app" -Context 0,2 | Select-Object -First 30
