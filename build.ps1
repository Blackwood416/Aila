cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }; Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green
cd e:\RiderProjects\Aila\build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd e:\RiderProjects\Aila
