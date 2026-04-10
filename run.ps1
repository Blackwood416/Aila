cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }; Write-Host ':: oneAPI environment initialized ::' -ForegroundColor Green
cd e:\RiderProjects\Aila\build
.\Aila.exe -m ..\Qwen3.5-0.8B | Tee-Object -FilePath "e:\RiderProjects\Aila\run_log.txt"
cd e:\RiderProjects\Aila