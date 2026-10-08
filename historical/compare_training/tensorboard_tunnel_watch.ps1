$ErrorActionPreference = 'Continue'
$ssh = 'C:\Windows\System32\OpenSSH\ssh.exe'
$log = Join-Path $env:USERPROFILE '.ssh\tensorboard_compare_6024_watch.log'
$mutex = New-Object System.Threading.Mutex($false, 'Local\G1D_Tensorboard_Compare_6024')
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    while ($true) {
        Add-Content -Path $log -Value "$(Get-Date -Format o) starting TensorBoard SSH forward"
        & $ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L '100.122.105.65:6024:127.0.0.1:6024' pytorch-ng-2223591 2>> $log
        Add-Content -Path $log -Value "$(Get-Date -Format o) forward exited ($LASTEXITCODE), retry in 5s"
        Start-Sleep -Seconds 5
    }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
