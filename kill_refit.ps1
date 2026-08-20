# Kill all dsv4_refit_experts.py processes (stop the refit pass).
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'dsv4_refit' }
foreach ($p in $procs) {
    Write-Host ("killing pid {0}: {1}" -f $p.ProcessId, $p.CommandLine.Substring(0, [Math]::Min(80, $p.CommandLine.Length)))
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host ("killed {0} refit process(es)" -f $procs.Count)
