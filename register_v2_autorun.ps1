# Registers a boot-triggered task that resumes the v2 full pass after a reboot.
# One-shot setup: powershell -File register_v2_autorun.ps1
$act = New-ScheduledTaskAction -Execute "C:\Program Files\Git\bin\bash.exe" `
    -Argument "-lc 'cd /c/HAGI_v2 && bash seq_v2_resume.sh >> seq_v2_autorun.out 2>&1'"
$trg = New-ScheduledTaskTrigger -AtStartup
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 2)
Register-ScheduledTask -TaskName "HAGI_v2_resume" -Action $act -Trigger $trg -Settings $set -Force
Write-Output "registered: HAGI_v2_resume (AtStartup)"
