param([string]$MatchPattern = "models.pipeline|geographic_holdout_eval")
# Resume the GreenGrid-AI Phase 5 training process tree (Windows).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/resume_training.ps1

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class ProcSuspend {
    [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
    [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
    const uint SUSPEND = 0x0800; // PROCESS_SUSPEND_RESUME
    public static int Suspend(int pid) {
        IntPtr h = OpenProcess(SUSPEND, false, pid);
        if (h == IntPtr.Zero) return -1;
        int r = NtSuspendProcess(h);
        CloseHandle(h);
        return r;
    }
    public static int Resume(int pid) {
        IntPtr h = OpenProcess(SUSPEND, false, pid);
        if (h == IntPtr.Zero) return -1;
        int r = NtResumeProcess(h);
        CloseHandle(h);
        return r;
    }
}
"@


$main = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match $MatchPattern } |
    Select-Object -First 1

if ($null -eq $main) { Write-Host "No training process found."; exit 1 }

$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$tree = @($main.ProcessId)
$changed = $true
while ($changed) {
    $changed = $false
    foreach ($p in $all) {
        if (($tree -contains $p.ParentProcessId) -and -not ($tree -contains $p.ProcessId)) {
            $tree += $p.ProcessId; $changed = $true
        }
    }
}
foreach ($procId in $tree) {
    $r = [ProcSuspend]::Resume($procId)
    Write-Host ("Resumed PID " + $procId + " -> " + $(if ($r -eq 0) {"OK"} else {"FAILED (" + $r + ")"}))
}
Write-Host ("Resumed " + $tree.Count + " process(es). Training continues from the exact point it stopped.")
