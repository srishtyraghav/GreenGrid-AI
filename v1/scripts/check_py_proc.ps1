Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id,
        @{n='CPU_s'; e={[math]::Round($_.CPU, 1)}},
        @{n='WS_MB'; e={[math]::Round($_.WorkingSet64 / 1MB)}} |
    Format-Table -AutoSize
