# Cross-platform startup script for ollama-proxy server (PowerShell)
# Works on Windows PowerShell and PowerShell Core (cross-platform)

[CmdletBinding()]
param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 11434,
    [string]$ApiKey = "",
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")]
    [string]$LogLevel = "INFO",
    [string]$ModelsFilter = "",
    [switch]$Dev,
    [switch]$Daemon,
    [string]$EnvFile = "",
    [switch]$CheckDeps,
    [switch]$DryRun,
    [switch]$Help
)

# Script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Colors for output
$Colors = @{
    Info = "Blue"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Type = "Info"
    )
    
    $color = $Colors[$Type]
    Write-Host "[$Type] $Message" -ForegroundColor $color
}

function Show-Usage {
    @"
Usage: .\start-server.ps1 [OPTIONS]

Start the ollama-proxy server with various configuration options.

PARAMETERS:
    -Host HOST              Host to bind to (default: 0.0.0.0)
    -Port PORT              Port to listen on (default: 11434)
    -ApiKey KEY             OpenRouter API key
    -LogLevel LEVEL         Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    -ModelsFilter FILE      Path to model filter file
    -Dev                    Run in development mode with auto-reload
    -Daemon                 Run in daemon mode (background service)
    -EnvFile FILE           Load environment from specific file
    -CheckDeps              Check dependencies before starting
    -DryRun                 Show what would be executed without running
    -Help                   Show this help message

ENVIRONMENT VARIABLES:
    OPENROUTER_API_KEY      OpenRouter API key (required)
    HOST                    Host to bind to
    PORT                    Port to listen on
    LOG_LEVEL               Logging level
    MODELS_FILTER_PATH      Path to model filter file

EXAMPLES:
    .\start-server.ps1                                    # Start with defaults
    .\start-server.ps1 -Dev                               # Start in development mode
    .\start-server.ps1 -Host 127.0.0.1 -Port 8080       # Custom host and port
    .\start-server.ps1 -ApiKey "sk-or-..." -Daemon       # Start as daemon
    .\start-server.ps1 -CheckDeps                         # Check dependencies only

"@
}

function Test-CommandExists {
    param([string]$Command)
    
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Test-Dependencies {
    Write-ColorOutput "Checking dependencies..." "Info"
    
    $missingDeps = @()
    
    # Check for Python
    if (-not (Test-CommandExists "python") -and -not (Test-CommandExists "python3")) {
        $missingDeps += "python"
    }
    
    # Check for uv
    if (-not (Test-CommandExists "uv")) {
        $missingDeps += "uv"
    }
    
    if ($missingDeps.Count -gt 0) {
        Write-ColorOutput "Missing dependencies: $($missingDeps -join ', ')" "Error"
        Write-ColorOutput "Please install the missing dependencies:" "Info"
        foreach ($dep in $missingDeps) {
            switch ($dep) {
                "python" {
                    Write-Host "  - Python 3.12+: https://www.python.org/downloads/"
                }
                "uv" {
                    Write-Host "  - uv: https://docs.astral.sh/uv/getting-started/installation/"
                }
            }
        }
        return $false
    }
    
    Write-ColorOutput "All dependencies are available" "Success"
    return $true
}

function Test-Environment {
    Write-ColorOutput "Validating environment..." "Info"
    
    # Check if we're in the project root
    if (-not (Test-Path (Join-Path $ProjectRoot "pyproject.toml"))) {
        Write-ColorOutput "Not in ollama-proxy project root. Expected pyproject.toml file." "Error"
        return $false
    }
    
    # Check if API key is set (if not provided via parameter)
    if (-not $ApiKey -and -not $env:OPENROUTER_API_KEY) {
        Write-ColorOutput "OPENROUTER_API_KEY not set. Make sure to provide it via -ApiKey or environment variable." "Warning"
    }
    
    Write-ColorOutput "Environment validation passed" "Success"
    return $true
}

function Import-EnvFile {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-ColorOutput "Environment file not found: $FilePath" "Error"
        return $false
    }
    
    Write-ColorOutput "Loading environment from $FilePath" "Info"
    
    Get-Content $FilePath | ForEach-Object {
        if ($_ -match '^([^#][^=]*?)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") {
                $value = $matches[1]
            }
            
            Set-Item -Path "env:$name" -Value $value
        }
    }
    
    return $true
}

# Show help if requested
if ($Help) {
    Show-Usage
    exit 0
}

# Change to project root
Set-Location $ProjectRoot

# Load environment file if specified
if ($EnvFile) {
    if (-not (Import-EnvFile $EnvFile)) {
        exit 1
    }
}

# Check dependencies if requested or always in dry-run mode
if ($CheckDeps -or $DryRun) {
    if (-not (Test-Dependencies)) {
        exit 1
    }
    
    if ($CheckDeps -and -not $DryRun) {
        exit 0
    }
}

# Validate environment
if (-not (Test-Environment)) {
    exit 1
}

# Build command arguments
$cmdArgs = @()

if ($ApiKey) {
    $cmdArgs += "--api-key", $ApiKey
}

if ($Host -ne "0.0.0.0") {
    $cmdArgs += "--host", $Host
}

if ($Port -ne 11434) {
    $cmdArgs += "--port", $Port
}

if ($LogLevel -ne "INFO") {
    $cmdArgs += "--log-level", $LogLevel
}

if ($ModelsFilter) {
    $cmdArgs += "--models-filter", $ModelsFilter
}

# Determine which entry point to use
if ($Dev) {
    $entryPoint = "ollama-proxy-dev"
    Write-ColorOutput "Starting in development mode..." "Info"
} elseif ($Daemon) {
    $entryPoint = "ollama-proxy-daemon"
    Write-ColorOutput "Starting in daemon mode..." "Info"
} else {
    $entryPoint = "ollama-proxy"
    Write-ColorOutput "Starting server..." "Info"
}

# Build final command
$finalCmd = @("uv", "run", $entryPoint) + $cmdArgs

# Show what would be executed
Write-ColorOutput "Command: $($finalCmd -join ' ')" "Info"
Write-ColorOutput "Host: $Host" "Info"
Write-ColorOutput "Port: $Port" "Info"
Write-ColorOutput "Log Level: $LogLevel" "Info"

if ($DryRun) {
    Write-ColorOutput "Dry run mode - not executing command" "Info"
    exit 0
}

# Execute the command
Write-ColorOutput "Starting ollama-proxy..." "Success"

try {
    & $finalCmd[0] $finalCmd[1..($finalCmd.Length-1)]
} catch {
    Write-ColorOutput "Failed to start ollama-proxy: $_" "Error"
    exit 1
}
