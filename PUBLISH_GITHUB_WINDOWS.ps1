# Continual AI Lab - safe and repeatable GitHub publisher (Windows PowerShell 5.1+).
# Publishes only this folder to the designated repository under RV-crtl.
# NEVER uploads data/local or modifies another repository.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath $PSScriptRoot

$ExpectedOwner = 'RV-crtl'
$RepositoryName = 'continual-ai-lab'
$FullName = "$ExpectedOwner/$RepositoryName"
$RepositoryURL = "https://github.com/$FullName"

function Run-Git {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: git $($Arguments -join ' ')" }
}

Write-Host '=== Continual AI Lab: GitHub publisher ===' -ForegroundColor Cyan
Write-Host 'Local AI setup and GitHub publishing are TWO DIFFERENT steps.'
Write-Host "Target: $FullName (public or private depending on existing repo)" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath 'continual_ai/cli.py') -or
    -not (Test-Path -LiteralPath '.github/workflows/online-learning.yml') -or
    -not (Test-Path -LiteralPath 'README.md')) {
    throw 'Wrong folder. Put this script in the extracted Continual-AI-Starter folder (beside README.md) and rerun.'
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is missing. Install from https://git-scm.com/downloads/win and open a new PowerShell.'
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is missing. Install from https://cli.github.com/ and open a new PowerShell.'
}

& gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub CLI is not signed in. Run: gh auth login  (choose GitHub.com, HTTPS, browser login).'
}

# Crucial: the GUI GitHub connection and the CLI can point to DIFFERENT accounts.
$WhoAmI = (& gh api user --jq '.login' | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($WhoAmI)) {
    throw 'Could not verify the account authenticated in GitHub CLI.'
}
$WhoAmI = $WhoAmI.Trim()
Write-Host "Authenticated GitHub CLI account: $WhoAmI"
if ($WhoAmI -ine $ExpectedOwner) {
    throw "Wrong GitHub CLI account ($WhoAmI). This publisher requires $ExpectedOwner. Switch accounts using 'gh auth switch' or 'gh auth login' and try again."
}

if (-not (Test-Path -LiteralPath '.git')) {
    Write-Host 'Initialising a new Git repository in the project folder...'
    Run-Git -Arguments @('init','-b','main')
}
$GitRoot = (& git rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Git worktree.' }
$RootFull = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\','/')
$GitFull = [System.IO.Path]::GetFullPath($GitRoot.Trim()).TrimEnd('\','/')
if ($RootFull -ine $GitFull) { throw "This folder belongs to a different Git repository: $GitFull. Aborting to protect unrelated projects." }
$Branch = (& git branch --show-current | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or $Branch.Trim() -ne 'main') {
    throw "Expected project branch main but found '$Branch'. Aborting to avoid pushing a different branch."
}

# A whitelist prevents user chats, databases, model weights, and other local data from being pushed.
$PublishPaths = @(
    '.github', 'continual_ai', 'tests', 'data/cloud/.gitkeep', 'data/local/.gitkeep',
    'requirements-train.txt', '.gitignore', 'README.md',
    'INSTALL_WINDOWS.ps1', 'CHECK_HARDWARE_WINDOWS.ps1',
    'START_CHAT_WINDOWS.cmd', 'START_LEARNING_WINDOWS.cmd',
    'START_DASHBOARD_WINDOWS.cmd', 'START_ALL_WINDOWS.cmd',
    'PUBLISH_GITHUB_WINDOWS.ps1', 'PUBLISH_GITHUB_WINDOWS.cmd'
)
$AvailablePaths = @($PublishPaths | Where-Object { Test-Path -LiteralPath $_ })
& git add -- @AvailablePaths
if ($LASTEXITCODE -ne 0) { throw 'Could not stage project files for GitHub.' }
& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    Run-Git -Arguments @('-c','user.name=Continual AI Setup','-c','user.email=setup@users.noreply.github.com',
              'commit','-m','Initialise/update independent continual AI experiment')
}

$RemoteNames = @(& git remote)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Git remotes.' }
$OriginExists = ($RemoteNames -contains 'origin')
$Origin = ''
if ($OriginExists) {
    $Origin = (& git remote get-url origin | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0) { throw 'Could not read Git origin.' }
}
if ($OriginExists) {
    $CleanOrigin = $Origin.Trim()
    $AllowedOrigins = @("https://github.com/$FullName.git", "https://github.com/$FullName", "git@github.com:$FullName.git")
    if ($CleanOrigin -notin $AllowedOrigins) {
        throw "The existing origin points to $CleanOrigin, NOT $FullName. Aborting without modifying it."
    }
}

# Check repo existence. Failure here can also be private-repo denial or network errors;
# gh repo create will fail safely rather than touch an unrelated repository.
$RepoInfo = $null
try {
    $RepoInfo = (& gh repo view $FullName --json nameWithOwner --jq '.nameWithOwner' 2>$null | Select-Object -First 1)
} catch { $RepoInfo = $null }
$Exists = ($LASTEXITCODE -eq 0 -and $RepoInfo -eq $FullName)

if (-not $Exists) {
    if ($OriginExists) {
        throw "Origin is already configured but $FullName is not accessible. Check GitHub permissions before retrying."
    }
    Write-Host "Creating NEW private repository: $FullName" -ForegroundColor Cyan
    & gh repo create $FullName --private --source . --remote origin --push --description 'Local and GitHub-hosted verified continual learning research prototype'
    if ($LASTEXITCODE -ne 0) {
        throw "Repository creation or push failed. Check gh auth status, then run: gh repo view $FullName"
    }
} else {
    Write-Host "Repository already exists: $FullName" -ForegroundColor Yellow
    if (-not $OriginExists) {
        Run-Git -Arguments @('remote','add','origin',"https://github.com/$FullName.git")
    }
    # Fast-forward only: no force-push or rewrite of remote history.
    Run-Git -Arguments @('push','-u','origin','main')
}

# An authenticated remote read is stronger evidence than a local success message.
& gh repo view $FullName --json url,visibility --jq '"URL: " + .url + " | Visibility: " + .visibility'
if ($LASTEXITCODE -ne 0) { throw 'Push finished but GitHub could not verify the repository. Review gh auth status.' }
Write-Host "SUCCESS: $RepositoryURL" -ForegroundColor Green
Write-Host "Actions: $RepositoryURL/actions"
Write-Host "Online workflow: $RepositoryURL/actions/workflows/online-learning.yml"
Write-Host ''
Write-Host 'To start the first hosted learning run (may use billable GitHub Actions minutes):'
Write-Host "  gh workflow run online-learning.yml --repo $FullName"
Write-Host "Then check: gh run list --repo $FullName --limit 5"
Write-Host 'Future runs are scheduled by .github/workflows/online-learning.yml; GitHub-hosted jobs are not continuously running.'
