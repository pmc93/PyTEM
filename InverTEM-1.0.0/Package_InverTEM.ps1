# Packs InverTEM.exe with its deployed Qt DLLs, plugins, icon and the MSVC
# runtime into one portable InverTEM_portable.exe using Enigma Virtual Box
# (free, https://enigmaprotector.com/en/downloads.html). The packed files are
# read from inside the exe; invertem_solver.txt, projects and reports are still
# written as normal files next to it.
param(
    [Parameter(Mandatory = $true)][string]$AppExe,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"

$appDir = Split-Path -Parent (Resolve-Path $AppExe)
if (-not $Output) { $Output = Join-Path $appDir "InverTEM_portable.exe" }

# Runtime files only: DLLs and the icon beside the exe, plus the windeployqt
# plugin folders (those holding DLLs). OpenGL/Direct3D fallbacks are not needed
# by this widgets-only (raster) application.
$skip = @("opengl32sw.dll", "d3dcompiler_47.dll", "dxcompiler.dll", "dxil.dll")
$rootFiles = @(Get-ChildItem $appDir -File | Where-Object {
    ($_.Extension -eq ".dll" -and $skip -notcontains $_.Name.ToLower()) -or $_.Name -eq "g4.png" })
$pluginDirs = @(Get-ChildItem $appDir -Directory | Where-Object {
    $_.Name -notmatch '^(CMakeFiles|Testing|\.qt|.*_autogen)$' -and
    (Get-ChildItem $_.FullName -Recurse -File -Filter *.dll | Select-Object -First 1) })

# Microsoft C++ and OpenMP runtimes, so the target PC needs no redistributable:
# from the Visual Studio redistributable folder, else from this PC's System32.
if ($env:VCToolsRedistDir) {
    $rootFiles += @(Get-ChildItem (Join-Path $env:VCToolsRedistDir "x64") -Directory |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.(CRT|OpenMP)$' } |
        ForEach-Object { Get-ChildItem $_.FullName -File -Filter *.dll })
} elseif ($env:SystemRoot) {
    $rootFiles += @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll",
                    "msvcp140_2.dll", "vcomp140.dll") |
        ForEach-Object { Join-Path $env:SystemRoot "System32\$_" } |
        Where-Object { Test-Path $_ } | ForEach-Object { Get-Item $_ }
}

$console = @($env:ENIGMA_VB_CONSOLE,
             "$env:ProgramFiles\Enigma Virtual Box\enigmavbconsole.exe",
             "${env:ProgramFiles(x86)}\Enigma Virtual Box\enigmavbconsole.exe") |
    Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $console) {
    Write-Host "Enigma Virtual Box was not found, so no portable single-file exe was made."
    Write-Host "Install it from https://enigmaprotector.com/en/downloads.html (or set ENIGMA_VB_CONSOLE)."
    exit 0
}

function FileNode([System.IO.FileInfo]$f) {
    "<File><Type>2</Type><Name>$([Security.SecurityElement]::Escape($f.Name))</Name>" +
    "<File>$([Security.SecurityElement]::Escape($f.FullName))</File><ActiveX>false</ActiveX>" +
    "<ActiveXInstall>false</ActiveXInstall><Action>0</Action><OverwriteDateTime>false</OverwriteDateTime>" +
    "<OverwriteAttributes>false</OverwriteAttributes><PassCommandLine>false</PassCommandLine>" +
    "<HideFromDialogs>0</HideFromDialogs></File>"
}
function FolderNode([string]$name, [string]$children) {
    "<File><Type>3</Type><Name>$([Security.SecurityElement]::Escape($name))</Name><Action>0</Action>" +
    "<OverwriteDateTime>false</OverwriteDateTime><OverwriteAttributes>false</OverwriteAttributes>" +
    "<HideFromDialogs>0</HideFromDialogs><Files>$children</Files></File>"
}
function DirectoryNode([System.IO.DirectoryInfo]$d) {
    $children = (@(Get-ChildItem $d.FullName -Directory | ForEach-Object { DirectoryNode $_ }) +
                 @(Get-ChildItem $d.FullName -File | ForEach-Object { FileNode $_ })) -join ""
    FolderNode $d.Name $children
}

$seen = @{}
$contents = (@($rootFiles | Where-Object { -not $seen.ContainsKey($_.Name.ToLower()) -and ($seen[$_.Name.ToLower()] = $true) } |
                ForEach-Object { FileNode $_ }) +
             @($pluginDirs | ForEach-Object { DirectoryNode $_ })) -join ""
$project = @"
<?xml version="1.0" encoding="windows-1252"?>
<>
  <InputFile>$([Security.SecurityElement]::Escape((Resolve-Path $AppExe).Path))</InputFile>
  <OutputFile>$([Security.SecurityElement]::Escape($Output))</OutputFile>
  <Files>
    <Enabled>true</Enabled>
    <DeleteExtractedOnExit>false</DeleteExtractedOnExit>
    <CompressFiles>true</CompressFiles>
    <Files>$(FolderNode "%DEFAULT FOLDER%" $contents)</Files>
  </Files>
  <Registries><Enabled>false</Enabled></Registries>
  <Packaging><Enabled>false</Enabled></Packaging>
  <Options>
    <ShareVirtualSystem>false</ShareVirtualSystem>
    <MapExecutableWithTemporaryFile>true</MapExecutableWithTemporaryFile>
    <AllowRunningOfVirtualExeFiles>true</AllowRunningOfVirtualExeFiles>
  </Options>
</>
"@
$projectFile = Join-Path $appDir "InverTEM_portable.evb"
[IO.File]::WriteAllText($projectFile, $project, [Text.Encoding]::GetEncoding(1252))

Write-Host "Packing $($rootFiles.Count) runtime files and $($pluginDirs.Count) plugin folders with Enigma Virtual Box..."
& $console $projectFile
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Output)) { throw "Enigma Virtual Box did not produce $Output" }
Write-Host "Portable executable: $Output ($([math]::Round((Get-Item $Output).Length / 1MB, 1)) MB)"
