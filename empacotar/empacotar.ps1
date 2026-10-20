# Gera dist\Transcricao\Transcricao.exe, instala em %LOCALAPPDATA%\Programs\Transcricao
# e cria os atalhos no Menu Iniciar e na area de trabalho.
# Uso: powershell -ExecutionPolicy Bypass -File empacotar\empacotar.ps1
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

if (-not (Test-Path "empacotar\icone.ico")) {
    uv run python empacotar\gerar_icone.py empacotar\icone.ico
}

# Com --specpath build, o PyInstaller resolve caminho relativo a partir de build\; por isso o absoluto.
$icone = Join-Path $raiz "empacotar\icone.ico"
uv run pyinstaller --noconfirm --clean --windowed --onedir `
    --name Transcricao `
    --icon $icone `
    --add-data "$icone;." `
    --collect-data faster_whisper `
    --collect-binaries ctranslate2 `
    --collect-binaries sherpa_onnx `
    --collect-all _soundfile_data `
    --distpath dist --workpath build --specpath build `
    empacotar\iniciar.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou com codigo $LASTEXITCODE" }

$destino = Join-Path $env:LOCALAPPDATA "Programs\Transcricao"
if (Get-Process Transcricao -ErrorAction SilentlyContinue) {
    throw "Feche o app Transcricao antes de instalar."
}
if (Test-Path $destino) { Remove-Item $destino -Recurse -Force }
Copy-Item "dist\Transcricao" $destino -Recurse

# "Transcricao de reuniao" com cedilha e til, montado por codigo de caractere
# porque o PowerShell 5 le este arquivo como cp1252.
$nome = "Transcri" + [char]0x00E7 + [char]0x00E3 + "o de reuni" + [char]0x00E3 + "o"
$exe = Join-Path $destino "Transcricao.exe"
$shell = New-Object -ComObject WScript.Shell
$pastas = @(
    [Environment]::GetFolderPath("Programs"),
    [Environment]::GetFolderPath("Desktop")
)
foreach ($pasta in $pastas) {
    $atalho = $shell.CreateShortcut((Join-Path $pasta "$nome.lnk"))
    $atalho.TargetPath = $exe
    $atalho.WorkingDirectory = $destino
    $atalho.IconLocation = "$exe,0"
    $atalho.Save()
    Write-Output "atalho: $(Join-Path $pasta "$nome.lnk")"
}
Write-Output "instalado: $exe"
