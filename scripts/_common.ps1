# _common.ps1
# Windows용 스크립트들이 공유하는 헬퍼: 프로젝트 경로, .env 로드,
# venv 파이썬 / quarto 실행 파일 탐색.
#
# 각 스크립트에서 dot-source 해서 사용한다:  . "$PSScriptRoot\_common.ps1"

$ProjectDir = Split-Path -Parent $PSScriptRoot

# .env 가 있으면 KEY=VALUE 를 현재 프로세스 환경변수로 읽어들인다.
function Import-DotEnv {
    $envFile = Join-Path $ProjectDir '.env'
    if (-not (Test-Path $envFile)) { return }

    foreach ($line in Get-Content $envFile) {
        $trimmed = $line.Trim()
        if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }

        $pair = $trimmed -split '=', 2
        if ($pair.Count -ne 2) { continue }

        # 앞의 [char]0xFEFF 는 .env 가 UTF-8 BOM 으로 저장된 경우의 BOM 제거용
        $key = $pair[0].Trim().Trim([char]0xFEFF)
        $value = $pair[1].Trim().Trim('"').Trim("'")
        if ($value -ne '') { Set-Item -Path "Env:$key" -Value $value }
    }
}

# venv 파이썬 경로. Windows venv 는 bin/ 이 아니라 Scripts/ 에 생성된다.
function Get-VenvPython {
    $python = Join-Path $ProjectDir 'venv\Scripts\python.exe'
    if (Test-Path $python) { return $python }

    throw @"
venv 파이썬이 없습니다: $python
먼저 가상환경을 만드세요:
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt
  venv\Scripts\python -m ipykernel install --user --name ai-ml-study
"@
}

# quarto 가 PATH 에 있는지 확인하고, 없으면 기본 설치 경로를 PATH 앞에 붙인다
# (winget 설치 직후 열려 있던 터미널에는 PATH가 아직 반영되지 않는다).
#
# quarto.cmd 를 전체 경로로 직접 호출하면 "C:\Program Files\..." 의 공백에서
# 경로가 잘려 깨지므로, 반드시 PATH 에 올려두고 quarto 로 호출해야 한다.
function Initialize-Quarto {
    if (Get-Command quarto -ErrorAction SilentlyContinue) { return }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Quarto\bin'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Quarto\bin')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'quarto.cmd')) {
            $env:PATH = "$candidate;$env:PATH"
            return
        }
    }

    throw @"
quarto 를 찾을 수 없습니다. 설치하세요:
  winget install --id Posit.Quarto -e
(이미 설치했다면 새 터미널을 열어야 PATH에 반영됩니다)
"@
}

# quarto 를 실행하고 종료 코드로만 성공 여부를 판단한다.
#
# quarto 는 진행 상황을 stderr 로 출력하는데, PowerShell 5.1 은
# ErrorActionPreference='Stop' 상태에서 네이티브 명령의 stderr 를
# 종료 오류로 바꿔버린다(실제로는 성공해도 실패로 보임). 그래서 호출 동안만
# 완화한 뒤 $LASTEXITCODE 로 판단한다.
function Invoke-Quarto {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$QuartoArgs
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        quarto @QuartoArgs
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }

    if ($exitCode -ne 0) {
        throw "quarto $($QuartoArgs -join ' ') 실패 (exit $exitCode)"
    }
}
