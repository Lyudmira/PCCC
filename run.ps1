$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python -m pip install -r requirements.txt
python .\reproduce_pccc.py @args

