# Run fallsense on an AWS EC2 instance instead of this laptop, at the same address
# (https://fall-safe.bid): the instance runs the same Cloudflare named tunnel, with the
# token from configs/server.local.yaml. No port is open to the internet: SSH goes through
# an EC2 Instance Connect Endpoint (free) with your AWS login, from any network.
#
#   .\scripts\aws.ps1 create          # once: key pair, firewall, endpoint, instance, deploy
#   .\scripts\aws.ps1 deploy          # push this checkout + demo data (-Build: web app first)
#   .\scripts\aws.ps1 up              # start the instance; the server starts with it
#   .\scripts\aws.ps1 down            # stop it (only the disk is billed, ~$1.50/month)
#   .\scripts\aws.ps1 status | logs [-Follow] | ssh
#   .\scripts\aws.ps1 preview         # the instance's wall at http://localhost:8001 (SSH)
#   .\scripts\aws.ps1 destroy         # delete the instance, endpoint, bucket, firewall, key
#
# Only one server may run the tunnel at a time: `down` here before .\scripts\start.ps1,
# and stop start.ps1 before `up`/`deploy` (both refuse while port 8000 is in use).
# To try the instance while the laptop serves the site, deploy it with -NoTunnel and
# look at it with `preview`; `up -NoTunnel` then starts it without the check.
# Needs the AWS CLI signed in (aws login --profile fallsense, used when it exists;
# AWS_PROFILE overrides) and Windows OpenSSH.
param(
    [Parameter(Position = 0)]
    [ValidateSet("create", "deploy", "up", "down", "status", "logs", "ssh", "preview", "destroy")]
    [string]$Command = "status",
    [switch]$Build,
    [switch]$Follow,
    [switch]$NoTunnel
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$Name = "fallsense"
$Type = "t3.small"   # 2 vCPU, 2 GB: about $0.03/hour running, with its public IP
$DiskGB = 16
$Ami = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
$Key = Join-Path $HOME ".ssh\fallsense-aws.pem"
$KnownHosts = Join-Path $HOME ".ssh\fallsense_known_hosts"
$env:AWS_PAGER = ""
if (-not $env:AWS_PROFILE -and (aws configure list-profiles) -contains $Name) {
    $env:AWS_PROFILE = $Name
}
$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { aws configure get region }
if (-not $Region) { $Region = "eu-west-2" }
if (-not $env:FALL_DATA) { $env:FALL_DATA = "C:\fall-data" }

function Invoke-Aws {
    $out = & aws --region $Region @args
    if ($LASTEXITCODE) { throw "aws $($args[0]) $($args[1]) failed (see above)" }
    $out
}

function Invoke-Quiet {
    # a command allowed to fail, output discarded; Windows PowerShell 5.1 would
    # otherwise turn its redirected stderr into an error that stops the script
    $ErrorActionPreference = "Continue"
    $cmd, $rest = $args
    & $cmd @rest 2>&1 | Out-Null
}

function Get-Instance {
    $row = Invoke-Aws ec2 describe-instances --output text `
        --filters "Name=tag:Name,Values=$Name" `
        "Name=instance-state-name,Values=pending,running,stopping,stopped" `
        --query "Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]" |
        Select-Object -First 1
    if (-not $row) { return $null }
    $id, $state, $ip = $row -split "\s+"
    [pscustomobject]@{ Id = $id; State = $state; Ip = $(if ($ip -ne "None") { $ip }) }
}

function Get-RunningInstance {
    $i = Get-Instance
    if (-not $i) { throw "No $Name instance: run .\scripts\aws.ps1 create" }
    if ($i.State -ne "running") { throw "The instance is $($i.State): run .\scripts\aws.ps1 up" }
    $i
}

function Test-TunnelService {
    # `cloudflared service install` runs the tunnel at every boot, pointing at this
    # laptop: Cloudflare would send visitors there too (502 with no server running)
    $svc = Get-Service Cloudflared -ErrorAction SilentlyContinue
    $svc -and $svc.Status -eq "Running"
}

$TunnelServiceFix = "The Cloudflared Windows service is running the tunnel on this laptop. " +
    "In an Administrator PowerShell: Stop-Service Cloudflared; " +
    "Set-Service Cloudflared -StartupType Manual   (start.ps1 does not need it)"

function Assert-LaptopIdle {
    if ($NoTunnel) { return }
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        throw "A server is running on this laptop (port 8000). Stop start.ps1 first: " +
            "two servers on one tunnel would split visitors between them."
    }
    if (Test-TunnelService) { throw $TunnelServiceFix }
}

function Get-SecurityGroup {
    Invoke-Aws ec2 describe-security-groups --output text `
        --filters "Name=group-name,Values=$Name" --query "SecurityGroups[0].GroupId"
}

function Get-Bucket {
    # private (S3's default) and named after the account, so the name is unique
    $account = Invoke-Aws sts get-caller-identity --query Account --output text
    $bucket = "$Name-$account-$Region"
    Invoke-Quiet aws --region $Region s3api head-bucket --bucket $bucket
    if ($LASTEXITCODE) { Invoke-Aws s3 mb "s3://$bucket" | Out-Null }
    $bucket
}

function Get-Endpoint {
    Invoke-Aws ec2 describe-instance-connect-endpoints --output text `
        --filters "Name=tag:Name,Values=$Name" `
        --query "InstanceConnectEndpoints[?State!='delete-complete'].[InstanceConnectEndpointId,State]" |
        Select-Object -First 1
}

function Grant-Ssh([string]$id) {
    # SSH reaches the instance only through the endpoint, which shares its firewall
    # group: port 22 is allowed from the group itself, never from the internet
    if ($script:EndpointReady) { return }
    $sg = Get-SecurityGroup
    # already allowed -> InvalidPermission.Duplicate, which is fine
    Invoke-Quiet aws --region $Region ec2 authorize-security-group-ingress --group-id $sg `
        --protocol tcp --port 22 --source-group $sg
    if (-not (Get-Endpoint)) {
        $subnet = Invoke-Aws ec2 describe-instances --instance-ids $id --output text `
            --query "Reservations[0].Instances[0].SubnetId"
        Invoke-Aws ec2 create-instance-connect-endpoint --subnet-id $subnet `
            --security-group-ids $sg `
            --tag-specifications "ResourceType=instance-connect-endpoint,Tags=[{Key=Name,Value=$Name}]" |
            Out-Null
        Write-Host "Creating the SSH endpoint (a few minutes, once)..."
    }
    for ($i = 0; $i -lt 60; $i++) {
        $ep = Get-Endpoint
        if ($ep -match "create-complete") { $script:EndpointReady = $true; return }
        if ($ep -match "failed") { throw "SSH endpoint: $ep" }
        Start-Sleep -Seconds 10
    }
    throw "SSH endpoint not ready after 10 minutes: $(Get-Endpoint)"
}

function SshArgs([string]$id) {
    # forward slashes: Git's ssh, if it comes first on PATH, runs this through bash
    $aws = (Get-Command aws).Source -replace "\\", "/"
    "-i", $Key, "-o", "HostKeyAlias=$Name", "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=$KnownHosts", "-o", "ConnectTimeout=20",
    "-o", "ProxyCommand=$aws ec2-instance-connect open-tunnel --instance-id $id --region $Region",
    "-o", "LogLevel=ERROR", "ubuntu@$id"
}

function Invoke-Ssh([string]$id, [string]$cmd) {
    & ssh @(SshArgs $id) $cmd
    if ($LASTEXITCODE) { throw "ssh failed: $cmd" }
}

function Wait-Ssh([string]$id) {
    Grant-Ssh $id
    $ErrorActionPreference = "Continue"  # failed attempts are expected (see Invoke-Quiet)
    for ($i = 0; $i -lt 30; $i++) {
        & ssh @(SshArgs $id) true 2>&1 | Out-Null
        if (-not $LASTEXITCODE) { return }
        Start-Sleep -Seconds 5
    }
    throw "SSH to $id did not come up"
}

function Wait-Server([string]$id) {
    Write-Host "Waiting for the server..."
    $ok = & ssh @(SshArgs $id) "for i in `$(seq 60); do curl -sf -o /dev/null localhost:8000/api/public && echo up && exit; sleep 2; done"
    if (-not $ok) { throw "The server did not come up; check .\scripts\aws.ps1 logs" }
    if ($NoTunnel) {
        Write-Host "Server up (no tunnel). See it with .\scripts\aws.ps1 preview"
        return
    }
    Write-Host "Waiting for the tunnel..."
    # only this run of the service (its main PID), not one from before a redeploy
    $pid_ = "_PID=`$(systemctl show -p MainPID --value $Name)"
    for ($i = 0; $i -lt 30; $i++) {
        $line = & ssh @(SshArgs $id) "journalctl $pid_ --no-pager -o cat | grep 'tunnel ready' | tail -1"
        if ($line) { Write-Host "Live: $($line -replace '.*tunnel ready: ', '')"; return }
        Start-Sleep -Seconds 3
    }
    Write-Warning "No 'tunnel ready' yet; check .\scripts\aws.ps1 logs"
}

function New-Instance {
    if (Get-Instance) { Write-Host "Instance exists; deploying to it."; return }
    if (-not (Test-Path $Key)) {
        Invoke-Quiet aws --region $Region ec2 delete-key-pair --key-name $Name
        $pem = Invoke-Aws ec2 create-key-pair --key-name $Name --key-type ed25519 `
            --query KeyMaterial --output text
        New-Item -ItemType Directory -Force (Split-Path $Key) | Out-Null
        [IO.File]::WriteAllText($Key, (($pem -join "`n").Trim() + "`n"))
        icacls $Key /inheritance:r /grant:r "${env:USERNAME}:(R)" | Out-Null
        Write-Host "Key pair saved to $Key"
    }
    $sg = Get-SecurityGroup
    if (-not $sg -or $sg -eq "None") {
        $sg = Invoke-Aws ec2 create-security-group --group-name $Name `
            --description "fallsense: SSH via Instance Connect Endpoint only (web via Cloudflare tunnel)" `
            --query GroupId --output text
    }
    $image = Invoke-Aws ssm get-parameters --names $Ami --query "Parameters[0].Value" --output text
    $id = Invoke-Aws ec2 run-instances --image-id $image --instance-type $Type `
        --key-name $Name --security-group-ids $sg `
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DiskGB,VolumeType=gp3}" `
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$Name}]" `
        --query "Instances[0].InstanceId" --output text
    Write-Host "Launched $id ($Type, $Region); waiting for it to run..."
    Invoke-Aws ec2 wait instance-running --instance-ids $id
    if (Test-Path $KnownHosts) { Invoke-Quiet ssh-keygen -R $Name -f $KnownHosts }
}

function Publish {
    Assert-LaptopIdle
    $inst = Get-RunningInstance
    if ($Build -or -not (Test-Path "$root\frontend\dist\index.html")) {
        Push-Location "$root\frontend"
        try { npm install --no-fund --no-audit; npm run build } finally { Pop-Location }
    }
    $bundle = Join-Path $env:TEMP "fallsense-bundle.tar.gz"
    $setup = Join-Path $env:TEMP "fallsense-setup.sh"
    & "$root\.venv\Scripts\python.exe" "$root\scripts\aws\bundle.py" $bundle
    if ($LASTEXITCODE) { throw "bundle failed" }
    # bash needs LF line endings, whatever git checked out
    $sh = [IO.File]::ReadAllText("$root\scripts\aws\setup.sh") -replace "`r`n", "`n"
    [IO.File]::WriteAllText($setup, $sh)

    # the bundle goes through S3 (the SSH endpoint manages ~80 KB/s): private bucket,
    # a link that works for 15 minutes, and the copy deleted once the instance has it
    $bucket = Get-Bucket
    $object = "s3://$bucket/bundle.tar.gz"
    Invoke-Aws s3 cp $bundle $object --only-show-errors
    $url = Invoke-Aws s3 presign $object --expires-in 900

    Wait-Ssh $inst.Id
    $opts = SshArgs $inst.Id
    & scp @($opts[0..($opts.Count - 2)]) $setup "ubuntu@$($inst.Id):/tmp/"
    if ($LASTEXITCODE) { throw "upload failed" }
    $serveArgs = if ($NoTunnel) { "--no-tunnel" } else { "" }
    try {
        Invoke-Ssh $inst.Id ("curl -fsS -o /tmp/fallsense-bundle.tar.gz '$url' && " +
            "bash /tmp/fallsense-setup.sh /tmp/fallsense-bundle.tar.gz $serveArgs")
    } finally {
        Invoke-Quiet aws --region $Region s3 rm $object
    }
    Wait-Server $inst.Id
}

switch ($Command) {
    "create" { New-Instance; Publish }
    "deploy" { Publish }
    "up" {
        Assert-LaptopIdle
        $inst = Get-Instance
        if (-not $inst) { throw "No $Name instance: run .\scripts\aws.ps1 create" }
        if ($inst.State -ne "running") {
            Invoke-Aws ec2 start-instances --instance-ids $inst.Id | Out-Null
            Invoke-Aws ec2 wait instance-running --instance-ids $inst.Id
        }
        Wait-Ssh $inst.Id
        Wait-Server $inst.Id
    }
    "down" {
        $inst = Get-Instance
        if (-not $inst) { Write-Host "No $Name instance."; break }
        Invoke-Aws ec2 stop-instances --instance-ids $inst.Id | Out-Null
        Write-Host "Stopping $($inst.Id). .\scripts\start.ps1 can take over once it has stopped:"
        Invoke-Aws ec2 wait instance-stopped --instance-ids $inst.Id
        Write-Host "Stopped."
    }
    "status" {
        $inst = Get-Instance
        if (-not $inst) { Write-Host "No $Name instance."; break }
        Write-Host "$($inst.Id): $($inst.State) $($inst.Ip)"
        if ($inst.State -eq "running" -and (Test-TunnelService)) {
            Write-Warning "Visitors are split with this laptop. $TunnelServiceFix"
        }
        if ($inst.State -eq "running") {
            Grant-Ssh $inst.Id
            & ssh @(SshArgs $inst.Id) "systemctl is-active $Name; journalctl -u $Name -n 5 --no-pager -o cat"
        }
    }
    "logs" {
        $inst = Get-RunningInstance
        Grant-Ssh $inst.Id
        $f = if ($Follow) { "-f" } else { "-n 200 --no-pager" }
        & ssh -t @(SshArgs $inst.Id) "journalctl -u $Name $f -o short-iso"
    }
    "preview" {
        $inst = Get-RunningInstance
        Grant-Ssh $inst.Id
        # the endpoint closes connections after an hour; run preview again if it drops
        Write-Host "The instance's wall: http://localhost:8001 (Ctrl+C to close)"
        Start-Process "http://localhost:8001"
        & ssh -N -L 8001:localhost:8000 @(SshArgs $inst.Id)
    }
    "ssh" {
        $inst = Get-RunningInstance
        Grant-Ssh $inst.Id
        & ssh @(SshArgs $inst.Id)
    }
    "destroy" {
        $inst = Get-Instance
        $ok = Read-Host "Delete the $Name instance, its disk (incident log), SSH endpoint, firewall and key pair? (yes/no)"
        if ($ok -ne "yes") { break }
        if ($inst) {
            Invoke-Aws ec2 terminate-instances --instance-ids $inst.Id | Out-Null
            Invoke-Aws ec2 wait instance-terminated --instance-ids $inst.Id
        }
        $ep = Get-Endpoint
        if ($ep) {
            Invoke-Aws ec2 delete-instance-connect-endpoint `
                --instance-connect-endpoint-id ($ep -split "\s+")[0] | Out-Null
            while (Get-Endpoint) { Start-Sleep -Seconds 10 }  # the group is in use until then
        }
        $sg = Get-SecurityGroup
        if ($sg -and $sg -ne "None") { Invoke-Aws ec2 delete-security-group --group-id $sg }
        $account = Invoke-Aws sts get-caller-identity --query Account --output text
        Invoke-Quiet aws --region $Region s3 rb "s3://$Name-$account-$Region" --force
        Invoke-Quiet aws --region $Region ec2 delete-key-pair --key-name $Name
        Remove-Item $Key, $KnownHosts -Force -ErrorAction SilentlyContinue
        Write-Host "Deleted."
    }
}
