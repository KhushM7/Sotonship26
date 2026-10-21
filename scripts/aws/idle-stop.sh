#!/usr/bin/env bash
# Runs on the AWS instance every minute (fallsense-idle.timer, installed by setup.sh):
# powers it off, which EC2 turns into a stop (only the disk is billed), once nobody has
# had the site open for IDLE_MIN minutes, or once the server has been up MAX_HOURS
# whatever happens (a tab left open somewhere). The Cloudflare Worker (scripts/aws/wake)
# starts it again when a visitor presses "Wake the demo".
#
# Both clocks start when the fallsense service last started (a boot, a wake-up or a
# deploy), never at the instance's own boot: an instance up for days that has just been
# redeployed must not stop at once.
#
# Keep it awake while you work on it: touch /opt/fallsense/state/keep-awake
# (rm it when done).
set -uo pipefail

IDLE_S=$(( ${IDLE_MIN:-15} * 60 ))
MAX_S=$(( ${MAX_HOURS:-3} * 3600 ))
[ -e /opt/fallsense/state/keep-awake ] && exit 0

# seconds since the service started (0 if it has never started this boot)
boot_s=$(cut -d. -f1 /proc/uptime)
start_us=$(systemctl show fallsense -p ExecMainStartTimestampMonotonic --value)
[ -n "$start_us" ] && [ "$start_us" -gt 0 ] || exit 0
up=$(( boot_s - start_us / 1000000 ))

idle=$(curl -sf -m 5 localhost:8000/api/idle | grep -o '"idle_s":[0-9]*' | cut -d: -f2)
if [ -z "$idle" ]; then
    idle=$up  # server not answering: counts as idle since it started
fi
if [ "$idle" -ge "$IDLE_S" ] || [ "$up" -ge "$MAX_S" ]; then
    logger -t fallsense-idle "stopping the instance: idle ${idle}s, server up ${up}s"
    systemctl poweroff
fi
