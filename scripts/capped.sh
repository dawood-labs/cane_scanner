#!/bin/bash
# Run a command and kill it if the memory of its whole process tree exceeds LIMIT_MB,
# so a blow-up can never reach the kernel OOM killer and take the session with it.
# On WSL an OOM takes the whole WSL instance down, not just the one process.
#
#   LIMIT_MB=10000 bash scripts/capped.sh python3 scripts/run_full_aoi.py --stage label ...
#
# Exits 99 and prints WATCHDOG if it had to kill; always prints PEAK_RSS_MB, which is
# the number to read before raising --jobs, --label-jobs or --tile-km.
#
# The first version summed nothing but the launched process. Everything heavy here runs
# in ProcessPoolExecutor children, so it reported near zero while the machine filled up,
# and the kernel got there first. It has to walk the tree.
LIMIT_MB=${LIMIT_MB:-10000}
"$@" &
PID=$!

# Every process whose parent chain reaches PID. Printed as "pid rss_kb" lines so the
# same walk serves both the accounting and the kill: `pkill -P` reaches only direct
# children, and the first time this fired it left two grandchildren holding 1.5 GB each.
tree_procs() {
  awk -v root="$1" '
    FILENAME ~ /\/stat$/ {
      line = $0
      sub(/^[0-9]+ \(.*\) [A-Za-z] /, "", line)
      split(line, f, " ")
      split(FILENAME, p, "/")
      ppid[p[3]] = f[1]
      pids[p[3]] = 1
    }
    FILENAME ~ /\/status$/ && /^VmRSS:/ {
      split(FILENAME, p, "/")
      rss[p[3]] = $2
    }
    END {
      for (pid in pids) {
        cur = pid
        for (hops = 0; hops < 40 && cur != "" && cur != "1"; hops++) {
          if (cur == root) { printf "%s %d\n", pid, rss[pid]; break }
          cur = ppid[cur]
        }
      }
    }
  ' /proc/[0-9]*/stat /proc/[0-9]*/status 2>/dev/null
}

tree_rss() { tree_procs "$1" | awk '{t += $2} END {printf "%d\n", t / 1024}'; }

PEAK=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(tree_rss $PID)
  [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ -n "$RSS" ] && [ "$RSS" -gt "$LIMIT_MB" ]; then
    echo "WATCHDOG: tree RSS ${RSS}MB exceeded ${LIMIT_MB}MB, killing the tree under $PID"
    for victim in $(tree_procs $PID | awk '{print $1}'); do
      kill -9 "$victim" 2>/dev/null
    done
    kill -9 $PID 2>/dev/null
    wait $PID 2>/dev/null
    echo "PEAK_RSS_MB=$PEAK"
    exit 99
  fi
  sleep 1
done
wait $PID; RC=$?
echo "PEAK_RSS_MB=$PEAK"
exit $RC
