#!/bin/sh

select_flux_memory_profile() {
  memtotal_kib=${1:-$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)}

  # The midpoint keeps 64 GB systems on 4B and 128 GB systems on 9B.
  if [ "$memtotal_kib" -ge 100663296 ]; then
    echo 128gb
  else
    echo 64gb
  fi
}
