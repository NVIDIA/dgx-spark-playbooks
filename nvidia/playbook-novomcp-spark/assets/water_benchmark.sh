#!/bin/bash
# SPC/E water MD scaling benchmark on GB10. Runs inside the gromacs-spark image.
# Per size: solvate -> steepest-descent EM (flexible water) -> production MD on GPU.
# ns/day is read from the production run. Energy minimization first is required —
# a freshly tiled water box has close contacts and MD NaNs at step 0 without it.
set -e
NT=${NT:-16}

# SIZE_nm  NSTEPS  TAG  EMSTEPS
LADDER=(
  "10   25000 s1 500"
  "21.5 5000  s2 500"
  "34   2000  s3 300"
  # "66 300   s4 150"   # ~28.3M atoms, memory-flex run (peak ~69 GB, slow). Uncomment to try.
)

for row in "${LADDER[@]}"; do
  set -- $row; SIZE=$1; NSTEPS=$2; TAG=$3; EMS=$4

  cat > em_$TAG.mdp <<EOF
integrator    = steep
nsteps        = $EMS
emtol         = 1000
emstep        = 0.01
define        = -DFLEXIBLE
cutoff-scheme = Verlet
nstlist       = 40
coulombtype   = PME
rcoulomb      = 1.0
rvdw          = 1.0
EOF

  cat > md_$TAG.mdp <<EOF
integrator    = md
nsteps        = $NSTEPS
dt            = 0.002
cutoff-scheme = Verlet
nstlist       = 40
coulombtype   = PME
rcoulomb      = 1.0
rvdw          = 1.0
constraints   = h-bonds
tcoupl        = v-rescale
tc-grps       = System
tau-t         = 0.1
ref-t         = 300
nstlog        = 0
nstenergy     = 0
nstxout       = 0
EOF

  cp topol_template.top topol_$TAG.top
  gmx solvate -cs spc216.gro -box $SIZE $SIZE $SIZE -o box_$TAG.gro -p topol_$TAG.top > solv_$TAG.log 2>&1
  ATOMS=$(sed -n 2p box_$TAG.gro | tr -d ' ')
  gmx grompp -f em_$TAG.mdp -c box_$TAG.gro -p topol_$TAG.top -o em_$TAG.tpr -maxwarn 10 > grompp_em_$TAG.log 2>&1
  gmx mdrun -deffnm em_$TAG -ntmpi 1 -ntomp "$NT" -nb gpu > mdrun_em_$TAG.log 2>&1
  gmx grompp -f md_$TAG.mdp -c em_$TAG.gro -p topol_$TAG.top -o md_$TAG.tpr -maxwarn 10 > grompp_$TAG.log 2>&1
  gmx mdrun -deffnm md_$TAG -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp "$NT" -pin on -notunepme > mdrun_$TAG.log 2>&1

  PERF=$(grep -iE "Performance:" md_$TAG.log | tail -1 | awk '{print $2}')
  echo "atoms=$ATOMS  ns/day=$PERF"
done
