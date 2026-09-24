set -e
export F50_ENGINE_DIR=/tmp/claude-0/-home-user-seiranwind-tech/a678a4e4-1091-5af0-b47e-7a1f1a816968/scratchpad/eng/F50_Compoud_eye_G3_v0.1.5/legacy/V39R3_FROZEN F50_VEC=/tmp/claude-0/-home-user-seiranwind-tech/a678a4e4-1091-5af0-b47e-7a1f1a816968/scratchpad/glove300.npy F50_WORDS=/tmp/claude-0/-home-user-seiranwind-tech/a678a4e4-1091-5af0-b47e-7a1f1a816968/scratchpad/glove_words.txt OMP_NUM_THREADS=4
TP=/tmp/claude-0/-home-user-seiranwind-tech/a678a4e4-1091-5af0-b47e-7a1f1a816968/scratchpad/tapes; mkdir -p runs
for t in tape_0_1357_11 tape_1_1357_12 tape_0_9753_21 tape_2_1357_13_zo tape_1_9753_22_zo; do
  python w11_replay.py $TP/$t.npz > runs/replay_$t.json
  python w11_river.py $TP/$t.npz > runs/river_$t.json
  python w11_macro_events.py $TP/$t.npz 12 > runs/macro_events_$t.json
done
python w11_ar.py $TP/tape_0_1357_11.npz $TP/tape_1_1357_12.npz $TP/tape_0_9753_21.npz $TP/tape_2_1357_13_zo.npz $TP/tape_1_9753_22_zo.npz > runs/ar_law.json
for a in "tape_2_1357_13_zo 2 1357 13" "tape_1_9753_22_zo 1 9753 22"; do set -- $a
  python w11_wave_drive.py $TP/$1.npz 0.02 > runs/wave_drive_$1.json
  python w11_meanfield.py $TP/$1.npz $2 $3 $4 > runs/meanfield_$1.json
  python w11_balance.py $TP/$1.npz $2 $3 $4 > runs/balance_$1.json
  python w11_slaved.py $TP/$1.npz $2 $3 $4 25,100,400 | tail -n +1 > runs/slaved_$1.txt
  python w11_rollout.py $TP/$1.npz $2 $3 $4 > runs/rollout_$1.json
done

for a in "tape_2_1357_13_zo 2 1357 13" "tape_1_9753_22_zo 1 9753 22"; do set -- $a
  python w11_decouple.py $TP/$1.npz $2 $3 $4 > runs/decouple_$1.json
  python w11_subbunch.py $TP/$1.npz $2 $3 $4 1,2,3,20 > runs/env_representation_$1.json
  python w11_shape_pc.py $TP/$1.npz $2 $3 $4 > runs/shape_pc_$1.json
done
echo ALLDONE2
