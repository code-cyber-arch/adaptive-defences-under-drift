# Phase 04: train the RL controllers

Both controllers choose continue learning or reset. The alarm controller has two states. The alarm_persistence controller also observes recent error deterioration and has four states. Both use the same accuracy reward and screening rule.

Use results/training/04_rl_training/alarm/ or alarm_persistence/, then seed_7, seed_17 or seed_27. Each folder contains q_table.json, policy.csv (actions, values and visit counts), episodes.csv and transitions.parquet.

Training uses separate SEA and RBF seed 112. Both representations use the same conditions and episode order for a matching RL seed. Validation uses seed 113. A fixed four-pass training budget is not a claim of convergence. Evaluation never changes the Q-table.
