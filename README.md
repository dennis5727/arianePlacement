# LLM-Guided Chip Placement (Kaggle)

Lightweight LLM loop guiding MaskPlace's greedy engine for chip macro placement.
Mode A: training-free, wiremask argmin. No RL, no neural network.

## Structure
- `maskplace/` — all code and data
  - `ariane/` — Ariane RISC-V benchmark (netlist.pb.txt)
  - `place_db.py`, `place_db_proto.py`, `prim.py` — PlaceDB loader (from MaskPlace)
  - `place_env/place_env.py` — placement environment (from MaskPlace)
  - `comp_res.py` — HPWL computation (from MaskPlace)
  - `greedy_place.py` — [OURS] training-free greedy placer
  - `region_constraint.py` — [OURS] region → grid mask
  - `parse_netlist.py` — [OURS] netlist → text summary for LLM
  - `history_tracker.py` — [OURS] iteration log + early stopping
  - `llm_interface.py` — [OURS] LLM prompt + API call + JSON parse
  - `llm_guided_placement.py` — [OURS] main loop
  - `visualize.py` — [OURS] placement image renderer
  - `evaluate.py` — [OURS] baselines + comparison

## Running on Kaggle
See KAGGLE_NOTEBOOK_SETUP.md for dependency installation and notebook setup.

## Reference
Base engine: MaskPlace (Lai et al., NeurIPS 2022)
Project plan: LLM_Chip_Placement_Project_Plan_v3.md
