# Extra — Is Cosmos-Embed1 better than InternVideo2 for actions?

## OMNI-X14 · SM-1 with Cosmos-Embed1 as the clip encoder

### Why this ticket exists

The PFM trains on clip vectors (OMNI-902). Changing the clip encoder after OMNI-902 means encoding every
training window again, so this is the cheapest moment to check whether InternVideo2, the weak side on
action queries (K-30), should be replaced.

### What was done

2026-11-23, part 1: the code. The Colab run is next.

- **Checked the model card first** (huggingface.co/nvidia/Cosmos-Embed1-336p, commit `0e8a28f7`,
  2026-03-13). It confirms: 768-D unit vectors, exactly 8 frames, 336 x 336 (stretched, not cropped), BF16,
  1,196,926,328 parameters, NVIDIA Open Model License, `trust_remote_code`. Kinetics-400 (F1): Cosmos-Embed1-336
  87.66 against **InternVideo2-1B-S2** 62.80
- **Two things the ticket did not say,** found while checking:
  - The card's InternVideo2 is the **1B** model. Ours is `InternVideo2_CLIP_S`, a small one (D-33). So Cosmos
    is not "the same size class": it is a much bigger model. This matters for the laptop CPU check below
  - The card says Cosmos is "optimized for 8 frames, sampled at 1-2 frames per second", 4 to 8 s of video.
    Our clip is the window `[t - 2Δ, t]` with Δ = 0.5 s: 8 frames in 1 s
  - Ahmad updated the ticket after both notes: the size claim corrected, and Cosmos tested in **two spacings**,
    A (8 frames over 1 s, Algorithm 2's window) and B (8 frames over the 4 s before `t`, as Cosmos was trained),
    the rule applied to the better one, and the GPU time per clip recorded for both models
- `scripts/encode_virat_cosmos.py`: reads each video's kept frames and clip windows from the InternVideo2
  archive (`embeddings/virat/`), decodes the video again, takes the same 8 frames
  (`clip_encoder.window_span`, `uniform_indices`), and encodes them with Cosmos; for B, the 4 s up to `t` instead, cut at 0 s at a video's start
  (`SPACINGS`, `clip_indices`). Each spacing has its own folder and the manifest refuses a mix. The SigLIP 2 frame vectors
  are copied unchanged, so `measure_sm1.load_archive` reads the new archive like the old one. Resumable, with a
  manifest per video holding the GPU time per clip. It refuses a video whose frame rate or frame count differs
  from the archive's. `tune()` makes D-40's choice again for Cosmos, with `tune_fusion.grid` on the tuning
  queries only
- `scripts/measure_sm1.py`: `measure` takes `weights` and `depth` for `fused`. The defaults are the product's
  (D-40), so InternVideo2's run is unchanged
- **GPU time, same conditions for both models:** OMNI-706's InternVideo2 number (297-319 ms per clip, L4,
  float32) was measured in another session and another way, so the notebook times InternVideo2 (1 s, float32,
  as the archive ran) and Cosmos (A and B, BF16) on the **same 64 clips** of one video, batches of 16,
  preparation included (`sample_clips`, `time_per_clip`), and turns each into cameras per GPU from the archive's
  kept frames per second of footage (clip stream only)
- `notebooks/cosmos_sm1_gpu.ipynb`: clone the branch, models, timing, encode A then B, chance check, tune per
  spacing, SM-1 per spacing, the better spacing (written in the code before the run: higher fused overall, then
  higher fused on actions, then A), the rule, save to `results/sm1_result_cosmos.json`

### Verification performed

- `scripts/tests/test_encode_virat_cosmos.py` (stand-in encoders, a synthetic video archived by OMNI-708's own
  runner): every archived frame gets the same 8 frames in the same order, every other field is kept, a rerun
  skips finished videos, mixing pins is refused, and a video that no longer matches its archive is recorded as
  failed; spacing B spreads the 8 frames over the 4 s up to each frame (a 7 s video, so late frames are not cut
  at 0 s); a folder holds one spacing; the timing sample is the runner's own first clips
- `scripts/tests/test_measure_sm1.py`: a new test shows `weights` and `depth` reach `fused` only. 30 + 6 tests pass

- **2026-11-23, part 2: Cosmos does not run on the repo's transformers.** Ahmad's first Colab run stopped in
  step 2: Cosmos's `modeling_qformer.py` imports `find_pruneable_heads_and_indices`, which transformers 5 removed.
  Checked every `transformers` import of Cosmos's seven code files at the pinned revision against 5.20.1: that one
  name, plus two model methods its Q-Former calls on every forward (`get_head_mask`, `invert_attention_mask`).
  Restoring the three with transformers 4.x's code then failed deeper, inside transformers 5's own weight setup
  (`all_tied_weights_keys`), building the full model with random weights on the laptop's CPU. Patching transformers
  5's internals was dropped
- **Two processes instead.** The same CPU check with transformers 4.51.3 (the version Cosmos's config was saved
  with) in an environment of its own: the full model built (1,173,904,191 parameters, random weights, BF16), one clip
  gave a 768-D unit vector (326 s on the CPU) and two texts unit vectors: the code path runs. So Cosmos runs in a
  process of its own on Colab, with transformers 4.51.3 installed `--no-deps` into `/content/cosmos_env` and first
  on its path (`encode_virat_cosmos.py cosmos`): the 100 queries' text vectors (`texts.npz`), its GPU time in both
  spacings, both archives. The notebook's process keeps 5.20.1 for SigLIP 2 and InternVideo2, so the query vectors
  on the frame side are made exactly as in SM-1's run, times InternVideo2 on the same 64 clips, and reads Cosmos's
  text vectors from the file (`StoredTexts`, which refuses a text it was not given). A new test covers the hand-over

- **2026-11-23, part 3: the run.** Ahmad's run on an A100, both spacings, each about 2 hours of encoding
  (`sm1_result_cosmos.json` on Drive). Both spacings' tuning chose D-40's own setting (depth 5, `w_f` action 0.6,
  scene 0.5). On SM-1: fused 74.0% for A and B, 60.0% on actions, the same as frame only. The rule says keep (D-50)
- Looked further than the rule, per query, against `sm1_result_tuned.json`: Cosmos's clip alone finds 9 queries
  InternVideo2's misses, and 4 action queries frame search misses (`a04`, `a05` closing a trunk, `a16` waving,
  `a25` exiting a building). Fused keeps none: the depth-5 fusion follows the frame ranks on actions. It also loses
  `s25`, the scene query InternVideo2's clip added, which is the whole gap from 76 to 74
- GPU time came out the other way from the size: Cosmos BF16 177 ms (A) and 166 ms (B) per clip, InternVideo2
  float32 186 ms, the same 64 clips

### Not verified

- An L4: timed on an A100 only
- Cosmos's text side on the laptop's CPU (the ticket's "Will it run in the product?"): skipped, as Cosmos is not
  adopted
- Whether a different fusion would keep Cosmos's four action finds: not tried, SM-1's queries cannot choose it

### Result

**Keep InternVideo2 (D-50).** Cosmos A: fused 74.0% overall (InternVideo2 76.0%), 60.0% on actions, no lift over frame
only. Its clip stream alone is better on actions (44% against 32%) and no slower, but the current fusion does not use
it. OMNI-X15 is not done; OMNI-902 starts on InternVideo2.
