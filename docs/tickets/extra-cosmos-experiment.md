# Extra — Is Cosmos-Embed1 better than InternVideo2 for actions?

**Not a slice · Mon 23 – Tue 24 Nov 2026 · runs before OMNI-902 · on branch `slice-09-10-forecast`**
Board: OmniSight Phase B · Assignee: Ahmad

> **By Tue 24 Nov:** SM-1 measured again on VIRAT with NVIDIA's **Cosmos-Embed1-336p** as the
> clip (action) encoder instead of InternVideo2, and a decision: switch or keep. Whatever
> wins is the clip encoder the forecasting model trains on (OMNI-902).

**Why now:** the PFM (Algorithm 4) trains on clip vectors. Once OMNI-902 encodes the training
windows, changing the clip encoder means encoding them all again. Nothing has been encoded
yet, so this is the cheapest moment to check.

**Why Cosmos-Embed1:** on actions, InternVideo2 is our weak side: SM-1's action queries are
60% fused against 60% frame only, no lift (K-30), and in a 2026 independent benchmark
InternVideo2 and SigLIP 2 ranked last. Cosmos-Embed1-336p (NVIDIA, about 1B parameters, 8
frames at 336 px, 768-D, a text encoder included, NVIDIA Open Model License) reports
Kinetics-400 87.7 against InternVideo2's 62.8 on its model card. It is the same size class as
InternVideo2's video side, and it keeps Algorithm 2's shape: one clip of 8 frames per kept
frame.

**Only the clip stream changes.** SigLIP 2 stays (scene queries 92%; the query classifier and
Cyrine's box fix depend on it).

**Not in this ticket:** changing the product's code (that is OMNI-X15, only if Cosmos wins);
other models.

---

## The rule, written before running

**Switch to Cosmos-Embed1 if, on SM-1's 50 queries:**

1. **fused recall@5 on the 25 action queries is above frame only's** (the lift SM-1 asks for,
   which InternVideo2 never gave), **and**
2. **fused recall@5 overall is at least InternVideo2's 76.0%.**

Otherwise keep InternVideo2 (D-33), and OMNI-902 starts with it. Either way the numbers go in
`results.md`.

---

## OMNI-X14 · SM-1 with Cosmos-Embed1 as the clip encoder · Mon 23 – Tue 24 Nov

### On Colab (GPU)

- [ ] A notebook `notebooks/cosmos_sm1_gpu.ipynb`, reusing the VIRAT archive's code
      (`encode_virat_archive.py`): the **same 275 videos and the same 35,704 kept frames**
      (frame index and clip window read from the existing archive, never re-gated), so only the
      encoder differs
- [ ] For each kept frame, its clip as today (the window `[t − 2Δ, t]`, 8 evenly spaced
      frames), encoded with **Cosmos-Embed1-336p** at a pinned revision (BF16, as its card
      says). Unit vectors, 768-D
- [ ] Cached to Drive (`embeddings/virat_cosmos/`), resumable like the archive
- [ ] Time per clip on the GPU, recorded

### SM-1 and the weights

- [ ] **Re-tune the fusion weights for Cosmos** with `scripts/tune_fusion.py` on the **tuning
      queries only** (`datasets/eval/sm1_dev_queries.csv`), exactly as D-40 did for
      InternVideo2. Never on SM-1's queries
- [ ] Run `scripts/measure_sm1.py` with Cosmos's clip vectors and text vectors: frame only,
      clip only, fused. Recall@5 overall, scene, action, and chance
- [ ] **Side by side** in `results.md`: InternVideo2 (the existing numbers) and Cosmos-Embed1

### Will it run in the product?

- [ ] Cosmos-Embed1's **text side** on this laptop's CPU, beside SigLIP 2 in `ai_workers`:
      memory used and time per query. It must fit Docker's memory with the rest of the stack
- [ ] Its loading needs `trust_remote_code=True`: pin the revision and note it

### Decide

- [ ] Apply the rule above. Record the decision (**D-49**): switch or keep, with the numbers.
      If switching, it replaces D-33 for the clip stream. (This moves the sprint's decisions
      to start at D-50.)
- [ ] Tell Ahmad the result in two lines before OMNI-902 starts

---

## OMNI-X15 · Switch the product to Cosmos-Embed1 · only if Cosmos wins · Tue 24 Nov

- [ ] `ai_workers`: Cosmos-Embed1's text side replaces InternVideo2's (and its video side for
      the GPU path, OMNI-X04); the pinned id and revision in `CommonSettings`
- [ ] Qdrant: `clip_vectors` becomes 768-D; the 9 VIRAT product videos and any video added
      from Drive get their Cosmos clip vectors (the existing import, from the new archive)
- [ ] The Colab upload path (`encode_uploads.py`, OMNI-X05) uses Cosmos for the clip stream
- [ ] The tuned weights from OMNI-X14 in the product's fusion
- [ ] **TC-4** run again on the new vectors; SM-1, TC-4 and the tracker updated
- [ ] `uv run pytest` passes; a search on the product shows Cosmos's action results
