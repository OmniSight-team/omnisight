# Extra — Which clip encoder should the forecasting model read? CCD decides

**Not a slice · Tue 24 – Wed 25 Nov 2026 · inside OMNI-902 to 904 · branch `slice-09-10-forecast`**
Board: OmniSight Phase B · Assignee: Ahmad

> **By Wed 25 Nov:** the same PFM trained twice on CCD, once on InternVideo2 clip vectors and
> once on Cosmos-Embed1 clip vectors, compared on the **validation** split, and a decision:
> which encoder the PFM reads from now on. UCF-Crime is then encoded with the winner only.

**Why:** OMNI-X14 (D-50) kept InternVideo2 **for search**, because the fused result did not
improve. But the PFM reads **only clip vectors**, no fusion, and there Cosmos was clearly
better: clip only 54% against 46%, on action queries 44% against 32%. Cosmos was also trained
on driving footage, which is what CCD is. Search is not prediction, and 25 queries is a small
sample, so the prediction task itself decides.

**Search does not change.** SigLIP 2 and InternVideo2 stay for search whatever wins here. If
Cosmos wins, it is the PFM's encoder only: no text side, so no laptop-memory question, and
search's `ai_workers` keeps its transformers version. **But anywhere Cosmos clip vectors are
made** (this ticket, OMNI-902, the Colab upload path, a GPU machine later) it needs **its own
process on transformers 4.51.3**, as X14 ran it (D-50).

**Live cameras lose nothing either way.** On the laptop's CPU neither encoder runs live:
InternVideo2 already doesn't (D-32), and Cosmos took 326 s per clip in a check. Both wait for
a GPU machine (OMNI-X04).

**The cost:** CCD is 4,500 windows, 16,700 s of footage: about 36,000 clips per encoder on the
0.5 s grid, the size of the VIRAT archive. About 2 hours per encoder on an A100, about 4 hours
of Colab on Tue for both. It fits only if Colab does not drop: the runner resumes.

**Not in this ticket:** UCF-Crime (encoded after the decision, with the winner); SM-2, SM-3 or
anything on held-out data.

---

## The rule, written before training

Train each encoder's PFM with **3 seeds**, the same config, the same samples. On the CCD
**validation** split:

- **Main measure:** ROC-AUC of Crash against Normal (mean of the 3 seeds)
- **Cosmos wins if** its mean AUC is higher than InternVideo2's by more than the larger of the
  two encoders' seed spreads (max − min), **and** its validation lead time (mean time before
  the accident at the threshold rule of OMNI-905, chosen on validation) is not lower
- **Otherwise InternVideo2 stays:** it is already in the product, so a tie keeps it

Held-out data is not touched.

---

## OMNI-X16 · CCD trained twice, one model per clip encoder · Tue 24 – Wed 25 Nov

### Tue 24 Nov · CCD encoded with both encoders (OMNI-902's CCD part)

- [ ] OMNI-902's runner on **every CCD window** (positive and negative, every split), on its
      0.5 s grid, the same clip (8 frames from the 1 s before `t`), **twice**:
  - **InternVideo2** (the product's pin, D-33)
  - **Cosmos-Embed1-336p** at X14's pin (`0e8a28f7`), BF16, in its own process with
    transformers 4.51.3, exactly as X14 ran it (D-50)
- [ ] Cached separately on Drive (`embeddings/pfm/ccd/internvideo2/`, `.../cosmos/`),
      resumable; time per clip for each
- [ ] OMNI-903's samples for CCD, built once per encoder from the same windows, so the only
      difference is the vectors

### Wed 25 Nov · the same model, trained twice

- [ ] OMNI-904's model (with its overfit and resume proofs), **CCD only**, one config for
      both encoders; only the input size differs (512 or 768)
- [ ] 3 seeds each, the same epochs and settings; nothing tuned separately per encoder
- [ ] On **validation**: ROC-AUC, top-1 accuracy, the error of `Δt` (mean absolute), and the
      lead time at the OMNI-905 threshold rule. Mean and spread over the seeds
- [ ] Side by side in `results.md`

### Decide

- [ ] Apply the rule. Record **D-53**: which encoder the PFM reads, with the numbers
- [ ] **If Cosmos wins:**
  - OMNI-902 encodes UCF-Crime with Cosmos only
  - OMNI-910 and OMNI-911: the PFM reads **Cosmos clip vectors** stored for it alone (a third
    vector set beside the two search ones; search is untouched)
  - The Colab upload path (`encode_uploads.py`, OMNI-X05) also makes Cosmos clip vectors,
    **in its own process on transformers 4.51.3** (as X14 and this ticket run it; the
    notebook's SigLIP 2 and InternVideo2 stay on 5.20.1), and `add_from_drive.py` imports
    them, so an added video can be scored
  - On a GPU machine later (OMNI-X04), live cameras get Cosmos too, in its own container on
    transformers 4.51.3; its GPU time is recorded
- [ ] **If InternVideo2 stays:** OMNI-902 encodes UCF-Crime with InternVideo2; the rest of the
      sprint is unchanged
- [ ] Tell Ahmad the result in two lines before UCF-Crime starts encoding
 