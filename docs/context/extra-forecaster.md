# Extra — The forecaster, trained to forecast: context

## OMNI-X17 · The decision, before any training

### Why this ticket exists

OMNI-914 to OMNI-918 trained the forecaster to say "Event" at full strength on every second of an event window, even
30 s before it, and measured it with a pooled AUC that compares different places. Ahmad replaced them with
`extra-forecaster.md` on Fri 27 Nov. Its rule has to be written before any training step, so the choice cannot be
fitted to the numbers.

### What was done

2026-11-28. **D-65** in `docs/decisions.md` (index row and section), holding every definition X18 to X22 use: the
eight measures and their ranges, Fix 1 (`τ0`, `s` per dataset) and Fix 2 (64 windows, the gaps, the margin 0.1,
`λ_rank = 1`), the total error, versions A, B, C, the epoch rule (UCF-Crime flat-line ≤ 0.60, then the highest
UCF-Crime within-window AUC), the served model, the Fix-2 and sharing comparisons, SM-2 and SM-3.

**Choices D-65 makes that the ticket leaves open:**
- Fix 2's extra batch: windows drawn from all train windows of the dataset, event and normal as they come; a window
  too short for the gap is not drawn; each second is an ordinary sample (sequence from the window's start, D-55)
- Lead time (mTTA) is the mean over warned event windows at `θ_d`; D-64's AP-averaged mTTA is dropped. Alarms per
  hour use the normal windows' summed spans
- Ties: the first epoch; for the served model the lower seed, then A. The patience counts only epochs that pass the
  flat-line check and improve. A seed with no passing epoch has no `best.pt`
- Run ids `pfm-v5{a,b,c}-s{seed}-e{epoch}`, folders `models/pfm/v5-{a,b,c}/`

**Said openly in D-65:** OMNI-916's event-only model rose more than v2 (+0.170 against +0.076), and the flat-line
check is the guard against a score that only climbs with time; v1's and v2's held-out numbers were already seen; the
UCF-Crime `close` range is [8, 13) s because train and validation windows end 8-9 s before the event.

### Verification performed

- D-65's numbers checked against the ticket line by line (ranges, `τ0`/`s`, 64 windows, gaps 5 s / 1 s, margin 0.1,
  `λ_rank`, model and optimiser settings, 0.60, 0.70, 1.80 s)
- The facts it cites are the ticket's (checked Fri 27 Nov) and earlier entries' (OMNI-916's rise, K-36); none
  re-measured here

### Not verified

- Committed before any training (bbefc88)

### Result

D-65 written. Next: OMNI-X18, the eight measures.

---

## OMNI-X18 · The eight measures

### Why this ticket exists

D-65's choice and SM-2 rest on measures that ask "does the score rise inside a window as the event nears, and stay
flat on normal footage". OMNI-917's `pfm_measures.py` had five of them with D-64's ranges; X18 brings it to D-65.

### What was done

2026-11-28. `scripts/pfm_measures.py` rewritten to D-65's eight, all on `score = 1 − p(Normal)`:
- `RANGES`: UCF-Crime `close` τ [8, 13) s (was [0, 10)), `far` [20, 30) s; CCD unchanged. `HELDOUT_CLOSE` [0, 8) s,
  added only with `measures(..., heldout=True)` as `within_window_auc_0_8`. `FLAT_LINE_MAX = 0.60`
- New: `threshold` (the 10% rule on `score` over normal windows' maxima), `at_threshold` (lead time over warned
  event windows, detection rate, share of normal windows warned, alarms per hour with each second standing for one
  grid step: 1 s UCF-Crime, 0.5 s CCD)
- `average_precision` alone; D-64's AP-averaged mTTA dropped. Per-window and pooled AUC kept
- `measures(data, p, theta=None, heldout=False)`: `theta=None` fits `θ_d` on that split (validation only)
- `kept`, and `uv run python scripts/pfm_measures.py kept validation` (or `held_out`): windows per range, no model
- `scripts/pfm_heldout.py` now passes validation's `θ` to `measures` (else it would have refit `θ_d` on held-out)
- `scripts/tests/test_pfm_model.py`: the epoch-choice test's expected score is 1.0, not 0.875 (with `close` from
  8 s, τ 8 is the window's only close second); what it tests (within-window beats pooled) is unchanged

**Said plainly:** the warning is now `score ≥ θ_d`, where `train_pfm.evaluate` warns when the top class is an
event and `max(p) ≥ θ`. For an Event/Normal model they agree whenever `θ_d > 0.5`. `train_pfm.validation` still
merges `evaluate`'s keys with these; X21 wires the epoch rule to the eight alone.

### Verification performed

- Tests on hand-made windows with answers worked out by hand: the ticket's example (10% → 30% → 70% gives 1.0, flat
  0.5); τ 2 s is outside UCF-Crime's `close` but inside the held-out [0, 8) s one; the flat-line check counts back
  from the last second (0.75 over two windows, one skipped); `θ_d` lets exactly 1 of 10 normal windows warn; lead
  time 5.0 s, detection 2/3, false alarms 0.5 of windows and 3 alarms in 8 s (1350 per hour), CCD's 0.5 s step; AP
  0.8333 and per-window AUC 0.75; a model with classes equals a merged one. 50 PFM tests pass
- **Windows each range keeps** (`kept`, from `datasets/samples/*.csv`, no model):

| Dataset · split | Event windows | with `close` | with `far` | both | Normal windows | both |
|---|---|---|---|---|---|---|
| CCD · validation | 225 | 225 | 225 | **225** | 450 | **445** |
| UCF-Crime · validation | 114 | 114 | 72 | **72** | 240 | **222** |
| CCD · held-out | 243 | 243 | 243 | **243** | 450 | **444** |
| UCF-Crime · held-out | 280 | 206 | 142 | **142** | 300 | **262** |

  UCF-Crime held-out with `close` [0, 8) s: 280 have it, 142 both. Validation's 42 skipped event windows (21 of 30 s,
  21 of 60 s) are videos whose event comes too early for 20 s of lead-up

### Not verified

- On a trained model: X21 runs them every epoch

### Result

The eight measures, D-65's ranges, tested. UCF-Crime's main measure rests on 72 validation and 142 held-out windows.
Next: OMNI-X19, Fix 1.

---

## OMNI-X19 · Fix 1: distance weights on the class error

### Why this ticket exists

Every second of an event window was labelled "Event" at full strength, even 30 s before the event, so training asked
the model to say "Event" where nothing is visible yet. Accident anticipation (DSA, DSTA) weights each second's error
by its distance to the event; D-65 sets the weights.

### What was done

2026-11-28. `scripts/pfm_model.py`:
- `Config.distance_weights`: per domain `[τ0, s]`; empty by default, so every older checkpoint loads and trains as
  before (a config saved before it has no such key and gets the default). `Config.weights_of(domain)`
- `DISTANCE_WEIGHTS = {"ucf_crime": [8, 10], "ccd": [0, 1]}`, D-65's settings, for X21's notebook to pass
- `distance_weight(τ, τ0, s) = exp(-max(0, τ - τ0) / s)` on event samples, 1 on normal ones (`τ` NaN)
- `domain_loss(..., weights)`: with weights, the CE is `Σ w · CE / Σ w`; without, the plain mean as before. The time
  error (MSE of `Δt` on event samples) is untouched; `loss` passes each domain's own weights

### Verification performed

- Three new tests: the weights match the ticket's table (τ 8 → 1.00, 15 → 0.50, 25 → 0.18, 35 → 0.07) and CCD's
  `exp(-τ)`; a badly wrong event sample raises the loss less when it is far (τ 28) than when it is close (τ 8), with
  the weighted means worked out by hand; a normal sample weighs 1; off by default and for a pre-D-65 config the loss
  is the old one; on, the CE is weighted and the MSE unchanged (4.0). 53 PFM tests pass
- A 3-epoch run on 32 hand-made CCD samples with the weights on: it trains, the loss falls (1.18, 1.07, 0.98)

### Not verified

- Its effect on a real model: X21's versions A to C

### Result

Fix 1 in the model, off by default, tested. Next: OMNI-X20, Fix 2.

---

## OMNI-X20 · Fix 2: ranking inside the same window

### Why this ticket exists

Fix 1 tells the model how much each second counts, not that the score should climb as the event nears. Early
detection work (Ma et al., CVPR 2016) adds a ranking error inside the same sequence: a later second of an event
window must score higher than an earlier one. On normal windows D-65 adds the opposite, no climbing for no reason,
so a score that only grows with time is penalised in training too, not only by the flat-line check.

### What was done

2026-11-28.
- `scripts/pfm_model.py`: `Config.rank_gaps` (per domain, empty = off, so older checkpoints load as before),
  `rank_windows = 64`, `rank_margin = 0.1`, `lambda_rank = 1`; `RANK_GAPS = {"ucf_crime": 5, "ccd": 1}` (D-65);
  `ranking_loss(early, late, event, margin)`: the mean of `max(0, margin − (late − early))` on event windows and
  `max(0, late − early)` on normal ones, with scores `1 − p(Normal)`
- `scripts/train_pfm.py`: `draw_pairs(data, rng, n, gap)`: `n` windows without repeats among those spanning at least
  the gap (event and normal as they come), one pair from each drawn uniformly among its pairs at least the gap
  apart; the windows are indexed once per split. In each step, every domain with a gap draws its pairs from the
  epoch's `(seed, epoch)` generator, scores both seconds in one forward pass (dropout on, as the usual batch), and
  adds `λ_d × λ_rank × ranking` to the total (`λ_d` is 1, so this is D-65's sum). The history gets `{domain}_rank`

### Verification performed

- Three new tests: `ranking_loss` on six hand-made pairs (a rise of 0.3 gives 0, of 0.05 gives 0.05, a fall of 0.2
  gives 0.3; a flat normal pair 0, a climbing one 0.4, a falling one 0); `draw_pairs` over 200 draws: both seconds
  from the same window, at least the gap apart, a window too short never drawn, every allowed pair drawn (9 of 9),
  the same generator gives the same pairs; training: off by default and for a pre-D-65 config, on it adds a
  non-negative `ccd_rank` and changes the loss, and a run stopped after epoch 1 and resumed gives the same losses as
  one that never stopped. 56 PFM tests pass

### Not verified

- Its effect on a real model, and its cost per step on Colab: X21's versions B and C
- The full suite (Docker was not running, so the Postgres tests could not run); the PFM tests do not need it

### Result

Fix 2 in training, off by default, tested. Next: OMNI-X21, training on Colab.
