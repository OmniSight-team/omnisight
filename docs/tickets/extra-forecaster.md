# Extra — The forecaster, trained to forecast: Algorithm 4 revised

**Not a slice · Sat 28 Nov 2026 (Colab runs may finish Sun 29 Nov) · branch `slice-09-10-forecast`**
Board: OmniSight Phase B · Assignee: Ahmad

> **By the end of Sat 28 Nov:** a forecasting model trained to **rise as an event nears and stay flat on normal
> footage**, measured with measures that ask exactly that, chosen on validation by a rule written before training,
> and measured once on held-out. SM-2 and SM-3 recorded.

**This ticket stands alone.** OMNI-914, OMNI-916, OMNI-917 and OMNI-918 (`week-09-10.md`) are not used: their
training asked for the wrong thing and their measures had wrong ranges. Everything needed is written here. Existing
code may be reused only where it matches the definitions below, with its tests updated to them.

---

## Why

1. **Training asked for the wrong thing.** Every second of an event window was labelled "Event" at full strength,
   even 30 s before the event, when nothing is visible yet. So training said "say Event everywhere", while what we
   want is "rise as the event nears, stay flat on normal footage".
2. **The measure asked the wrong question.** The AUC put every second of every video in one pile: a second 30 s
   before the event counted as "event", and seconds of different videos (different places) were compared, so a model
   could score by telling places apart ("Auditing Frame-Level AUC", arXiv 2608.11985).
3. **The accident-anticipation papers already solve (1):** each second's error is weighted by its distance to the
   event (DSA, DSTA, AdaLEA: arXiv 2106.10197, 1804.02675). Early-detection work adds a ranking loss so the score
   does not drop as the event nears (Ma et al., CVPR 2016).
4. **Joint training is unproven.** One model learns CCD and UCF-Crime together, their errors added. Whether CCD helps
   UCF-Crime, or gets in its way, was never tested. Version C tests it.

**Facts this ticket is built on (checked Fri 27 Nov):**
- UCF-Crime's **train and validation** windows all end **8-9 s before the event** (UCA's times are not exact, D-49).
  **Held-out's** end **at the event** (official times: 248 of 280 windows have a second under 1 s before it)
- CCD's windows reach the crash (τ from 0.1 s), with about 3.5 s of lead-up

---

## Algorithm 4, then and now

| | Phase A | Now |
|---|---|---|
| Input | the clip vectors of the window | each clip vector **minus the window's first** (the place cancels out) |
| Labels | each dataset's classes (UCF-Crime: 13 crimes + Normal) | **Event or Normal** (the alert says "something is coming, and when") |
| Class error | cross-entropy, every second equal | cross-entropy **weighted by distance to the event** (Fix 1) |
| Extra error | none | **ranking inside the same window** (Fix 2) |
| Time error | MSE of `Δt` | unchanged |
| The main measure | the real event in the top 3 | **within-window AUC** (rises before events) + **flat-line check** (calm on normal) |
| Threshold `θ_d` | on validation | unchanged: the 10% rule on validation normal windows |

The book's Algorithm 4 is updated after this ticket, with the results (not on Saturday).

---

## OMNI-X17 · The decision, before any training · Sat morning

- [x] A new D-number holding **every definition in X18-X22**: the measures and their ranges, Fix 1's and Fix 2's
      settings, the versions, how the epoch, the seed and the served model are chosen, SM-2 and SM-3. Written and
      committed before the first training step; nothing in it changes once a number exists
- [x] It says openly what was seen before it was written (the old event-only model rose more than v2), and that the
      flat-line check is the guard against a score that only climbs with time

## OMNI-X18 · The eight measures · Sat morning

Every measure scores a second by `1 - p(Normal)`. Each is tested on hand-made windows with a known answer
(for example scores 10% → 30% → 70% give a within-window AUC of 1.0; a flat line gives 0.5).

1. **Within-window AUC (the main one).** Inside each event window, every `close` second against every `far` second
   **of the same window**: the share where `close` scores higher, ties half. The mean over the windows that have
   both, and how many were skipped
   - UCF-Crime: `close` τ **[8, 13) s**, `far` τ **[20, 30) s**, the **same on validation and held-out**
   - CCD: `close` τ [0, 1) s, `far` τ [2.5, 5) s
   - Held-out UCF-Crime only: also with `close` τ [0, 8) s, reported **separately** (closer than training saw)
2. **Flat-line check.** The same inside each **normal** window, counting time back from the window's last second,
   the same ranges. About 0.5 is right; **≤ 0.60 is required**
3. **Lead time (mTTA):** the mean seconds before the event of each event window's first warning at `θ_d`
4. **Detection rate:** the share of event windows warned at `θ_d`
5. **False alarms:** the share of normal windows warned at `θ_d`, and alarms per hour of normal footage (a run of
   consecutive warning seconds is one alarm)
6. **AP:** every window's highest score is a warning line; at each line an event window is a hit if it warns before
   the event, a normal window a false alarm if it warns at all; the area under precision-recall
7. **Per-window AUC:** each window's highest score, event windows against normal windows
8. **Pooled AUC:** every second of every window in one pile, as papers report it; for comparison only

- [x] Code and tests; a printout of how many validation windows each range keeps, per dataset

## OMNI-X19 · Fix 1: distance weights on the class error · Sat morning

- [ ] An event second's cross-entropy is multiplied by `w = exp(-max(0, τ - τ0) / s)`:
  - UCF-Crime: `τ0` = 8 s, `s` = 10 s (τ 8 → 1.00, 15 → 0.50, 25 → 0.18, 35 → 0.07)
  - CCD: `τ0` = 0 s, `s` = 1 s (DSTA's setting)
  - Normal seconds: weight 1
  - The class error of a batch is the weighted mean: `Σ w · CE / Σ w`
- [ ] The time error (`Δt`'s MSE on event seconds) is unchanged
- [ ] A `Config` option, off by default, so older checkpoints load as before. Tested: a far event second's error
      counts less than a close one's; a normal second's counts fully

## OMNI-X20 · Fix 2: ranking inside the same window · Sat morning

- [ ] Each training step, beside the usual batch, one extra batch per dataset of **64 windows** from the train split,
      **two seconds from each** (drawn at random, the later at least 5 s after the earlier on UCF-Crime, 1 s on CCD)
- [ ] Error per pair, with score = `1 - p(Normal)`:
  - event window: `max(0, 0.1 - (score_late - score_early))`: the closer second must score higher, by 0.1
  - normal window: `max(0, score_late - score_early)`: no climbing for no reason
- [ ] The ranking error is the mean over the pairs, times `λ_rank` = 1, added to the total
- [ ] A `Config` option, off by default. Tested on hand-made windows (a rising pair gives 0, a falling pair a positive
      error; a climbing normal pair a positive error, a flat one 0)

## OMNI-X21 · Training on Colab: three versions · Sat midday → Sun

**The total error of one step** = for each dataset, (weighted class error) + `α_d` × (time error) + (ranking error,
version B and C only), the datasets' totals added. `α_d` = 1 / var(τ) of that dataset's train split.

**Unchanged and stated here:** one shared Transformer (width 128, 2 layers, 4 heads) and one head per dataset
(chance of Event, and `Δt`); the input minus the window's first vector; labels Event or Normal; the fixed split
(D-22) and the committed samples; AdamW, lr 3e-4, weight decay 0.01, batch 256, gradient clip 1.0; up to 120
epochs, patience 30; 3 seeds (0, 1, 2); `θ_d` by the 10% rule on validation normal windows.

| Version | Datasets | Fix 1 | Fix 2 | What it answers |
|---|---|---|---|---|
| **A** | CCD + UCF-Crime | ✓ | | what Fix 1 alone does |
| **B** | CCD + UCF-Crime | ✓ | ✓ | what Fix 2 adds |
| **C** | UCF-Crime only | ✓ | ✓ | does learning crashes help forecast crimes? (B against C) |

- [ ] **Choosing the epoch** (every version): after every epoch, all eight measures on validation go into the history.
      An epoch can become `best.pt` only if its **UCF-Crime flat-line check is ≤ 0.60**; among those, the highest
      **UCF-Crime within-window AUC** wins. The same rule drives the patience
- [ ] One notebook, new folders and run ids per version and seed; resumable. **Order: B, A, C** (the served model
      comes from A or B, so those first; C may finish on Sunday)
- [ ] A local dry run first (a few videos, 2 epochs, each version) so nothing breaks on Colab

## OMNI-X22 · The choice, held-out once, the record · Sat evening (C's comparison when it ends)

- [ ] **The served model:** the seed of **A or B** with the highest UCF-Crime validation within-window AUC (both
      datasets need a head, so C is not served). If no epoch of any seed passed the flat-line check, nothing new is
      served (the product keeps its current model) and the failure is recorded
- [ ] **What Fix 2 adds:** A's and B's mean and seed spread of the within-window AUC; claimed only if the difference
      is larger than the larger spread
- [ ] **Does sharing help:** B's and C's UCF-Crime numbers the same way. If C is clearly better, a follow-up ticket
      trains separate models per dataset
- [ ] **Validation table:** every seed of every version, both datasets, all eight measures, and the windows kept
- [ ] **Held-out once**, only for the served model, all eight measures (plus the τ [0, 8) s within-window AUC):
  - **SM-2:** within-window AUC **≥ 0.70** on held-out UCF-Crime
  - **SM-3:** lead time **≥ 1.80 s** on held-out CCD
- [ ] `results.md`: SM-2's and SM-3's rows, this ticket's row; `decisions.md`: the D-number's result
- [ ] OMNI-910 serves the chosen model (only its run id changes)

---

## Saturday at a glance

| When | What |
|---|---|
| Morning | X17 decision → X18 measures → X19 Fix 1 → X20 Fix 2, all tested; local dry run |
| Midday | X21 on Colab: B, then A, then C |
| While Colab trains | OMNI-910 · serving the model (`week-09-10.md`) |
| Evening | X22 for A and B: the choice, held-out once, the record |
| Sunday | C's comparison when it ends |
