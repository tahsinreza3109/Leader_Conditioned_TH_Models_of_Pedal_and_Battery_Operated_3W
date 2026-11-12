# Leader-conditioned time-headway models of pedal and battery-operated three-wheelers

Code for the article "Following Behavior and Regulation: Leader-Conditioned Time-Headway Distributions of Pedal and Battery-Operated Three-Wheelers in Non-Lane-Based Traffic".

The study models the following time headway of pedal rickshaws (PR) and battery-operated three-wheelers (BTW, e-rickshaws) at two crowded urban road segments in Dhaka, Bangladesh, and tests the speed limits of the 2026 e-rickshaw operation rules and a dedicated BTW lane.

**This repository holds the code only.** The data are published on Mendeley Data:

> Reza, T., Raihan, M. A., & Hasan, M. M. U. (2026). *Time headways of pedal rickshaws and battery-operated three-wheelers in Dhaka* [Dataset]. Mendeley Data, V1. https://doi.org/10.17632/b2krjbv299.1

## Contents

```
README.md
requirements.txt               Python packages (tested versions)
notebooks/
  1_data_preparation.ipynb     raw extraction to data01-data04; pooling tests; traffic-state figures
  2_headway_models.ipynb       truncated distributions, leader-conditioned AFT models, covariates
  3_mechanism_and_policy.ipynb spacing-speed decomposition, speed limits, dedicated lane, window sensitivity, article tables
  4_all_and_submission_figures.ipynb   runs 1-3 in one pass and exports every figure as a separate file
  hwdist.py                    truncated headway distributions and AFT models
  style.py                     colors, fonts and file formats
```

## How to run

1. Download the dataset from the DOI above and copy its `data/` folder into the root of this repository, so that `data/raw/` sits next to `notebooks/`. The `data/README.md` in the dataset describes every file and variable.
2. Install Python 3.12 or later and the packages: `pip install -r requirements.txt`. The figures use the Cambria font.
3. Run the notebooks from the `notebooks/` folder, top to bottom, in order 1, 2, 3. Notebook 1 rebuilds data01 to data04 in `outputs/data/`; notebooks 2 and 3 write the result tables to `outputs/tables/`, the figures to `outputs/figures/` and the article tables to `outputs/Manuscript_Tables.xlsx`. Notebook 2 takes the longest (about 30 minutes).
4. Alternatively, run notebook 4 alone. It repeats notebooks 1 to 3, saves the charts at 1000 dpi in `outputs_TRA/`, draws the study-area and vehicle figures, and exports all figures as `Figure_1` to `Figure_10` and `Figure_A1` to `Figure_A13` in `outputs_TRA/submission_figures/`.

All random draws use fixed seeds, so a rerun reproduces the published numbers. The notebooks are stored without outputs.

## Method in brief

All models are fitted separately for four cases: Shahjahanpur PR (n = 77), Shahjahanpur BTW (191), Tikatuli PR (150) and Tikatuli BTW (476). Events were extracted only when 0.5 s <= headway <= 5 s, so every likelihood is truncated to that window.

| Step | Question | Method |
|---|---|---|
| Data | From video to an analysis dataset | Manual frame coding at two reference lines; space-mean speeds; rolling 30-s flow from YOLO counts; pooling of vehicle types by equivalence (TOST) and Kruskal-Wallis tests |
| 1 | Which distribution describes the headway? | Truncated lognormal, log-logistic, Weibull, gamma and generalized gamma; AIC; parametric-bootstrap KS, AD and CvM tests; nested likelihood-ratio tests |
| 2 | How does the leader class change the headway? | Leader-conditioned truncated Weibull AFT model; time ratios; Cox-Snell residuals |
| 3 | Do off-centered following, occupancy or leader speed matter? | Crude and leader-adjusted models; Holm-adjusted likelihood-ratio tests |
| 4 | Is the PR-BTW gap a spacing gap or a speed gap? | Exact split log H = log S - log v at the same leader mix; stratified bootstrap |
| 5 | What do the e-rickshaw speed limits do? | 25 km/h and 15 km/h limits applied to BTW following events; exposure to headways below 2 s and 1.5 s |
| 6 | What would a dedicated BTW lane do? | BTW-led events at 25 km/h, with a centered-only check |
| 7 | Do the results depend on the extraction window? | Refits on narrower windows |

## License

CC BY 4.0, the same as the dataset.
