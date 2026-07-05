# Spain vs Portugal — Control, Transition Risk and Knockout Uncertainty

A professional pre-match analytical model for **Spain vs Portugal**, built as the next step after the Spain vs Austria project. The goal is not to publish a simple winner prediction, but to create a structured football analytics framework that explains **how the match can be decided**, **which mechanisms matter most**, and **how knockout uncertainty changes the interpretation of the model**.

## Core idea

Spain's main advantage is control: possession, territory, sustained pressure, passing volume and the ability to move the game into Portugal's half.

Portugal's main threat is what happens when that control breaks: transition attacks, individual quality, dangerous turnovers and the possibility of turning fewer attacking sequences into high-impact chances.

The central question of the project is:

> After Spain's 3-0 win over Austria, which parts of Spain's dominance are repeatable against Portugal, and which parts become risky against a stronger transition team?

## What this project contributes

This notebook is designed around five analytical contributions:

1. **Model validation before model extension**  
   The Austria model is audited first. The notebook checks which projected intervals captured the real match and which areas needed correction before adapting the framework to Portugal.

2. **A data layer that prioritises the current World Cup**  
   The model gives the highest weight to matches from the current tournament, then uses preparation and comparable matches as secondary context. This prevents the analysis from being driven by outdated form while still retaining enough information to estimate team style.

3. **A structural model rather than a generic prediction**  
   The model separates control, transition threat, defensive structure, chance quality and penalty-shootout edge. These components are then combined into an adjusted expected-goals framework.

4. **A knockout simulation instead of a 90-minute-only forecast**  
   The model estimates not only 90-minute outcomes, but also extra time, penalties and qualification routes. This matters because the probability of advancing is not the same as the probability of winning in regulation time.

5. **Visual outputs designed for interpretation**  
   The figures are selected to explain the logic of the model: what the previous model missed, how the evidence is weighted, how the tactical profiles differ, how xG is built, how scorelines are distributed and how qualification probability is decomposed.

## Data design

The project uses editable CSV inputs stored in `data/`:

| File | Purpose |
|---|---|
| `match_logs_master.csv` | Weighted team match evidence from World Cup, preparation and comparable-match layers |
| `tactical_priors.csv` | Structural priors for control, transition threat, defensive security, chance quality and related style dimensions |
| `austria_model_validation.csv` | Post-match audit of the previous Spain vs Austria model |
| `penalty_inputs.csv` | Penalty conversion and goalkeeper-adjustment assumptions used in the shootout module |

The data is intentionally transparent. The notebook is built so assumptions can be inspected and edited instead of hidden behind a black-box model.

## Model structure

The analysis follows this pipeline:

```text
Spain vs Austria model
        ↓
Post-match validation
        ↓
Correction of weak points
        ↓
Spain vs Portugal opponent-specific model
        ↓
Style-adjusted xG
        ↓
90-minute Monte Carlo simulation
        ↓
Extra-time and penalty simulation
        ↓
Qualification probability by route
```

### 1. Previous model audit

The first stage checks whether the Austria model's predicted intervals contained the observed values. This is used as a calibration step before facing a different type of opponent.

### 2. Evidence weighting

The model separates evidence into layers and gives more importance to current World Cup matches. Preparation and comparable matches are used as stabilisers, not as the main driver.

### 3. Structural indexes

The notebook builds team-level structural indexes:

- **Control index**: possession control, passing security, territory and sustained pressure.
- **Transition index**: direct attacks, counterattacking threat and individual attacking ceiling.
- **Defensive index**: defensive security, rest defence and protection against transitions.

### 4. Style-adjusted xG

Expected goals are not treated as a standalone number. The model decomposes xG into:

- baseline attacking strength,
- opponent defensive adjustment,
- control and territory adjustment,
- transition-risk adjustment,
- chance-quality adjustment,
- set-piece / penalty-related uncertainty.

### 5. Monte Carlo knockout simulation

The simulation produces:

- 90-minute scoreline probabilities,
- probability of Spain win / draw / Portugal win in regulation,
- extra-time outcomes,
- probability of reaching penalties,
- conditional penalty-shootout edge,
- overall qualification probability.

## Final figures

The notebook generates the following output figures in `outputs/`:

| Figure | Output file | Analytical role |
|---|---|---|
| 1 | `01_austria_model_audit_interval_calibration.png` | Audits the previous Spain vs Austria model and identifies what needed correction |
| 2 | `02_data_weighting_world_cup_vs_preparation.png` | Shows how the model prioritises current World Cup evidence over preparation/comparable matches |
| 3 | `03_projected_match_statistics_public_metrics.png` | Main public-facing statistical projection, using variables that are interpretable and easier to validate |
| 4 | `04_tactical_style_radar_professional.png` | Compares tactical profiles and shows the control-versus-transition contrast |
| 5 | `05_route_to_xg_decomposition.png` | Explains how the final expected-goals values are built from different model components |
| 6 | `06_scoreline_probability_matrix_90min.png` | Shows the full 90-minute scoreline probability distribution |
| 7 | `07_qualification_routes_clean_final.png` | Decomposes qualification probability into 90 minutes, extra time and penalties |

The final notebook deliberately avoids unnecessary duplicate graphics. Each figure is meant to answer a distinct analytical question.

## Key analytical interpretation

The model's interpretation is not simply that Spain are favoured. The more important conclusion is that Spain's advantage depends on the quality and security of their control.

Spain can improve their path to qualification if they:

- sustain possession in advanced areas,
- turn territorial dominance into chance quality,
- recover quickly after losing the ball,
- limit Portugal's transition volume.

Portugal's route becomes stronger if they:

- survive Spain's pressure without being pinned back for long periods,
- exploit Spain's losses in advanced zones,
- generate fewer but higher-leverage attacks,
- push the match towards extra time or penalties.

## How to run the project

Install the required packages:

```bash
pip install -r requirements.txt
```

Open and run the notebook:

```text
notebooks/spain_portugal_final_project.ipynb
```

The notebook regenerates all tables and figures in:

```text
outputs/
```

## Project structure

```text
spain_portugal_final_project/
├── data/
│   ├── austria_model_validation.csv
│   ├── match_logs_master.csv
│   ├── penalty_inputs.csv
│   └── tactical_priors.csv
├── notebooks/
│   └── spain_portugal_final_project.ipynb
├── outputs/
│   ├── 01_austria_model_audit_interval_calibration.png
│   ├── 02_data_weighting_world_cup_vs_preparation.png
│   ├── 03_projected_match_statistics_public_metrics.png
│   ├── 04_tactical_style_radar_professional.png
│   ├── 05_route_to_xg_decomposition.png
│   ├── 06_scoreline_probability_matrix_90min.png
│   ├── 07_qualification_routes_clean_final.png
│   └── model output CSV files
├── src/
│   └── run_full_model.py
├── requirements.txt
└── README.md
```

## Notes and limitations

This is an analytical framework, not an official betting model. Some variables are model inputs or estimated priors built for transparency and interpretability. The purpose is to structure football interpretation through data, not to claim certainty about the result.

The most important output is therefore not a single probability, but the explanation of the routes by which each team can gain advantage.
