# Methodology

## Data Scope

The dashboard covers Myanmar conflict events from `2021-02-01` onward using ACLED as the base source. The current processed dataset in this repository extends through `2026-03-27`.

## Pipeline Overview

The pipeline lives in `pipeline/pipeline.py` and supports two modes:

- Full rebuild: process the historical Excel source, then top up from the ACLED API
- Incremental refresh: load the existing Parquet dataset and fetch a rolling overlap from the latest stored event date

The incremental update path now uses timestamp-based syncing, which is the method ACLED documents for catching newly added rows, edits to older rows, and deletions.

## Output Files

The pipeline writes these canonical outputs to `data/processed/`:

- `acled_cleaned.parquet`
- `acled_actor_level.parquet`
- `acled_actor_ally_pairs.parquet`
- `monthly_township.parquet`
- `last_updated.txt`

These Parquet files are the application source of truth.

The pipeline also maintains `acled_sync_state.json`, which stores the last sync cursor used for timestamp-based incremental refreshes.

## Cleaning And Recoding

The pipeline applies the following main transformations:

- parses and validates event dates
- drops future-dated rows
- normalizes `admin1` names
- matches township P-codes
- recodes `key_event`
- recodes `detailed_event`
- standardizes actor labels and actor-type groupings
- normalizes civilian-targeting flags
- coerces fatalities to integers
- recodes civilian-targeting events with `fatalities >= 5` as `Massacres`

## Key Event Logic

The dashboard groups raw ACLED event data into higher-level analytical categories:

| Dashboard label | ACLED source | Notes |
|---|---|---|
| Ground-based attack | `event_type` = Battles OR (Explosions/Remote violence, non-air sub-types) | Bundles armed clashes, artillery/shelling, IEDs, landmines, grenades, and suicide bombs. Also captures airstrikes that occurred during ground battles — see ACLED merging rule below. |
| Air attack | `sub_event_type` = Air/drone strike, `inter1` = State forces, "drone" absent from notes | Heuristic split — not an official ACLED category |
| Drone attack | `sub_event_type` = Air/drone strike, `inter1` ≠ State forces OR "drone" appears in notes | Heuristic split — not an official ACLED category |
| Massacres | `civilian_targeting` = Yes AND `fatalities` ≥ 5 | Dashboard-defined threshold; overrides whatever key_event ACLED assigned |
| Violence against civilians | ACLED event type, not reclassified as Massacre | ACLED "civilian targeting" flag = civilian was *main or only* target; incidental civilian harm in battles does NOT set this flag |
| Protests | ACLED event type | Direct mapping |
| Arrests | `sub_event_type` = Arrests | From Strategic developments |
| Looting/property destruction | `sub_event_type` = Looting/property destruction | From Strategic developments |
| Displacement | `sub_event_type` = Other AND "displacement" in notes | Heuristic; a small residual |
| Others | Anything not matched above | Dominated by strategic developments: agreements, base establishment, non-violent territory transfers, group-activity changes |

### ACLED event-merging rule and Air attack undercounting

Per the ACLED codebook, when multiple violence types occur at the same location and date, they are merged into a single event coded at the **hierarchically highest type**. Battles ranks above Explosions/Remote violence. This means airstrikes that accompany ground battles are absorbed into the Battles record and appear in this dashboard as **Ground-based attack**, not as Air attack or Drone attack.

The practical implication: **Air attack and Drone attack counts are understated relative to total aerial activity**. They capture standalone strikes only. This limitation is inherited from ACLED's coding structure, not from our pipeline.

### Massacres threshold

The threshold of `civilian_targeting = "Yes"` AND `fatalities >= 5` is a dashboard-defined operational cut. It is not an ACLED category. The civilian targeting flag, per the ACLED codebook, is set only when civilians were "the main or only target" of the event — incidental civilian deaths during battles or bombardments do not trigger it.

### Actor classification and the Air/Drone split

ACLED classifies actors into eight inter-types: State Forces, Rebel Groups, Political Militias, Identity Militias, Rioters, Protesters, Civilians, and External/Other Forces. The pipeline uses `inter1` (actor type of the primary actor) to distinguish Air attack (state forces) from Drone attack (other actors or explicit drone mentions in notes). This is a practical heuristic — the distinction between regime air assets and non-state drones is supported by the Myanmar conflict record but is not directly coded in ACLED.

## Geographic Matching

Township P-codes are matched using a district-plus-township lookup. The pipeline prefers the external lookup workbook when present and falls back to rebuilding the lookup from the existing cleaned Parquet dataset when necessary.

Some city-level rows require hard-coded fallback handling for places such as Mandalay, Nay Pyi Taw, and Yangon when ACLED does not provide a useful township field.

## Actor-Level Datasets

`acled_actor_level.parquet` is derived from armed conflict event categories, including `Massacres`. Each event is expanded into actor-role rows so the dashboard can analyze:

- who appeared in the event
- which dashboard-coded participant side they appeared on
- which actors appeared alongside them on the same side

`acled_actor_ally_pairs.parquet` is then built by exploding same-side actor relationships into pair rows.

The current actor-side dataset is built from the recoded `primary_actor / assoc_actor_1` side and `secondary_actor / assoc_actor_2` side. This is a dashboard analytical grouping. ACLED's `Actor1` and `Actor2` fields do not, by themselves, identify an aggressor, victim, or side that suffered more harm. Any UI wording that implies `attacker`, `target`, `offensive`, or `defensive` roles should therefore be treated with caution unless supported by additional coding logic and documentation.

## Pre-Aggregation For Performance

The dashboard does not build all map inputs from the full event-level dataset on every interaction. Instead, the pipeline pre-aggregates township-month metrics into `monthly_township.parquet`, which the Overview page uses for much faster choropleth and chart updates.

## Map Rendering Strategy

The animated choropleths use Plotly native animation frames bundled into the figure, rather than per-frame server callbacks. This keeps quarter-by-quarter playback responsive in the browser.

To reduce the size of the initial map callback, the app prefers a simplified web geometry file, `data/shapes/boundaries_web.geojson`, instead of the full raw township boundary file when that optimized asset is available.

Neighboring-country border context is drawn from a separate clipped Natural Earth-derived file, `data/shapes/neighbor_borders.geojson`, rather than Plotly's built-in world-outline layer. This keeps the surrounding borders source-controlled and avoids unrelated distant border segments appearing in the Myanmar map view.

## Fatality Figures

ACLED records fatalities as the most conservative estimate across conflicting source reports. Vague language is standardized: "several" or "many" → 3 or 10; "dozens" → 12; "hundreds" → 100. ACLED has **no minimum fatality requirement** for event inclusion — zero-fatality events are recorded alongside mass-casualty events. As a result:

- All event counts on this dashboard include zero-fatality events.
- All fatality totals are **conservative lower bounds**, not confirmed ground-truth figures.
- Fatality figures are subject to retroactive revision as ACLED updates source coverage.

## ACLED Actor Fields and Dashboard Side Assignment

ACLED records two actor slots (`Actor1`, `Actor2`) and two associated-actor slots (`assoc_actor_1`, `assoc_actor_2`). The codebook notes that `Actor1` and `Actor2` do not, by themselves, identify an aggressor or victim — they reflect who participated, organized from most to least active.

The dashboard's "offending side" and "targeted side" groupings are derived from ACLED's `inter1/inter2` actor-type codes and contextual recoding in the pipeline. They are analytical approximations, not definitive role assignments from ACLED.

"Associated actors" in the Actor Analysis page corresponds directly to ACLED's `assoc_actor_1/2` field — actors who appeared on the same side of the same event as the primary actor. Co-appearance does not imply formal alliance, command relationship, or pre-coordination.

## Limits And Biases

- ACLED reflects **reported** events, not every event that happened. Coverage gaps are unevenly distributed across time and geography.
- All fatality figures are **conservative lower-bound estimates**. They can be revised upward as ACLED updates source coverage.
- Fatalities are **not attributable to a specific actor** from ACLED data alone.
- **Air attack and Drone attack counts are understated** because ACLED's event-merging rule absorbs airstrikes that occur during ground battles into the Battles record (our Ground-based attack category).
- **Drone classification is a heuristic**. The pipeline flags `Air/drone strike` events as drone attacks when the notes contain "drone" or when the primary actor is not state forces. This approximation may misclassify some events.
- **Massacres is a dashboard-defined category** (civilian targeting flag + ≥5 fatalities), not a standard ACLED label. Comparisons with other ACLED-based analyses should account for this.
- Local actor naming varies across source reports. Our actor labels are standardized for readability and may not match all variant spellings in the raw data.
- Township choropleths show reported event or fatality density, not territorial control or ground-truth conflict intensity.
- Dashboard actor-side groupings should not be interpreted as definitive proof of who initiated violence or who bore all harm in an event.
