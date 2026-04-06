# UI Audit

Last updated: 2026-04-03

## Purpose

This audit evaluates the current local dashboard against:

- [PRODUCT_BRIEF.md](/Users/kaykay/Desktop/MyanmarConflictDashboard/docs/PRODUCT_BRIEF.md)
- [METRIC_GLOSSARY.md](/Users/kaykay/Desktop/MyanmarConflictDashboard/docs/METRIC_GLOSSARY.md)
- [LITERATURE_REVIEW.md](/Users/kaykay/Desktop/MyanmarConflictDashboard/docs/LITERATURE_REVIEW.md)

The goal is to identify the current interface problems before any redesign work begins.

## Audit Basis

This audit is based on:

- current runtime page structures in `pages/1_overview.py`, `pages/2_actor.py`, and `pages/3_about.py`
- current shared styles in `assets/style.css`
- current method/trust documentation in `docs/`
- recent local testing and iteration history from the current workspace

## Rating Scale

- `High`: directly conflicts with the product brief, glossary, or trust requirements
- `Medium`: weakens clarity, hierarchy, usability, or consistency
- `Low`: polish issue or secondary improvement area

## Current Strengths

- The product now has a clean three-page structure: `Overview`, `Actor Analysis`, and `About`.
- `Data through` and `Last checked with ACLED` should stay separate in the page headers.
- The app has good technical foundations for redesign: pre-aggregated monthly data, markdown-backed documentation, and a simpler runtime surface than before.
- Overview and Actor Analysis both already have the right broad analytical ingredients: map, trend, composition/association view, and filtering.
- The interface is visually cleaner than a default admin dashboard and avoids obvious framework clutter.
- Mobile filter toggles and sticky action buttons already exist as a starting point.

## Top Findings

### 1. Actor-side terminology is currently not method-safe

Severity: `High`

Where it appears:

- `pages/2_actor.py`
- current Actor Analysis KPI labels and trend subtitle

Problem:

The current UI still uses:

- `Offensive Events`
- `Defensive Events`
- `as attacker`
- `as targeted`

This conflicts with the metric glossary and ACLED guidance. ACLED does not treat `Actor1` and `Actor2` as aggressor/victim fields, and our actor page is built on dashboard-side analytical grouping rather than a native aggressor variable.

Why it matters:

- trust risk
- methodological overclaim
- likely to mislead non-expert users

Required redesign direction:

- replace these labels with method-safe primary-side / secondary-side wording
- add a visible actor-side caveat near the KPI row or trend subtitle

### 2. The pages still do not establish a single dominant analytical story fast enough

Severity: `High`

Where it appears:

- `pages/1_overview.py`
- `pages/2_actor.py`

Problem:

The main pages still ask the user to parse too many competing elements near the top:

- page header
- freshness badge
- large filter panel
- filter summary chips
- map card header
- map mode toggle
- metric toggle
- KPI cards

The result is better than before, but the product brief requirement of a clear first answer within 10 seconds is not yet reliably met.

Why it matters:

- weak first impression
- slower comprehension
- increased cognitive load

Required redesign direction:

- reduce above-the-fold competition
- choose one dominant visual or answer per page
- demote secondary controls and secondary summaries

### 3. Trust and interpretation support are still too far from the visuals they qualify

Severity: `High`

Where it appears:

- both main pages
- current bottom-of-page disclaimer pattern

Problem:

The current app has a bottom disclaimer, but it does not place the most important interpretation support close enough to the map, KPI labels, or Actor Analysis side metrics.

Missing or under-expressed cues:

- map caveat
- fatality estimate caveat
- actor-side caveat
- reminder that ACLED reflects reported events, not full event occurrence

Why it matters:

- users can overread the charts before they encounter the caveat
- methodology becomes an afterthought rather than part of the interface

Required redesign direction:

- move key caveats closer to the relevant views
- use short trust notes in-page, not only one footer disclaimer

### 4. The map is structurally central, but interpretively under-supported

Severity: `High`

Where it appears:

- `pages/1_overview.py`
- `pages/2_actor.py`

Problem:

The left rail is organized around a large sticky choropleth. That makes sense only if the map can carry the page strongly. Right now the map remains visually dominant, but the explanatory support around it is still too weak.

Examples:

- Overview map subtitle is too generic
- Actor map subtitle is too vague and colloquial
- there is no short explanation of what the shading does not mean
- the map and support rail still feel like parallel content blocks rather than one story

Why it matters:

- choropleths are easy to overread
- the page makes the map look more definitive than it is

Required redesign direction:

- keep map importance, but make the support rail more explicitly interpretive
- give the map a stronger companion explanation or ranking cue

## Overview Page Findings

### 5. Overview still lacks a clear primary message

Severity: `High`

Problem:

The Overview page contains the right pieces, but they are still arranged more like a capable dashboard than a clearly led analytical experience. `Conflict Geography`, `Reported Events`, `Reported Fatality Estimate`, `Event Trend`, and `Incident Types` all compete for top-tier attention.

Recommended change:

- decide what the Overview page is primarily saying
- make the other views support that primary message

### 6. KPI language is not yet fully aligned with the glossary

Severity: `High`

Problem:

The current UI still uses `Recorded Events` and `Recorded Fatalities`. The glossary now prefers:

- `Reported Events`
- `Reported Fatality Estimate`

Recommended change:

- update the KPI labels and subtitle wording during the redesign pass

### 7. The event/fatality metric toggle is still too prominent too early

Severity: `Medium`

Problem:

The user is asked to switch between `Events` and `Fatalities` before the map’s interpretive frame is fully established. This is especially risky because fatalities are more method-sensitive and easier to overread.

Recommended change:

- visually demote the toggle
- consider treating fatalities as a secondary view rather than a co-equal first interaction

### 8. `Most Active Township` needs clearer metric context

Severity: `Medium`

Problem:

The label is acceptable shorthand, but it does not say that the ranking is by reported event count in the selected scope.

Recommended change:

- keep the short label if needed
- add supporting clarification nearby or in the subtitle logic

### 9. The filter summary chip row adds useful context but contributes to stacking pressure

Severity: `Medium`

Problem:

The chip row is clearer than the old sentence summary, but it still creates another full-width band between filters and content.

Recommended change:

- keep the idea
- reconsider placement, size, or whether part of it should merge into the page header or map card

## Actor Analysis Findings

### 10. Actor Analysis still overclaims role semantics

Severity: `High`

Problem:

This is the most important issue on the Actor page. The page still suggests offensive and defensive roles in a way that exceeds what the source fields directly support.

Recommended change:

- rewrite KPI row
- rewrite trend subtitle
- add actor-side interpretation note

### 11. `Associated Actors` can still be overread as alliance evidence

Severity: `High`

Problem:

The subtitle `Co-involved in the same armed conflict events` is better than calling them allies, but the surrounding presentation still invites overinterpretation.

Recommended change:

- keep the concept
- add explicit wording that this reflects same-side co-appearance in the dashboard recode, not proof of alliance or command structure

### 12. The page header and actor banner compete for hierarchy

Severity: `Medium`

Problem:

The page has:

- main page header
- amber armed-conflict subtitle
- actor banner
- filter card

This creates a long pre-analysis lead-in before the user reaches the actual actor evidence.

Recommended change:

- collapse actor framing into a stronger, more compact top section

### 13. `Point at a township · blue = high activity` is too weak as map support copy

Severity: `Medium`

Problem:

The subtitle is informal and underspecified for a serious public-facing dashboard.

Recommended change:

- replace with copy that states what the map is showing in analytical terms

## About Page Findings

### 14. About is accurate, but not yet optimized for onboarding

Severity: `Medium`

Where it appears:

- `pages/3_about.py`
- `docs/ABOUT.md`

Problem:

The About page is currently a well-structured markdown document, but it still reads more like project documentation than a first-time onboarding surface.

Missing or weak elements:

- quick “how to read this dashboard” section near the top
- concise trust summary box
- prominent explanation of who the product is for

Recommended change:

- keep markdown-backed approach
- improve content structure rather than rebuilding the route architecture

## Mobile Findings

### 15. Mobile still behaves like a collapsed desktop interface

Severity: `High`

Where it appears:

- `assets/style.css`
- both main pages

Problem:

The app now has mobile filter toggles and stacked layouts, but the overall information architecture is still desktop-first. On mobile, the product likely remains too dense and too sequential.

Examples:

- same content blocks mostly survive, just stacked
- no major reprioritization of page sections
- the map still occupies a lot of attention before supporting interpretation is established

Recommended change:

- define separate mobile wireframes
- decide what should appear first on mobile instead of merely stacking desktop elements

### 16. Support text is still too small in several mobile-relevant places

Severity: `Medium`

Problem:

The current CSS still uses many values around `0.60rem` to `0.80rem` for labels, subtitles, and helper text. This is better than default framework output, but still too tight for a public-facing dashboard with serious content.

Recommended change:

- raise the minimum support-text scale in the redesign system

## Visual System Findings

### 17. The visual language is coherent but still not distinctive enough

Severity: `Medium`

Problem:

The app is clean, but it still reads as a refined analytics template rather than a fully intentional product. Repeated card patterns and similar visual weights flatten the hierarchy.

Recommended change:

- reduce sameness between major and minor cards
- create a stronger rhythm between hero, map, support rail, and trust/status elements

### 18. Card weight and spacing are still too uniform

Severity: `Medium`

Problem:

Many cards use similar shadow, radius, border, and padding patterns, which reduces information hierarchy.

Recommended change:

- differentiate major analytical surfaces from secondary support surfaces

## Performance And Interaction Findings

### 19. Performance has improved, but the redesign should not assume the job is done

Severity: `Medium`

Problem:

Important first-load improvements have already been made, but the current page model still depends on several sizable figure updates and repeated Python-side filtering logic.

Recommended change:

- treat performance as part of the redesign, not as a postscript
- profile the redesigned Overview before pushing

### 20. The current UI still exposes too many first-order controls

Severity: `Medium`

Problem:

Even after cleanup, the interface still puts filters, mode cards, metric toggle, chips, KPI cards, and support charts into the same user attention band.

Recommended change:

- reduce the number of controls that feel primary
- clarify which controls are essential for first-time use

## Summary Judgment

The current app is not broken. It is a credible, working analytical dashboard with a stronger technical and documentation base than before.

However, it is not yet ready for a push-level UI redesign release because:

- key Actor Analysis terminology remains method-unsafe
- the page hierarchy still does not strongly communicate one primary answer per page
- trust cues are still too detached from the visuals they qualify
- mobile is still mostly a collapsed desktop experience

## Priority Order For Redesign

1. Fix method-sensitive Actor Analysis terminology in the redesigned UI.
2. Rebuild Overview around one dominant national picture.
3. Bring trust and interpretation support closer to maps and KPIs.
4. Create separate mobile wireframes rather than relying on stacking.
5. Rebuild Actor Analysis only after the new Overview structure is approved.

## Recommended Next Artifact

The next deliverable should be:

- `docs/IA_AND_WIREFRAMES.md`

That file should translate this audit into low-fidelity layouts for:

- Overview desktop
- Overview mobile
- Actor Analysis desktop
- Actor Analysis mobile
