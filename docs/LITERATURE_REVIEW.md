# Dashboard Literature Review

Last updated: 2026-04-03

## Purpose

This note captures the research base we are using to guide the next redesign phase of the Myanmar Conflict Dashboard. The goal is not to collect every paper on dashboards, but to identify high-signal evidence that should shape design, content, interaction, performance, and trust decisions for a public-facing conflict-data web app.

## Scope and Search Approach

Searches were conducted on 2026-04-03 across:

- dashboard design and dashboard typologies
- usability and usefulness of public-health dashboards
- data visualization and graphical perception
- choropleth and animated map design
- mobile map user experience
- conflict event data quality and interpretation
- official Plotly Dash performance guidance
- ACLED methodology and update guidance

Priority was given to peer-reviewed studies, systematic reviews, open-access academic sources, and official documentation.

## Executive Takeaways

- Dashboards work best when they are designed for a clearly defined decision context, not as catch-all data portals.
- Simple, consistent layouts and familiar chart types repeatedly outperform dense or novel interfaces, especially for mixed audiences.
- Public-facing dashboards need stronger interpretation support than expert-only dashboards: definitions, guidance, context, and explicit limits.
- Choropleth maps are useful for answering "where" questions, but they carry interpretation risks around aggregation, small-number problems, and ecological fallacy.
- Animated choropleths are attractive but cognitively fragile. They are better as a secondary exploratory mode than as the primary way to communicate spatial patterns.
- Conflict-event data require strong uncertainty communication. Fatalities are conservative reported estimates, can change over time, and usually cannot be attributed to specific actors from ACLED alone.
- For Dash specifically, callback cost, serialization cost, and redundant recomputation are central performance risks; caching and client-side work are first-line remedies.

## Core Evidence

### 1. Dashboard design is a distinct design problem

- Sarikaya et al. argue that dashboards are not just small visual analytics systems; they differ in design goals, interaction levels, and use practices. This matters because our app should not try to behave like a full exploratory analysis workbench for every visitor. It needs a clearer primary job.
  Source: [What Do We Talk About When We Talk About Dashboards?](https://pubmed.ncbi.nlm.nih.gov/30136958/)

- A 2024 scoping review of US public-health dashboards found rapid growth in dashboards but sparse and inconsistent rigorous evidence on optimal design, implementation, and evaluation. That suggests we should combine available evidence with disciplined local testing, rather than assuming "common dashboard patterns" are automatically good.
  Source: [Design, Application, and Actionability of US Public Health Data Dashboards: Scoping Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12138306/)

- A 2021 systematic review of patient-safety dashboards found that dashboard development rarely incorporated human-factors principles or post-implementation usability evaluation. Inference for this project: we should treat usability testing as a core workstream, not a polish step.
  Source: [Dashboards for visual display of patient safety data: a systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8496385/)

### 2. Usability and usefulness depend on audience fit

- Ansari and Martin developed a usability checklist for public-health dashboards organized around 11 principles: spatial organization, information coding, consistency, removal of extraneous ink, recognition rather than recall, minimal action, dataset reduction, flexibility to user experience, understandability of contents, scientific integrity, and readability. They found the biggest problems in understandability, flexibility, and scientific integrity.
  Source: [Development of a usability checklist for public health dashboards to identify violations of usability principles](https://pmc.ncbi.nlm.nih.gov/articles/PMC9552210/)

- In a later case study with domain experts, Ansari found five recurring requirements for usable public-health dashboards: familiar charts with clear legends and labels, a simple and consistent layout, contextual information for interpretation, clear communication of data limitations, and guidance for user interaction. This is directly relevant to our project.
  Source: [Evaluating the usability of public health data dashboards as information sources for professionals and the public](https://pmc.ncbi.nlm.nih.gov/articles/PMC12723333/)

- Yanovitzky et al. frame useful dashboards as tools that provide timely, relevant, credible, and actionable information. Their 2025 study also found that many dashboards do not explicitly identify intended users and many lack stronger credibility or accessibility affordances. Project inference: our dashboard should say who it is for and how to use it, not assume that is obvious.
  Source: [Usability and usefulness of U. S. federal and state public health data dashboards](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695813/)

- Ansari and Martin's human-centered design case study for New York State STI dashboards shows that stakeholder requirements, expert review, user evaluation, and prototype usability testing each changed chart choice, interactivity, prompts, and data notes. Project inference: we should expect the final layout to be shaped by iterative feedback, not one design pass.
  Source: [Integrating human-centered design in public health data dashboards](https://pmc.ncbi.nlm.nih.gov/articles/PMC10797265/)

### 3. Visual perception should guide chart choice

- Cleveland and McGill's foundational work, replicated and extended by Heer and Bostock, supports the long-standing principle that viewers judge position and aligned length more accurately than area, angle, or color-based encodings. Project implication: bar and line charts should carry the key comparisons; area-heavy displays and decorative KPI treatments should stay secondary.
  Sources:
  [Cleveland & McGill, 1984](https://doi.org/10.1080/01621459.1984.10478080)
  [Heer & Bostock, 2010](https://idl.uw.edu/papers/crowdsourcing-graphical-perception)

- A 2024 eye-tracking study on dashboard layout order found that logical layout organization affects interface complexity and visual search. Project inference: the relationship between sections matters, not just the quality of each individual card.
  Source: [The Effects of Layout Order on Interface Complexity: An Eye-Tracking Study for Dashboard Design](https://www.mdpi.com/1424-8220/24/18/5966)

### 4. Choropleth maps are useful, but they need guardrails

- CDC guidance on choropleth design emphasizes that map purpose and audience should be defined early, that end users should be involved during development, and that maps alone are often insufficient: tables, charts, and explanatory text may also be needed to answer users' questions.
  Sources:
  [Choropleth Map Design for Cancer Incidence, Part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811518/)
  [Choropleth Map Design for Cancer Incidence, Part 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811519/)

- The same CDC guidance warns about modifiable areal unit effects, ecological fallacy, small-number instability, and misleading assumptions that values are uniform within each geographic unit. Project implication: hotspot maps should be paired with explicit caveats and secondary charts, not treated as self-evident truth.
  Source: [Choropleth Map Design for Cancer Incidence, Part 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811519/)

- Research on choropleth legend design suggests that readers benefit when the legend helps them interpret both geographic pattern and underlying data distribution. Project inference: our legends and subtitles should do more explanatory work than a bare scale.
  Source: [Choropleth map legend design for visualizing community health disparities](https://pmc.ncbi.nlm.nih.gov/articles/PMC2760860/)

### 5. Animated maps have real cognitive limits

- Fish et al. found that readers often fail to detect important changes in animated choropleths and tend to overestimate their own change-detection ability. This is a strong argument against making animation the primary communication mode for key findings.
  Source: [Change Blindness in Animated Choropleth Maps: An Empirical Study](https://www.tandfonline.com/doi/abs/10.1559/15230406384350)

- Cybulski's 2022 study found animated choropleths are more suitable for presenting temporal trends than spatial patterns, and recommended highlighting, redundancy, or interactive aids to support recognition. Project implication: animation should be paired with controls, annotations, and a static summary mode.
  Source: [An Empirical Study on the Effects of Temporal Trends in Spatial Patterns on Animated Choropleth Maps](https://www.mdpi.com/2220-9964/11/5/273)

- Cybulski and Medyńska-Gulij found cartographic redundancy can reduce change blindness in spatio-temporal maps. Project inference: if we keep animated mode, we should consider stronger visual cues for important changes instead of relying on color shifts alone.
  Source: [Cartographic Redundancy in Reducing Change Blindness in Detecting Extreme Values in Spatio-Temporal Maps](https://www.mdpi.com/2220-9964/7/1/8)

### 6. Mobile map UX deserves separate treatment

- Somaskantharajan et al. studied choropleth and graduated-symbol map UX on mobile devices and found that UX outcomes change with design variables like color distance. They also note that mobile map design needs explicit modeling rather than assuming desktop design transfers cleanly.
  Source: [An Exploratory Study of Models of Mobile Map User Experience](https://link.springer.com/article/10.1007/s42489-023-00136-8)

- The 2022 public-health dashboard usability checklist study explicitly asked reviewers to check mobile friendliness because dashboard readability and interpretability can degrade quickly on smaller screens. Project implication: mobile should be evaluated as its own interface, not only as a responsive variant.
  Source: [Development of a usability checklist for public health dashboards to identify violations of usability principles](https://pmc.ncbi.nlm.nih.gov/articles/PMC9552210/)

### 7. Conflict-event data need visible uncertainty handling

- ACLED's codebook and fatality methodology emphasize that fatalities are reported estimates, are often the most biased and least accurate component of conflict reporting, are revised over time, and are not generally attributable to one actor or another. Project implication: our UI should avoid any copy that implies exact or actor-specific fatality attribution unless supported by methodology.
  Sources:
  [ACLED Codebook](https://acleddata.com/knowledge-base/codebook/)
  [ACLED Fatality Methodology](https://acleddata.com/knowledge-base/faqs-acled-fatality-methodology/)
  [ACLED FAQ on actor-specific fatalities](https://acleddata.com/faq/it-possible-identify-number-people-killed-specific-group-or-total-number-civilians-killed)

- ACLED documents that the dataset is updated weekly and that deleted events must also be processed to stay aligned with the current dataset. Project implication: freshness should communicate both event coverage and sync date, and internal update logic must handle deletions as well as additions.
  Sources:
  [ACLED Update Log](https://acleddata.com/conflict-data/knowledge-base/update-log)
  [ACLED Deleted Endpoint](https://acleddata.com/api-documentation/deleted-endpoint)

- Demarest and Langer's Total Event Error framework shows that conflict-event data are vulnerable to source-selection effects, measurement ambiguity, coder decisions, and under-coverage. They specifically note that local sources can be better suited for low-level events and subnational analysis. Project implication: a township-level Myanmar dashboard should foreground source structure and caution against treating mapped event density as complete ground truth.
  Source: [How Events Enter (or Not) Data Sets: The Pitfalls and Guidelines of Using Newspapers in the Study of Conflict](https://journals.sagepub.com/doi/10.1177/0049124119882453)

- Schutte and Kelling show that spatial conflict-event studies can generate false inference under plausible conditions. Project inference: we should be careful with "hotspot" language and treat spatial clustering as descriptive unless we perform more rigorous spatial analysis.
  Source: [A Monte Carlo analysis of false inference in spatial conflict event studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8982878/)

### 8. Dash web-app performance is part of UX

- Official Dash guidance states that callback cost is usually the main performance bottleneck and recommends memoization, background callback caching, clientside callbacks, partial property updates, and more efficient serialization. Project implication: perceived UX quality depends as much on callback architecture as on visual polish.
  Source: [Dash Performance Documentation](https://dash.plotly.com/performance)

- The same guidance notes that SVG rendering can slow down with large datasets and that WebGL variants may be better for some chart types. Project inference: map and trend designs should be chosen with rendering cost in mind, especially if we later add denser point or line views.
  Source: [Dash Performance Documentation](https://dash.plotly.com/performance)

### 9. Official Dash exemplars are useful pattern libraries

- Plotly's official examples page and Dash sample-apps repository show that strong public-facing Dash apps usually have a clear story, a limited number of primary interactions, and a tighter relationship between page purpose and layout. Project implication: our redesign should benchmark against mature Dash patterns, not just generic dashboard aesthetics.
  Sources:
  [Plotly Dash App Examples](https://plotly.com/examples/)
  [Dash Sample Apps Repository](https://github.com/plotly/dash-sample-apps)

- Public-facing Dash exemplars such as OECD Pensions Explorer, Femicide in Bolivia, and Centre de Controle des Incidents suggest three recurring patterns: question-led storytelling, a strong first visual, and support views that extend rather than compete with the primary analysis surface. Project implication: our Overview page should choose one dominant analytical frame and let the other visuals support it.
  Source: [Plotly Dash App Examples](https://plotly.com/examples/)

- Plotly's Explore Page submission guidance emphasizes polished presentation, distinct story, practical usefulness, and often live or regularly refreshed data. This is not a scientific paper, but it is a useful signal of what the Dash ecosystem itself treats as a strong exemplar.
  Source: [Share Your App - Explore Page - March 2026](https://community.plotly.com/t/share-your-app-explore-page-march-2026/96324)

## Evidence-Based Design Rules For This Project

These are working rules derived from the literature above.

### A. Clarify audience and task

- Pick a primary audience for each page.
- Design each page around a short list of real questions users need answered.
- Avoid building every page for researchers, journalists, advocates, and the general public equally.

### B. Keep hierarchy strong and interaction purposeful

- Put the main question and answer at the top of the page.
- Use one primary visual per page and a small number of secondary views.
- Make filters support the analysis, not dominate first impression.
- Use familiar chart forms unless there is strong evidence that novelty improves comprehension.

### C. Treat maps as analytical context, not self-sufficient truth

- Use choropleths mainly to answer "where" questions.
- Pair maps with ranking/trend/context views.
- Explain the geographic unit and what the shading does and does not mean.
- Be explicit about aggregation risks, small numbers, and interpretation limits.

### D. Treat animation as optional exploration

- Keep a strong static default.
- Use annotations, event markers, and redundancy if animation remains.
- Do not rely on users noticing subtle temporal shifts unaided.

### E. Make trust visible

- Separate "data through" from "synced" dates.
- Keep methodology and data-limit notes close to the charts they qualify.
- Avoid wording that overclaims precision, causality, or actor-specific fatalities.
- Make source provenance, update cadence, and download/export options easy to find.

### F. Design mobile deliberately

- Do not assume desktop layouts scale down cleanly.
- Use simpler interaction patterns on mobile.
- Keep text readable without zooming.
- Reduce side-by-side comparisons and control density on small screens.

### G. Treat performance as a first-class UX feature

- Cache repeated computations.
- Avoid shipping very large initial payloads.
- Prefer client-side updates where appropriate.
- Minimize figure rebuilds when only small properties change.

## Immediate Implications For Myanmar Conflict Dashboard

These are project-specific inferences from the literature and the current app state.

- The Overview page should communicate one main story first, then let users branch into exploration.
- The default map mode should remain static; animation should be secondary and explicitly exploratory.
- The map should be paired with ranking, trend, and interpretation aids because township shading alone is easy to overread.
- Labels such as `fatalities`, `defensive`, `offensive`, `most active`, and `recorded` need method-aware wording.
- Data notes should explicitly distinguish between reported event counts and reported fatality estimates.
- If we keep a public audience, we need more glossary-like support, not less.
- If we optimize for expert users, we should say that openly and reduce explanatory clutter elsewhere.

## Proposed Evaluation Checklist For Our Next Phase

Before pushing a redesign, test each candidate page against these questions:

- Can a first-time visitor state what the page is for within 10 seconds?
- Can they identify the primary metric or conclusion without interacting?
- Can they explain what the map colors mean and what they do not mean?
- Can they find the time frame and data freshness without hunting?
- Can they use filters without losing page context?
- Can they interpret fatalities without assuming false precision?
- Can they complete the same tasks on mobile without frustration?
- Does the page remain responsive when filters change repeatedly?

## Suggested Next Work Sequence

1. Define the primary audience and primary task for each page.
2. Translate the evidence above into a page-level design rubric.
3. Audit the current Overview page against that rubric.
4. Redesign the Overview page locally.
5. Run quick usability checks on desktop and mobile before any push.

## Source List

- [What Do We Talk About When We Talk About Dashboards?](https://pubmed.ncbi.nlm.nih.gov/30136958/)
- [Dashboards for visual display of patient safety data: a systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8496385/)
- [Development of a usability checklist for public health dashboards to identify violations of usability principles](https://pmc.ncbi.nlm.nih.gov/articles/PMC9552210/)
- [Evaluating the usability of public health data dashboards as information sources for professionals and the public](https://pmc.ncbi.nlm.nih.gov/articles/PMC12723333/)
- [Usability and usefulness of U. S. federal and state public health data dashboards](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695813/)
- [Integrating human-centered design in public health data dashboards](https://pmc.ncbi.nlm.nih.gov/articles/PMC10797265/)
- [Design, Application, and Actionability of US Public Health Data Dashboards: Scoping Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12138306/)
- [Cleveland & McGill, 1984](https://doi.org/10.1080/01621459.1984.10478080)
- [Crowdsourcing Graphical Perception](https://idl.uw.edu/papers/crowdsourcing-graphical-perception)
- [The Effects of Layout Order on Interface Complexity: An Eye-Tracking Study for Dashboard Design](https://www.mdpi.com/1424-8220/24/18/5966)
- [Choropleth Map Design for Cancer Incidence, Part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811518/)
- [Choropleth Map Design for Cancer Incidence, Part 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811519/)
- [Choropleth map legend design for visualizing community health disparities](https://pmc.ncbi.nlm.nih.gov/articles/PMC2760860/)
- [Change Blindness in Animated Choropleth Maps: An Empirical Study](https://www.tandfonline.com/doi/abs/10.1559/15230406384350)
- [An Empirical Study on the Effects of Temporal Trends in Spatial Patterns on Animated Choropleth Maps](https://www.mdpi.com/2220-9964/11/5/273)
- [Cartographic Redundancy in Reducing Change Blindness in Detecting Extreme Values in Spatio-Temporal Maps](https://www.mdpi.com/2220-9964/7/1/8)
- [An Exploratory Study of Models of Mobile Map User Experience](https://link.springer.com/article/10.1007/s42489-023-00136-8)
- [ACLED Codebook](https://acleddata.com/knowledge-base/codebook/)
- [ACLED Fatality Methodology](https://acleddata.com/knowledge-base/faqs-acled-fatality-methodology/)
- [ACLED FAQ on actor-specific fatalities](https://acleddata.com/faq/it-possible-identify-number-people-killed-specific-group-or-total-number-civilians-killed)
- [ACLED Update Log](https://acleddata.com/conflict-data/knowledge-base/update-log)
- [ACLED Deleted Endpoint](https://acleddata.com/api-documentation/deleted-endpoint)
- [How Events Enter (or Not) Data Sets: The Pitfalls and Guidelines of Using Newspapers in the Study of Conflict](https://journals.sagepub.com/doi/10.1177/0049124119882453)
- [A Monte Carlo analysis of false inference in spatial conflict event studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8982878/)
- [Dash Performance Documentation](https://dash.plotly.com/performance)
- [Plotly Dash App Examples](https://plotly.com/examples/)
- [Dash Sample Apps Repository](https://github.com/plotly/dash-sample-apps)
- [Share Your App - Explore Page - March 2026](https://community.plotly.com/t/share-your-app-explore-page-march-2026/96324)
