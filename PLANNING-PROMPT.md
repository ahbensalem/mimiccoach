# MimicCoach — Project Planning Brief

## Context
We are building a submission for the Qdrant "Think Outside the Bot" Virtual Hackathon.
- Deadline: 2026-06-01.
- Rules: must use Qdrant as a material part; no chatbots; all code written during the
  hackathon period; submit GitHub repo + README + demo video (max 3 min).
- Judging: technical functionality, creativity, innovative use of vector search, UX.
- Prizes: $5K / $3K / $2K + best-in-category bonuses.
- Development assist: building with the Claude Code assistant, so calendar-day
  budgeting is not the binding constraint — plan by logical phases, not days.

## Idea
MimicCoach is a self-coaching tool. A user uploads a phone video of themselves
performing one of 7 supported motions. The app extracts a pose sequence with
MediaPipe, segments it into motion phases, encodes each phase as a token vector
using a pre-trained pose embedding model, and queries Qdrant against a curated
library of pro reference clips using late-interaction multivector search (MaxSim).
Output: closest pro match, side-by-side overlay, per-phase similarity scores,
and a coaching tip targeting the weakest phase.

The Qdrant feature this showcases is **late-interaction multivector storage with
MaxSim** applied to motion (pose-token sequences), the way ColBERT/ColPali use
it for documents. No public Qdrant demo currently does pose-based motion retrieval.

## Sports & motions (in scope)
1. **Tennis** — serve, forehand, backhand
2. **Fitness** — squat, bench press, bent-over row
3. **Golf** — full swing

Total: 7 motions. Phases per motion vary (5–6 typical). The plan must define
explicit phase boundaries for each motion.

**Hero motion** (gets the most polish, anchors the demo video): tennis serve.
Other 6 motions must be functional but can be lower-fidelity.

## Tech stack (decided)
- **Pose extraction**: MediaPipe Pose Landmarker (33 keypoints, free, CPU-fast).
- **Pose embedding**: pre-trained model — planner to choose between MotionBERT,
  PoseC3D, ST-GCN++, or a simpler hand-crafted joint-angle/velocity feature
  vector projected to fixed dim. Bias toward whichever is fastest to integrate
  and produces stable per-phase tokens.
- **Vector DB**: Qdrant Cloud free tier (1 GB), multivector collection per sport
  with MaxSim late-interaction scoring.
- **Backend**: Python on Modal (free $30/mo credit, GPU on demand).
- **Frontend**: Next.js on Vercel.
- **Input**: upload-only (MP4 from phone). No live webcam in MVP.

## Reference library
- Hybrid acquisition:
  - Open datasets: Penn Action (tennis_serve, golf_swing, bench_press, squat)
    and any other open pose/action datasets that cover our motions.
  - YouTube scraping for the 3 motions not covered (forehand, backhand,
    bent-over row) and to top up volume — target ~30–50 clips per motion.
- Each clip carries a payload: { sport, motion, athlete, skill_level
  (beginner/intermediate/pro), body_type (e.g., light/medium/heavy or
  height bucket), source_url, license_note }.
- Skill level and body type labels are required (not optional) — filter
  support is a judging differentiator.

## Filters (must work end-to-end)
- sport (tennis | fitness | golf)
- skill_level (beginner | intermediate | pro)
- body_type (configurable bucketing — propose schema)

All three filters must be honored at query time using Qdrant payload-indexed
filtering combined with multivector search.

## The demo video (3-minute north star)
The plan must work backward from this video. Storyboard:
1. (0:00–0:15) Cold open: split screen, user's amateur tennis serve next to
   a pro serve, skeletons overlaid, color-coded by phase.
2. (0:15–0:45) Floating per-phase scores appear: e.g., "Toss: 0.78 — Trophy:
   0.62 — Contact: 0.42 ← weakest — Follow-through: 0.91". One-sentence
   coaching tip generated from the weakest-phase diff.
3. (0:45–1:30) Quick montage of the other 6 motions working: squat, bench,
   row, golf, forehand, backhand. 5–10s each.
4. (1:30–2:15) Architecture explainer: animate the multivector storage and
   MaxSim query against the Qdrant collection. Emphasize "this is what
   ColBERT does for text — we are doing it for motion."
5. (2:15–3:00) Filter demo (skill level / body type), cherry-picked judges-
   impressing query, closing card with GitHub URL.

Every planning decision should pass the filter: "does this make the 3-min
video stronger or just churn?"

## Open questions the plan must resolve
1. Phase segmentation strategy: auto-detect via velocity/acceleration zero-
   crossings on key joints, or hand-annotate phase timestamps for the
   reference library and use a learned classifier on user uploads? Pick one,
   justify, and define phases for each of the 7 motions.
2. Pose embedding model choice — pick exactly one and lock it.
3. Body type bucketing scheme — propose 3–5 buckets that are auto-derivable
   from MediaPipe keypoints (e.g., shoulder-to-hip ratio, limb length proxy)
   so we don't need to manually label every clip.
4. Number of phase tokens per motion (one vector per phase, or k tokens
   sampled within each phase?). Affects MaxSim quality and Qdrant cost.
5. How to render the side-by-side overlay client-side in Next.js — canvas
   on top of a <video> element with synchronized playback, or pre-render
   the diff video server-side on Modal? Plan must pick.
6. Coaching-tip generation: rule-based from joint-angle deltas (preferred,
   no LLM dependency, no chatbot risk) vs. small templated LLM call.
   Default to rule-based unless plan justifies otherwise.

## Constraints / non-goals
- No chatbot UI. Interaction is upload → result page.
- No mobile native app. Mobile web is fine if responsive.
- No live multi-user features.
- No fine-tuning a pose model from scratch. Pre-trained only.
- All third-party content (YouTube clips) must have license notes captured
  in payload; favor Creative Commons sources where possible.

## Deliverables expected from the plan
- Phased delivery schedule organized as logical phases (not calendar days),
  with explicit dependencies between phases and a clearly marked
  "demo-recordable" milestone before the final video + README polish phase.
- Risk register with top 5 risks and mitigations (data acquisition, phase
  segmentation accuracy, pose embedding stability, Modal/Vercel cold-start
  latency on demo day, license issues).
- Component-level architecture with file/module layout for backend and
  frontend.
- Concrete dataset acquisition checklist: what to pull from Penn Action,
  what to scrape from YouTube, target counts per motion, labeling workflow.
- Qdrant schema: one collection per sport vs. one collection total; named
  vectors per phase vs. multivector arrays; payload index definitions.
- A "could be cut" list — features that ship if time allows, drop cleanly
  if not.

## Team
Solo lead with optional collaborators to be added. Plan should identify
2–3 well-bounded chunks that can be parallelized if collaborators come on
board (good candidates: reference library curation, frontend overlay
rendering, demo video production).

Begin with the phased schedule and the dataset acquisition checklist,
since those are the highest-risk items.
