# Demo video storyboard — 3 minutes

The 3-minute video is the artifact every planning decision optimized for. Hit
five beats in this order. Voiceover lines are written to be readable at a
calm 140 WPM.

## Beat 1 — Cold open (0:00–0:15)

**On screen.** Your phone tennis-serve clip on the left. The closest pro
serve on the right. Both videos play in sync. MimicCoach skeletons overlay
both in matching phase colors (red → orange → amber → green → blue).

**Voiceover.** "I uploaded a phone clip of my tennis serve. MimicCoach
matched it against a library of pro serves and ran a side-by-side. Watch the
two skeletons line up — phase by phase."

**Tip.** Pre-record both videos before recording the screen so the playback
is rock-solid. Use your warmest Modal container (run a `/analyze` call right
before recording starts).

---

## Beat 2 — Per-phase scores + coaching tip (0:15–0:45)

**On screen.** Score chips appear one at a time on top of the split video.
Trophy chip turns amber and grows ("weakest" badge). Coaching-tip card
slides up from below with the rule-based one-liner.

**Voiceover.** "MimicCoach doesn't just average the whole motion into one
similarity score. It stores each clip as five-to-six pose tokens — one per
phase of the serve — and queries Qdrant with late-interaction MaxSim. The
output is a per-phase score breakdown. My toss is a 0.78. My contact, 0.84.
My trophy position is 0.62 — and that's where the coaching tip targets."

**Tip.** Expose the `coaching_tip` and the per-phase scores from the
`/api/proxy/analyze` JSON response — they're already in the result page UI.

---

## Beat 3 — All 7 motions montage (0:45–1:30)

**On screen.** Quick cuts, ~6 seconds per motion: squat → bench → row →
forehand → backhand → golf swing. Same UI shape every time: split video,
score chips, focus phase highlighted.

**Voiceover.** "It's not just tennis. The same pipeline runs on barbell
squats, bench press, bent-over rows, the tennis forehand and backhand, and
the golf swing. Seven motions, twenty-seven phases total — all encoded as
multivector points in one Qdrant collection."

**Tip.** Pre-render each clip's analysis on a separate browser tab to keep
the cuts fast. Don't try to live-upload all 7 in 45 seconds.

---

## Beat 4 — Architecture beat (1:30–2:15)

**On screen.** Animated diagram (or screen-record the ASCII diagram from
docs/architecture.md while you talk). Highlight the multivector point and
the MaxSim arrow. Cut briefly to a snippet of `qdrant_io/schema.py` showing
the `MultiVectorConfig(comparator=MAX_SIM)` line.

**Voiceover.** "Here's why this works. ColBERT and ColPali use late-
interaction multivector retrieval to compare documents at the token level —
not just one averaged vector per document. We're doing the same thing, but
each *token* is a phase of a sport motion. The user's tokens go up against
every stored point's tokens, MaxSim picks the best alignment per phase, and
Qdrant aggregates. It's ColBERT for motion."

**Tip.** This is the soundbite-worthy beat. Lead with "ColBERT for motion"
and let it land.

---

## Beat 5 — Filter demo + close (2:15–3:00)

**On screen.** UI is showing a result for a tennis serve. You toggle the
"skill_level: pro" chip — the top match changes. You toggle "body_type:
broad" — it changes again. Each filter swing is a visible cut.

**Voiceover.** "And because Qdrant's payload-indexed filters run alongside
the multivector query, you can ask 'show me only pros' or 'show me pros
with my body type' — without re-encoding, without a second roundtrip.
That's it. The repo is on GitHub. Thanks for watching."

**On screen at the very end (5–10 seconds).** Closing card:
```
github.com/<your-username>/mimiccoach
MimicCoach · Late-interaction pose retrieval with Qdrant
```

**Tip.** Pre-stage two filter combinations that you've verified return
visibly different top matches against your reference library. Don't gamble
on live filter swings if your library is sparse.

---

## Production checklist

- [ ] Vercel deploy is live
- [ ] Modal `fastapi_app` is warm (run `/healthz` yourself once before recording)
- [ ] Browser dev tools closed, fullscreen mode on
- [ ] Reference library has at least 5 clips per motion (synthetic library is enough)
- [ ] Sample user clip per motion ready to upload (or already pre-loaded)
- [ ] Screen recording at 1920×1080 minimum, 60 fps where possible
- [ ] Voiceover recorded separately and laid over the cut (cleaner than live)
- [ ] Music: avoid copyrighted tracks; YouTube CC audio library or YouTube
      Audio Library are safe defaults
- [ ] Final cut ≤ 3:00 (the rules cap it; check before render)

## Tools

- [OBS Studio](https://obsproject.com/) for screen recording
- [DaVinci Resolve (free)](https://www.blackmagicdesign.com/products/davinciresolve) for editing
- [Audacity](https://www.audacityteam.org/) for voiceover
