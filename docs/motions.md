# Motion phase definitions

This file describes each motion's phase boundaries in human terms. The
machine-readable version lives at [`backend/pipeline/motions.yaml`](../backend/pipeline/motions.yaml)
— that is the source of truth.

## Tennis serve (HERO motion)

| Phase | Description | Key joint signal |
|---|---|---|
| Stance | Pre-serve setup, ball held against racket | clip start |
| Toss | Tossing arm rises, ball released | tossing-arm wrist Y peak |
| Trophy | Hitting arm cocked back, knees flex | hitting-arm elbow angle minimum (max flex) |
| Acceleration / Contact | Racket accelerates up, contact ball | hitting-arm wrist Y peak |
| Follow-through | Racket finishes down across body | clip end |

## Tennis forehand / backhand

| Phase | Description |
|---|---|
| Ready | Athlete in split-step / neutral stance |
| Take-back | Racket goes back (horizontal velocity reverses negative) |
| Forward-swing | Racket reverses forward toward ball |
| Contact | Peak forward velocity at ball contact |
| Follow-through | Swing completes |

## Barbell back squat

| Phase | Description |
|---|---|
| Setup | Bar on back, athlete standing |
| Descent | Hips travel down (vertical velocity becomes positive-downward) |
| Bottom | Hips at lowest point (velocity zero-crosses to upward) |
| Ascent | Drive up, hips rise |
| Lockout | Knees and hips extended again |

## Bench press

| Phase | Description |
|---|---|
| Unrack | Bar lifted off rack |
| Descent | Bar comes down toward chest |
| Touch | Bar reaches chest |
| Ascent | Bar driven back up |
| Lockout | Arms extended at top |

## Bent-over row

| Phase | Description |
|---|---|
| Hinge | Athlete in hinged position, bar hanging |
| Pull | Bar pulled toward torso |
| Contraction | Bar contacts torso, brief pause |
| Eccentric | Bar lowered back down |
| Reset | Returns to hinged position |

## Golf full swing

| Phase | Description |
|---|---|
| Address | Stance set, club behind ball |
| Backswing | Club travels up and back |
| Top | Club at apex of backswing |
| Downswing | Club accelerates toward ball |
| Impact | Contact with ball (peak horizontal hand velocity) |
| Finish | Follow-through complete |

GolfDB ground-truth events (Address, Toe-Up, Mid-Backswing, Top,
Mid-Downswing, Impact, Mid-Follow-Through, Finish) are collapsed to these
six phases for consistency with the rest of the library; the GolfDB events
are still used to tune the segmentation thresholds.
