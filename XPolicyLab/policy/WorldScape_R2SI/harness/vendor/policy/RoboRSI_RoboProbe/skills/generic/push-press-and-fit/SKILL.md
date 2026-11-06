# Pushing, pressing and other contact

## When it applies
Whenever the robot pushes, slides, presses or strikes something, fits parts tightly, or moves a part along a mechanism.

## How
robo.stroke moves in small increments and stops where the arm is blocked. Contact shows up as the arm no longer following its commands.

To push: a closed gripper or the side of an open one a little above the surface, behind the object, moving along the line through its centre. A push off-centre turns it, and a push high on a tall thing tips it over. Go in short segments with a fresh measurement between them, and withdraw before you turn the wrist or go around. To slide something with control, press down on it or hold it with the fingers around it. What you hold follows the hand, so you can measure, correct while holding and stop when a correction would be coarser than what is left to correct. A tool's working end is what must make contact: remember it with robo.hold_point.

To press: from straight along the part's axis, go until it stops, then come back fully.

Tight fits: align position and axis first, at a visible distance, then go in with short steps and look again from a second direction.

Contact guided by a mechanism: work out what the guide leaves free, move only along that, in short pieces, and look after each. Act on the part itself, not on what it is mounted on.

## Checks
Blocked does not mean it worked: look. Before you let go, see from a second direction that what should support the thing really does. When the arm stops following, stop and look. A moving part has moved only when you see it reach the state asked for: drive it to the end of its travel and check again after letting go, once it has come to rest.

## Recovery
If a press does nothing, it did not go deep enough, or it missed; check the part in the image before pressing again. If it jams or tilts, back out to where it is visible and realign; do not push harder.
