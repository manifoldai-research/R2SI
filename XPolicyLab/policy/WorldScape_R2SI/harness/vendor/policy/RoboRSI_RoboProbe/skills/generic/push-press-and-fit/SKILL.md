# Pushing, pressing and other contact

## When it applies
Whenever the robot pushes, slides, presses or strikes something, fits parts tightly, or moves a part along a mechanism.

## How
robo.stroke moves in small increments and stops where the arm is blocked. Contact shows up as the arm no longer following its commands.

To push: a closed gripper or the side of an open one a little above the surface, behind the object, moving along the line through its centre. A push off-centre turns it, and a push high on a tall thing tips it over. Go in short segments with a fresh measurement between them, and withdraw before you turn the wrist or go around. Stop correcting once the error left is no larger than one push reliably moves, or once a push overshot by about what it fixed, and keep the steps to return; between pushes keep the hand closed and near. To slide something with control, press down on it or hold it with the fingers around it. For an exact final pose of a flat piece the jaw can span, hold it at the surface and slide or turn it without lifting; pushes overshoot by about what they fix. What you hold follows the hand, so you can measure, correct while holding and stop when a correction would be coarser than what is left to correct. A tool's working end is what must make contact: remember it with robo.hold_point. Choose the grasp on a tool from the pose its working end needs at contact, checked with robo.reach before you take it; hold still whatever you push into. With a held tool only the working end may touch; keep the shaft over free space. To push loose pieces into a receiver, keep it still with its lip flat on the surface and its opening toward them, and push each piece fully past the lip; to see what is in, look rather than lift or tip the loaded receiver.

To press: from straight along the part's axis, go until it stops, then come back fully. To strike with a held tool, end the stroke two or three centimetres beyond the surface and withdraw at once: the tool gives in the grip, while a stroke that ends at the surface may never touch it.

Tight fits: align position and axis first, at a visible distance, then go in with short steps and look again from a second direction. To thread an opening onto a peg, post or hook, take its axis from two measured points along it, as it may slope, mark the held opening in two cameras with robo.hold_point, and move it along that axis. A screw or nut goes only a few millimetres along its thread per full turn: turn it through several full turns, regrasping when the wrist runs out, and measure its height after each turn until it no longer goes down.

Contact guided by a mechanism: work out what the guide leaves free, move only along that, in short pieces, and look after each. Act on the part itself, not on what it is mounted on.

## Checks
Blocked does not mean it worked: look. Before you let go, see from a second direction that what should support the thing really does. When the arm stops following, stop and look. A moving part has moved only when you see it reach the state asked for: drive it to the end of its travel and check again after letting go, once it has come to rest.

## Recovery
If a press does nothing, it did not go deep enough, or it missed; check the part in the image before pressing again. If it jams or tilts, back out to where it is visible and realign; do not push harder. Do not let go of a part that has not gone in: back off slightly, shift it a millimetre at a time around the opening, and press again.
