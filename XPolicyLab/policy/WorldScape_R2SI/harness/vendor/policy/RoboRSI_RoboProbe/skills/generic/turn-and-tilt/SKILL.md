# Turning and tilting

## When it applies
Whenever a held thing must be turned, tipped or tilted, including pouring loose contents from it.

## How
robo.turn rotates about a pivot in checked increments. To tip a held thing past horizontal, hold it with a level hand and roll: the last joint alone does it, while a hand from above soon runs out of range. wrist_pivot=True swings the hand around the wrist with the least arm motion.

To transfer loose contents by tilting, keep the opening free of the hand, remember it with robo.hold_point, and bring it to a centimetre or three above the target's rim before anything starts to move. What flows out leaves in the direction the container leans and lands beyond the point under the opening, further the faster it runs: start over the part of the target it leans away from, and move toward the middle as it steepens. Steepen in short pieces with slow=True and keep the opening over the target; what lands outside cannot be recovered. Then return the container.

## Checks
Before tilting anything held, check the grip again: a hold that survived a straight lift may still slip when turned. Angle and elapsed steps prove nothing: the images, or robo.watch on the target, show when the flow has stopped.

## Recovery
Rolling about the hand's own pointing axis uses only the last joint and is nearly always reachable; a rotation about a world axis usually moves the whole arm, and may not be reachable from where you are, in which case hold the object differently or reposition first.
