# Turning and tilting

## When it applies
Whenever a held thing must be turned, tipped or tilted, including pouring loose contents from it.

## How
robo.turn rotates about a pivot in checked increments. To tip a held thing past horizontal, hold it with a level hand and roll: the last joint alone does it, while a hand from above soon runs out of range. wrist_pivot=True swings the hand around the wrist with the least arm motion.

To transfer loose contents by tilting, first check with robo.reach, before taking the container, that every tilt of the pour (about every 30 degrees to past horizontal) is reachable at the pouring spot; if not, change the grasp or the side. Keep the opening free of the hand, remember it with robo.hold_point, and bring it to a centimetre or three above the target's rim before anything starts to move. What flows out leaves in the direction the container leans and lands beyond the point under the opening, further the faster it runs: start over the part of the target it leans away from, and move toward the middle as it steepens. Steepen in short pieces with slow=True about the remembered opening and keep it over the target; what lands outside cannot be recovered. Unless the instruction asks for only part, pour until it is nearly empty: when the flow stops, steepen further, until the opening points clearly downward and steepening brings nothing more; a flow that stops at a shallow tilt has left contents inside. Then lift the opening a few centimetres clear above the target's rim and turn the container back upright about the grasp: turned back with its lip still in the mouth, it is driven into the rim and the hold shifts.

## Checks
Before tilting anything held, check the grip again: a hold that survived a straight lift may still slip when turned. Angle and elapsed steps prove nothing: the images, or robo.watch on the target, show when the flow has stopped.

## Recovery
Rolling about the hand's own pointing axis uses only the last joint and is nearly always reachable; a rotation about a world axis usually moves the whole arm, and may not be reachable from where you are, in which case hold the object differently or reposition first.
