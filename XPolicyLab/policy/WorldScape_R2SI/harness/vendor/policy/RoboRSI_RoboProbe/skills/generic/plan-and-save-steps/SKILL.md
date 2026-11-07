# Planning and saving steps

## When it applies
Before the first motion, and whenever you choose the next motions.

## How
Native steps are what runs out. Plan the whole job before the first motion, from the end backwards: for each object, which arm reaches both where it is and where it goes, with which hand orientation, and leaving what space for the fingers. See what stands in the way of the end state and deal with it first. Choosing the arm by the destination avoids hand-overs, which are expensive. Compare estimated motion costs and avoid unnecessary moves. Do not infer a total allowance from the instruction or assume how many steps remain.

Steps go to these things, in order of waste: repeating failed grasps, passing objects between hands without need, returning to the initial pose in mid-task, and moving axis by axis. To see past an arm, lift it or move it aside just enough: sending it home and back costs twice home_steps. Go from A to B in one motion with position and orientation together. Two independent goals, one per arm, belong in one robo.both, which saves most of the steps of doing them in turn. Do the far and bulky moves first, and do not disturb what is already right. Order the moves so that none covers or blocks what a later one must reach or see, and lift only as high as the next move needs. A hand carrying something far out should first retract a little, then lift.

## Checks
Check the plan with robo.reach and robo.estimate.

## Recovery
Track the actions spent on each item; if attempts keep failing, change the approach or safely work on another item before revisiting it. Set an item aside only after two different approaches failed, and come back to set-aside items cheapest first; finish small remaining corrections before moving on, and leave long recoveries of fallen things for last.
