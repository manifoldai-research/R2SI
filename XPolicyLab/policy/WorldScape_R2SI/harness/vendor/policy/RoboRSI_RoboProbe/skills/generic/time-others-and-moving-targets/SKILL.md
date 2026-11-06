# Time, memory and others

## When it applies
Whenever something must be waited for, others act in the workspace, something seen now will be needed later, or a target moves.

## How
Time passes only in native steps, and anything else in the scene moves only while they run. When you must wait, robo.wait or robo.watch; wait in short pieces when an event is near.

Others may act in the workspace. Give them room: the place you know is clear is where your arms started, so go back there, keep still while they act, and go on once they have finished and withdrawn.

Moving targets: two robo.locate calls with a robo.wait between them tell whether a thing moves and how fast. robo.pick with velocity meets a thing where it will be; a jaw closing along its direction of travel forgives being early or late. robo.estimate gives the steps of any other motion, during which it moves on. It moves only during native steps, so a cell can do all this without a model call in between.

## Checks
Note in variables what you see happen and in what order, and anything you will need later while you can still see it: of the images, only the first views and the latest stay in the conversation, and robo.show looks again at what is there now. Things out of view stay where you last saw them unless something moved them.

## Recovery
If you miss, measure again before trying again.
