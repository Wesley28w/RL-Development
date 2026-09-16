# notebook for tracking changes made
- Decieded to switch to 12 seed tests
- Fixed a velocity and write error in the Franka Cabinet Code
- Added gating to lift and franka cabinet so they can't sample empty buffer indices
- Realized curriculum environments can add to buffer, in which they do so immediately with all sub-tasks they completed. 

* may want to add: reset to next state instead of just the previous, that way it doesn't get stuck (seems the gripper doesn't know to grasp the handle and reward hacks) Probably best not to fully change, as it complicates things, but we could add in a small section of the environments that do this in order to resist this issue.

- Realized Factory is failing one seed and the 16 run made results sucky. I need to figure out the issue..
- Realized lift sub-task 2 was only being called, meaning something is broken. Fixed, but untested. So need to rerun
