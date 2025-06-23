"""
This example is a simple example of how to use the LiveModelAnimation class to animate a model in real-time.
The user can interact with the model by changing the joint angles using sliders.
"""

from pyorerun import LiveModelAnimation

#model_path = "C:/Users/emmam/Documents/GIT/bioptim/bioptim/models/merge/female1_racine_main_armMerged.bioMod"
model_path = "./examples/getting_started/models/test_rotation.bioMod"
#model_path = "./examples/getting_started/models/triple_pendulum.bioMod"
animation = LiveModelAnimation(model_path, with_q_charts=True)
animation.rerun()
