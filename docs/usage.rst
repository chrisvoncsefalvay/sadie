=====
Usage
=====

There are multiple ways to use Sadie.


Using pre-defined objects
-------------------------

Sadie comes fully loaded with a range of objects that can provide for a wide range of use cases. These can be directly
instantiated and used.

* `Agent`s are basic elements of agent-based models.
* `Walker`s are agents that implement various random walk patterns, including having their own state and conditionality.
* `Forager`s are special walkers that alter their behaviour as a function of their environment.

Creating your own objects
-------------------------

Changing azimuth and distance distributions
...........................................

To create your own objects, subclass the closest base class and add your own desired functionality. Consider, for
instance, the way the Uniform-Lévy walker is constructed:

.. code-block:: python

    class UniformLevyRandomWalker(BaseWalker):
        def retarget(self):
            self.set_polar_target(np.random.uniform(0, 2*π), stats.levy().rvs())


Simply overriding the `retarget()` method modifies the targeting behaviour of a walker -- you can, for instance, pick a
different distribution or set parameters to the distribution.

Changing wait/move behaviour
............................

To change wait/move behaviour, e.g. to halt the object, consider overriding its `update()` method:

.. code-block:: python

    class WaitingUniformWalker(BaseWalker):
        def __init__(self, x_init: float, y_init: float, velocity: float = 1.0, wait_transition_probability: float = 0.4):
            super(WaitingUniformWalker, self).__init__(x_init=x_init, y_init=y_init, velocity=velocity)
            self.wait_transition_probability = wait_transition_probability

        def update(self):
            if self.target == (None, None):
                self.retarget()
            elif self.is_on_target:
                if np.random.random() > self.wait_transition_probability:
                    self.retarget()
                else:
                    self.wait()
            else:
                self.move()

Here, we override the `__init__()` method first, adding a new parameter to it to represent the wait transition
probability. We assign this to a field, then, overriding the `update()` method, we provide a checkpoint: once an agent
is on target and due for re-targeting, we insert a random value condition whereby a random float must exceed the
pre-defined `wait_transition_probability`.
