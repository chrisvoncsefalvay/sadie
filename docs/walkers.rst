=======
Walkers
=======

Walkers are spatial objects (objects with spatial coordinates) that perform and track movements in space. They inherit
from the :class:`sadie.agents.BaseWalker` class.

Simple walkers
--------------

Simple walkers exhibit a uniform angular distribution and a uniform distance distribution:

* :class:`sadie.agents.walkers.BaseWalker` is a simple agent with a set maximum distance and uniform distance and angular distribution.
* :class:`sadie.agents.walkers.WaitingUniformWalker` is a simple agent that uses a transition probability property to determine whether to move or wait at a target.


Lévy walkers
------------

Lévy walkers implement some flavour of a Lévy flight, i.e. a movement pattern with a uniform angular distribution, but with distance being given by the Lévy distribution

.. math::

    f(x) =  \frac{e^{- \frac{1}{2x}}}{\sqrt{2 \pi x^3}}



The uniform Lévy walker
.......................

The uniform Lévy walker (:class:`sadie.agents.walkers.UniformLevyRandomWalker`) is a pure case of the Lévy walk, i.e.
unbounded Lévy flight with a uniform angular distribution.

The bounded uniform Lévy walker
...............................

The bounded uniform Lévy walker (:class:`sadie.agents.walkers.BoundedUniformLevyWalker`) takes a continuous
distribution (subclassed from Scipy's `rv_continuous`) as one of its parameters, and an arbitrary number of keyword
arguments. These are passed to the constructor of the random distribution. The distribution will then be used to
determine whether to retarget the Walker, by comparing it to the current trip's length.

For instance,

.. code-block:: python

    from scipy import stats

    w = BoundedUniformLevyWalker(bounding_distribution = stats.beta, a = 3, b = 5)

creates a walker that uses a beta distribution with :math:`\\alpha = 3` and :math:`\\beta = 5` to determine whether to
retarget the walker. Since the beta distribution is by definition in the domain :math:`[0, 1]`, it is useful to
introduce a scale factor:

.. code-block:: python

    from scipy import stats

    w = BoundedUniformLevyWalker(bounding_distribution = stats.beta, scale_factor = 100, a = 3, b = 5)

This multiplies resulting values from the random distribution by the scale factor before the comparison to trip length.


Homesick Lévy walkers
.....................

Homesick Lévy flights were introduced in a 2014 paper by `Fujihara and Miwa <https://arxiv.org/abs/1408.0427>`_ and
described as having ichi-go ichi-e characteristics, i.e. most agents in probabilistic space only cross paths once per
period of simulation. Sadie currently implements two subtly different versions of the homesick Lévy walk:

* Homesick Lévy walk (:class:`sadie.agents.walkers.HomesickLevyWalker`) is a non-waiting Lévy walker that exhibits 'homesick' behaviour around its point of origin, i.e. it assumes the point of origin as its target with the probability :math:`\alpha` over stops, i.e. approximately once in :math:`1^{-\alpha}` trips.
* Rapid homesick Lévy walk (:class:`sadie.agents.walkers.RapidHomesickLevyWalker`) is a non-waiting Lévy walker that exhibits 'constant homesick' behaviour around its point of origin, i.e. it assumes the point of origin as its target with the probability :math:`\alpha` over distance, i.e. the likelihood of retargeting for the home location at any given step is :math:`\alpha`. Consequently, a much lower :math:`\alpha` is used for rapid homesick Lévy walks.
