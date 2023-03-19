
This repository implements an algorithm for the unidimensional transportation problem.

# 1D transportation problem

In the 1D transportation problem, we need to allocate supply from n sources to m sinks, with limits on supply and demand. The total supply may be smaller than the demand.

Each sink and source has a position (an integer). The cost of allocating one unit of supply is the distance between the source and the sink.

# Complexity

With n sources and m sinks, the traditional algorithm runs in time larger than O(n²m²), with some approximation algorithms running in time O(mn).
The algorithm presented here runs in only O(n log n + m log m).
