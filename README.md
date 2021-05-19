# Retroactive Priority Queues

This repository includes an implementation of [partially retroactive priority queues](https://erikdemaine.org/papers/Retroactive_TALG/), with an experimental implementation of fully retroactive priority queues (using [hierarchical checkpointing](https://erikdemaine.org/papers/FullyRetroactive_WADS2015/)). It is similar in interface [to another 6.851 project](https://github.com/6851-2021/retroactive-priority-queue) but is internally different. 

## Data structures
See PDF writeup. We implement [weight-balanced B-trees](https://erikdemaine.org/papers/CacheObliviousBTrees_SICOMP/paper.pdf), [scapegoat trees](https://opendatastructures.org/newhtml/ods/latex/scapegoat.html), and a number of tree augmentations (sum, prefix sum, arbitrary marking).

## Using
This project has no dependencies.  The most useful class in this project is `PRPriorityQueue`, which supports `insert()`, `delete_min()`, and `delete_op()` operations.
```python
>>> from retroactive_pq.partial_pq import PRPriorityQueue
>>> q = PRPriorityQueue()
>>> q.insert(1)
>>> q.insert(2)
>>> q
Qnow: 1 2
events:
1.0: insert 1
2.0: insert 2

>>> q.insert(3, t=2.5)
>>> q.delete_min(t=2.25)
>>> q
Qnow: 2 3
events:
1.0: insert 1
2.0: insert 2
2.25: delete min
2.5: insert 3

>>> q.delete_op(2.25)
>>> q
Qnow: 1 2 3
events:
1.0: insert 1
2.0: insert 2
2.5: insert 3
```

## Running tests
We use `pytest` for testing. The tests for the weight-balanced B-tree, scapegoat tree, and prefix sum tree implementations are automatically generated; they check invariants at each step and are therefore computationally expensive (but extremely useful when debugging). To speed the tests up, run the tests with multiple cores [using `pytest-xdist`](https://pypi.org/project/pytest-xdist/).

## Future work
See PDF writeup.
