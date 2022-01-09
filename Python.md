# Python

Here you can find some implementation details for Python, which usually turn out to be useful.

## Data structures

### Lists

- Lists are dynamically resized: a `tuple` differs from a list from the fact that it is **immutable**
- You can `append(el)`, `remove(value)`, `insert(index, value)`
- `A in array` checks for presence of `A` with complexity `O(n)`
- `B=A` copies the reference, `B=list(A)` copies the array, `copy.deepcopy(A)` does a deep copy
- `min(A)` and `max(A)` are nice to use
- For binary search, you can use `bisect.bisect(array, element)` which returns the `i` for which we can split the array and have everything that's less than `element` to the left
- You can use slicing to rotate a list: `A[k:]+A[k:]`
- Slicing can shallow-copy too: `B =A[:]`
- Revert a list with a negative stepsize `A[::-1]`
- List comprehension is multi-level too: `[(x,y) for x in A for y in B]` works

### Binary search

- In a sorted array, use `bisect.bisect_left` to get the first occurrence, `bisect.bisect_right` for the last+1

### Dictionaries

- Remove elements using `del dictionary[key]`, or `popitem()` if you wanna pop a random one
- It works like [this](https://mail.python.org/pipermail/python-list/2000-March/048085.html):
  - The hashtable is a contiguous vector of records, whose slots are of 3 types: key+value pairs, virgins, turds
  - The table is doubled when the virigns are below 1/3 of the slots
  - Note that when you `del` an element, the table isn't shrinked (as Python assumes you'll fill it with something else)
  - `dict=dict.copy()` gets rid of the turds
  - The hashing allows us to compute the **table address** from the **key**
  - The hash function should be **uniform**, at least for the table addresses (as to avoid collisions), and usually gets modulo'ed to get the final table address
  - Knowing the set `S` of keys, it is teoretically possible to construct a **perfect hash function**
  - As eliminating collisions is technically impossible, some techniques are used: _separated chainring_ consists in building a dictionary having as values nodes of a linked list that will then be linearly searched (so, if two elements point to the same hash, they will be consequent in the LL), while in _open addressing_ the elements are inserted into the bucket, but they will just be inserted in the first empty address
  - Python dictionaries use open hashing based on a primitive polynomial over Z/2
- Use `collections.defaultdict` to avoid incurring in `KeyError`s when retrieving new keys
- Time-complexity is `O(1)` for access and addition

### LinkedLists

- We can consider _singly_ and _doubly_ LinkedLists: the latter have a `node.previous` attribute too
- Nice because the elements do not have to be saved contiguously in the memory
- Not allowing random access, many operations are slower
- _Circularly linked_ lists are interesting to represent arrays that are natural linked (think about the vertices of a polygon)
- Often, using **two pointers** to traverse a LinkedList can be useful
- To find loops in a LL, use two pointers, one traversing `pointer = pointer.next.next`: if they meet, there's a loop

### Stacks/Queues

- Both stacks and queues are implemented using LinkedLists, and just restrict the type of operations

- You can use `collections.deque` for a list with `O(1)` append and retrieval from both sides
- To create a **queue using two stacks**, you can keep a a tail and head, then fill the head inverting the tail **when it is empty**

### Other syntactic sugar

You can use `dataclasses.dataclass` to create what is basically a struct:

```python
from dataclasses import dataclass
@dataclass
class Rectangle
   # included in sort
    left :int
    bottom :int
    right :int
    # NOT included in sort.
    top :int = field(compare = false)
    # __init__() is autocreated

# Use *args to create
coords = [1,2,3,4]
rec = Rectangle(*coords)
```

You can use `collections.namedtuple` to have a **named tuple**:

```python
from collections import namedtuple
Entry = namedtuple("Entry",['time', 'value'])
e = Entry(time = 1, value = 2)
e[0] == e.time #True
```

## Itertools

Itertools is an awesome library that deals with iterators for efficient looping.

## Decorators

WIP

## Testing

- `assert` is the simplest operation you can do to test values, which fails with an `AssertionError` in case of false

- `unittest` has been the standard library for unit tests since 2.1

  - You put your tests into classes as methods

  - Then use a series of special assertion methods in the `unittest.TestCase.class` instead of the built-in `assert`

  - ```python
    import unittest
    class TestSum(unittest.TestCase):
        def test_sum(self):
            self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
        def test_sum_tuple(self):
            self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

    if __name__ == '__main__':
        unittest.main()
    ```

  - You can run tests with `python -m unittest test` instead of that `unittest.main()`

- Please put your tests into a separate file!
