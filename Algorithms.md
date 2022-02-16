# Algorithms

## Sorting algorithms overview

- **InsertionSort**: keep sorted part and not, take first element of unsorted and decide where to put it
- **SelectionSort**: find smallest element in unsorted and append it to sorted part
- **BubbleSort**: compare adjacent elements and swap them if they are not in order (stop if no swaps are made)
- **MergeSort**: divide array into two parts and sort them recursively
- **QuickSort**: pick a pivot element, put everything that's smaller than it to the left and everything that's bigger to the right. Iterate on left and right subarrays.
- **HeapSort**: heapify the array, then move the elements knowing which are leaves.
- **CountingSort**: create a new array with the same size as the original and fill it with zeros. For each element in the original array, add 1 to the corresponding index in the new array. Build array from counts.
- **RadixSort**: sort by digits, starting from least significant digit.
- **BucketSort**: sort by buckets, each bucket is a sorted array. Similar to countsort, but with buckets (works for floating point numbers).
- **BogoSort**: sort by swapping elements randomly until the array is sorted.
