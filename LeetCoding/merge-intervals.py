def merge_intervals(intervals):
    i = 0
    intervals.sort()
    while i < len(intervals) - 1:
        if intervals[i][1] >= intervals[i + 1][0]:
            print(intervals[i], intervals[i + 1])
            removed = intervals.pop(i + 1)
            intervals[i][1] = removed[1]
        else:
            i += 1
    return intervals


if __name__ == "__main__":
    print(merge_intervals([[1, 4], [4, 5]]))
