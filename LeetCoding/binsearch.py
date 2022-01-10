def leftmost(a, query):
    l, r = 0, len(a) - 1
    while l < r:
        m = l + (r - l) // 2
        if query > a[m]:  # If the element is equal, we move the right index
            l = m + 1
        else:
            r = m
    return l


def rightmost(a, query):
    l, r = 0, len(a) - 1
    while l < r:
        m = l + (r - l) // 2
        if query < a[m]:  # If the element is equal, we move the left index
            r = m
        else:
            l = m + 1
    return r - 1


if __name__ == "__main__":
    print(leftmost([1, 2, 3, 3, 3, 3, 4, 5, 6], 3))
    print(rightmost([1, 2, 3, 3, 3, 3, 4, 5, 6], 3))
