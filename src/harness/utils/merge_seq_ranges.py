def merge_seq_ranges(ranges: list[tuple[int, int]]):
    merged: list[tuple[int, int]] = []
    
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][-1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (
                previous_start,
                max(end, previous_end)
            )
    return merged