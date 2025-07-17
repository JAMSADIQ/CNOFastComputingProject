import numpy as np


def boxes_intersect(box1, box2):
    """Check if two 3D boxes intersect."""
    for (min1, max1), (min2, max2) in zip(box1, box2):
        if max1 < min2 or max2 < min1:
            return False  # No overlap in this dimension
    return True  # Overlaps in all 3 dimensions


def check_intersections(obstacles_, ac_):
    for ac_name, ac_box in ac_.items():
        for obs_name, obs_box in obstacles_.items():
            if boxes_intersect(ac_box, obs_box):
                print(f"⚠️  AC '{ac_name}' intersects with obstacle '{obs_name}'")
            else:
                print(f"✅ AC '{ac_name}' does NOT intersect obstacle '{obs_name}'")



def compute_overlap_volume(box1, box2):
    """Compute overlapping volume between two boxes. Returns 0 if no overlap."""
    overlap_dims = []
    for (min1, max1), (min2, max2) in zip(box1, box2):
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        if overlap_max <= overlap_min:
            return 0.0  # No overlap
        overlap_dims.append(overlap_max - overlap_min)
    return overlap_dims[0] * overlap_dims[1] * overlap_dims[2]


def compute_box_volume(box):
    """Compute volume of a single 3D bounding box."""
    return np.prod([maxv - minv for (minv, maxv) in box])


def check_intersections_with_overlap(obstacles_, ac_):
    results = []
    for ac_name, ac_box in ac_.items():
        ac_volume = compute_box_volume(ac_box)
        for obs_name, obs_box in obstacles_.items():
            overlap_volume = compute_overlap_volume(ac_box, obs_box)
            overlap_percent = (overlap_volume / ac_volume * 100) if ac_volume > 0 else 0
            if overlap_volume > 0:
                print(f"⚠️ AC '{ac_name}' intersects with obstacle '{obs_name}'")
                print(f"   Overlap Volume: {overlap_volume:.3f} (of aircraft volume {ac_volume:.3f})")
                print(f"   → Overlap = {overlap_percent:.2f}% of aircraft")
            else:
                print(f"✅ AC '{ac_name}' does NOT intersect obstacle '{obs_name}'")
            results.append((ac_name, obs_name, overlap_volume, overlap_percent))
    return results


#sim2
#obstacles_ = {
#    'obs0': [(5.75, 9.5), (1.04, 3.37), (6.35, 9.29)],
#}
#
#ac_ = {
#    'ac0': [(8.01, 8.65), (1.59, 2.52), (6.69, 8.18)],
#}
#sim18
#obstacles_ = {
#    'obs0': [(2.59, 8.13), (2.94, 8.42), (5.16, 9.5)],
#    'obs1': [(1.8, 3.6), (2.69, 7.45), (5.93, 9.5)],
#    'obs2': [(5.31, 8.63), (4.18, 5.84), (8.36, 9.26)],
#}

#ac_ = {
#    'ac0': [(4.91, 6.51), (2.74, 4.28), (5.05, 6.23)],
#}



print(check_intersections(obstacles_, ac_))
print(check_intersections_with_overlap(obstacles_, ac_))
