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

#sim 21
#obstacles_ = {
#    'obs0': [(8.97, 9.5), (4.09, 8.49), (2.87, 4.24)],
#    'obs1': [(7.28, 9.5), (2.57, 4.84), (6.42, 9.27)],
#    'obs2': [(6.49, 9.5), (7.11, 9.5), (3.17, 4.06)],
#}
#ac_ = {
#    'ac0': [(8.03, 9.5), (5.96, 6.78), (6.6, 7.7)],
#}

#sim 32
#obstacles_ = {
#    'obs0': [(8.63, 9.37), (3.57, 6.13), (1.13, 3.28)],
#}
#ac_ = {
#    'ac0': [(5.87, 7.78), (8.86, 9.5), (8.22, 9.5)],
#}

#sim 34
#obstacles_ = {
#    'obs0': [(8.89, 9.5), (1.61, 2.31), (6.6, 9.5)],
#    'obs1': [(5.41, 9.5), (1.51, 5.1), (1.8, 7.79)],
#    'obs2': [(3.18, 7.48), (3.04, 3.76), (6.21, 9.5)],
#}
#
#ac_ = {
#    'ac0': [(8.88, 9.5), (4.52, 5.5), (5.1, 6.03)],
#}

#sim 38
#obstacles_ = {
#    'obs0': [(8.29, 9.12), (8.08, 9.5), (1.29, 6.34)],
#    'obs1': [(7.32, 9.5), (5.9, 9.5), (3.8, 7.4)],
#}
#
#ac_ = {
#    'ac0': [(7.96, 9.44), (3.9, 5.65), (5.62, 7.04)],
#}

#sim 39
#obstacles_ = {
#    'obs0': [(5.65, 7.0), (0.53, 1.13), (6.49, 8.19)],
#}
#
#ac_ = {
#    'ac0': [(8.89, 9.5), (3.37, 4.69), (2.17, 3.02)],
#}
#
##sim 41
#obstacles_ = {
#    'obs0': [(6.08, 9.5), (4.66, 5.23), (3.93, 5.75)],
#}
#
#ac_ = {
#    'ac0': [(2.43, 2.95), (4.91, 5.94), (7.73, 8.78)],
#}
#
#
##sim 48
#obstacles_ = {
#    'obs0': [(5.64, 6.28), (6.9, 9.5), (8.94, 9.5)],
#}
#
#ac_ = {
#    'ac0': [(7.59, 8.59), (3.98, 5.24), (6.41, 8.35)],
#}

#sim 8 
#obstacles_ = {
#    'obs0': [(1.38, 2.04), (8.97, 9.5), (7.61, 8.16)],
#}
#
#ac_ = {
#    'ac0': [(7.06, 7.98), (0.51, 1.91), (9.37, 9.5)],
#}
#
#sim 27
obstacles_ = {
    'obs0': [(1.3, 5.86), (1.04, 6.18), (5.74, 9.5)],
}

ac_ = {
    'ac0': [(1.9, 2.66), (1.91, 2.97), (6.34, 6.9)],
}


#sim 31
#obstacles_ = {
#    'obs0': [(8.24, 9.5), (1.46, 7.11), (4.08, 9.5)],
#    'obs1': [(4.01, 7.96), (1.89, 4.44), (3.09, 7.37)],
#}
#
#ac_ = {
#    'ac0': [(6.17, 6.88), (2.23, 3.46), (7.91, 9.5)],
#}
#sim  33
#obstacles_ = {
#    'obs0': [(8.55, 9.16), (4.01, 7.33), (2.3, 7.81)],
#}
#
#ac_ = {
#    'ac0': [(0.59, 2.57), (6.98, 8.32), (9.39, 9.5)],
#}

#sim 40
#obstacles_ = {
#    'obs0': [(9.29, 9.5), (3.93, 9.01), (7.94, 9.5)],
#    'obs1': [(0.92, 2.73), (5.99, 9.5), (2.51, 5.27)],
#}
#
#ac_ = {
#    'ac0': [(4.37, 4.91), (4.31, 5.07), (9.32, 9.5)],
#}

#sim 49
#obstacles_ = {
#    'obs0': [(8.89, 9.5), (8.04, 9.5), (2.28, 3.45)],
#    'obs1': [(5.98, 9.5), (0.94, 5.3), (8.15, 9.5)],
#    'obs2': [(7.62, 9.35), (0.59, 5.08), (7.49, 9.5)],
#}
#
#ac_ = {
#    'ac0': [(3.95, 5.69), (2.93, 4.65), (8.48, 8.99)],
#}
#
##sim 1 
#obstacles_ = {
#    'obs0': [(5.11, 6.1), (6.97, 9.5), (3.45, 5.55)],
#    'obs1': [(0.8, 3.92), (5.56, 7.58), (5.62, 8.08)],
#    'obs2': [(7.02, 9.5), (6.75, 9.5), (5.91, 9.47)],
#}
#
#ac_ = {
#    'ac0': [(6.06, 7.53), (5.59, 6.27), (1.17, 2.37)],
#}
#
##sim 5
obstacles_ = {
    'obs0': [(1.23, 6.69), (3.81, 7.64), (5.41, 9.5)],
}

ac_ = {
    'ac0': [(4.05, 5.36), (9.5, 9.5), (6.25, 7.2)],
}

print(check_intersections(obstacles_, ac_))
print(check_intersections_with_overlap(obstacles_, ac_))
