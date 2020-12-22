from numpy import array, tile, int32, sum, sqrt
# from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS

kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri', # 7
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank', # 13
                 'r_eye', 'l_eye', # 15
                 'r_ear', 'l_ear', 'com'] # 18

# min_profile = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11] # minimum requirement
right_profile = [1, 2, 3, 4, 8] # all required
left_profile = [1, 5, 6, 7, 11] # all required
prof_len = len(right_profile)

face_profile = [0, 16, 17] # one of these must be present: first in list has highest priority
# full_profile = [9, 12, 10, 13] # includes legs
transform_pairs = [[2, 5]]

""" Section: Proximal/Distal: Body Mass ratio: Proximal ratio
(1) HEAD-NECK-THORAX: Greater Trochanter/Glenohumeral Joint: 0.578M: 0.66
(2) UPPER ARM: Glenohumeral Joint/Elbow: 0.028M: 0.436
(3) FOREARM: Elbow/Ulnar Styloid: 0.022M: 0.430
(4) THIGH: Greater Trochanter/Femoral Condyles: 0.1M: 0.433
(5) FOOT&LEG: Femoral Condyles/Medial Malleolus 0.061M: 0.606 (0.394 Distal) (0.69 magic factor)
(S) TOTAL LEG: Greater Trochanter/Medial Malleolus: 0.161M: 0.447 (0.553 Distal) (2.07 magic factor)
"""
mass_ratios = array([[0.578, 0.028, 0.028, 0.022, 0.022, 0.1, 0.061, 0.161, 0.1, 0.061, 0.161]])
mass_ratios = tile(mass_ratios.transpose(), (1, 2))
proximal_ratios = [0.66, 0.436, 0.43, 0.433, 0.606, 0.69, 2.07] # final 2 are magic factors for missing points


def compute_com(kpt_ids, pose_keypoints):
    """Computes center of mass from available points for each pose.
    Requires at least one arm (shoulder, elbow, wrist), neck and hips. 
    Required keypoints to return result: at least one arm with hip, neck and [nose OR ear]

    :param kpt_id: IDs of keypoints in pose_keypoints. Corresponds to kpt_names.
    :param pose_keypoints: keypoints for parts of a pose. All types are in kpt_names.
    :return COM/BOS tuple: tuple of main center of mass' x,y coordinates (ndarray), segment COMs (ndarray),
        BOS coordinates (list of list of int)
    """
    C_pts = [] # minor center of mass points
    BOS = [[-1, -1], [-1, -1]] # base of support
    COM = array([-1, -1]).astype(int32) # final center of mass
    # legs are 3.5 to 4 heads
    # 25 and 20: 20 front, 5 back
    # Find length from nose/ears to neck and multiply 0.5 for front foot, 0.14 for back foot.
    
    ## Heuristics
    no_right = False
    no_left = False
    for r_id in right_profile:
        if r_id not in kpt_ids:
            no_right = True
            break
    for l_id in left_profile:
        if l_id not in kpt_ids:
            no_left = True
            break
    face_id = -1
    for f_id in face_profile:
        if f_id in kpt_ids:
            face_id = f_id
            break
    if face_id == -1:
        return (COM, array(C_pts), BOS)
    elif no_right and no_left:
        return (COM, array(C_pts), BOS)
    
    ## Transformation
    """Two scenarios
    (1) Front/Back of body: do nothing
    (2) Side of body: copy point to side if needed
    """
    if not no_right and no_left:
        for indx in range(prof_len):
            r_id = right_profile[indx]
            l_id = left_profile[indx]
            if pose_keypoints[l_id, 0] == -1:
                pose_keypoints[l_id] = pose_keypoints[r_id]
    elif no_right and not no_left:
        for indx in range(prof_len):
            r_id = right_profile[indx]
            l_id = left_profile[indx]
            if pose_keypoints[r_id, 0] == -1:
                pose_keypoints[r_id] = pose_keypoints[l_id]
    
    ## Compute COM sections
    face_pt = pose_keypoints[face_id]
    neck_pt = pose_keypoints[1]
    head_vector = (neck_pt - face_pt) # points down
    nose_neck_len = sqrt(sum(head_vector * head_vector))
    head_vector[0] = 0 # project to y-axis
    # head_vector[1] = head_vector[1] * 1.5
    
    r_sho_pt = pose_keypoints[2]
    l_sho_pt = pose_keypoints[5]
    upperRidge_pt = (r_sho_pt + l_sho_pt)/2
    
    r_hip_pt = pose_keypoints[8]
    l_hip_pt = pose_keypoints[11]
    lowerRidge_pt = (r_hip_pt + l_hip_pt)/2
    # Thorax COM
    thorax_vector = (lowerRidge_pt - upperRidge_pt) * proximal_ratios[0]
    C_pts.append((upperRidge_pt + thorax_vector).tolist())
    # Upper Arms COM
    r_elb_pt = pose_keypoints[3]
    l_elb_pt = pose_keypoints[6]
    r_uparm_vector = (r_sho_pt - r_elb_pt) * proximal_ratios[1]
    l_uparm_vector = (l_sho_pt - l_elb_pt) * proximal_ratios[1]
    C_pts.append((r_uparm_vector + r_elb_pt).tolist())
    C_pts.append((l_uparm_vector + l_elb_pt).tolist())
    # Forearms COM
    r_forarm_vector = (r_elb_pt - pose_keypoints[4]) * proximal_ratios[2]
    l_forarm_vector = (l_elb_pt - pose_keypoints[7]) * proximal_ratios[2]
    C_pts.append((r_forarm_vector + pose_keypoints[4]).tolist())
    C_pts.append((l_forarm_vector + pose_keypoints[7]).tolist())
    # Thigh COM and Leg COM (OR) Total Leg COM (if pts missing)
    # Right Side
    if pose_keypoints[9,0] == -1: # missing leg estimation
        r_total_leg_com = (head_vector * proximal_ratios[6]) + r_hip_pt
        C_pts.append([0,0])
        C_pts.append([0,0])
        C_pts.append(r_total_leg_com.tolist())
        BOS[0] = ((head_vector * 3.5) + r_hip_pt).tolist()
    else:
        r_knee_pt = pose_keypoints[9]
        r_thigh_vector = (r_hip_pt - r_knee_pt) * proximal_ratios[3]
        C_pts.append((r_thigh_vector + r_knee_pt).tolist())
        if pose_keypoints[10, 0] == -1: # missing ankle estimation
            r_leg_com = (head_vector * proximal_ratios[5]) + r_knee_pt
            C_pts.append(r_leg_com.tolist())
            BOS[0] = ((head_vector * 1.75) + r_knee_pt).tolist()
        else:
            r_ankle_pt = pose_keypoints[10]
            r_leg_vector = (r_knee_pt - r_ankle_pt) * proximal_ratios[4]
            C_pts.append((r_leg_vector + r_ankle_pt).tolist())
            BOS[0] = r_ankle_pt.tolist()
        C_pts.append([0,0])
    # Left Side
    if pose_keypoints[12,0] == -1: # missing leg estimation
        l_total_leg_com = (head_vector * proximal_ratios[6]) + l_hip_pt
        C_pts.append([0,0])
        C_pts.append([0,0])
        C_pts.append(l_total_leg_com.tolist())
        BOS[1] = ((head_vector * 3.5) + l_hip_pt).tolist()
    else:
        l_knee_pt = pose_keypoints[12]
        l_thigh_vector = (l_hip_pt - l_knee_pt) * proximal_ratios[3]
        C_pts.append((l_thigh_vector + l_knee_pt).tolist())
        if pose_keypoints[13, 0] == -1: # missing ankle estimation
            l_leg_com = (head_vector * proximal_ratios[5]) + l_knee_pt
            C_pts.append(l_leg_com.tolist())
            BOS[1] = ((head_vector * 1.75) + l_knee_pt).tolist()
        else:
            l_ankle_pt = pose_keypoints[13]
            l_leg_vector = (l_knee_pt - l_ankle_pt) * proximal_ratios[4]
            C_pts.append((l_leg_vector + l_ankle_pt).tolist())
            BOS[1] = l_ankle_pt.tolist()
        C_pts.append([0,0])
    
    ## Compute COM from C_pts, and BOS
    C_pts = array(C_pts, dtype=int32)
    COM = sum(C_pts * mass_ratios, axis=0).astype(int32)
    
    # was BOS[0][0] == BOS[1][0]
    if no_left^no_right: # sagittal spreading; greedy approach
        min1, min2, min3, min4 = [-1, -1, -1, -1]
        if no_left: # facing towards right of image
            min1 = round(BOS[0][0] - (nose_neck_len * 0.14)) # constants 0.14 and 0.5 based on my estimates
            min2 = round(BOS[1][0] - (nose_neck_len * 0.14)) # of nose-neck length and foot length relative
            max1 = round(BOS[0][0] + (nose_neck_len * 0.5))  # to ankle point.
            max2 = round(BOS[1][0] + (nose_neck_len * 0.5))
        else: # facing towards left of image
            min1 = round(BOS[0][0] - (nose_neck_len * 0.5))
            min2 = round(BOS[1][0] - (nose_neck_len * 0.5))
            max1 = round(BOS[0][0] + (nose_neck_len * 0.14))
            max2 = round(BOS[1][0] + (nose_neck_len * 0.14))
        if min1 < min2:
            BOS[0][0] = min1
        else:
            BOS[0][0] = min2
        if max1 > max2:
            BOS[1][0] = max1
        else:
            BOS[1][0] = max2
    
    return (COM, C_pts, BOS)

