# Modified from GVHMR [https://github.com/zju3dv/GVHMR].
# Load EMDB data

import pickle
import torch

EMDB1_LIST = [
    'P8/69_outdoor_cartwheel/P8_69_outdoor_cartwheel_data.pkl', # 656
    'P5/42_indoor_dancing/P5_42_indoor_dancing_data.pkl', # 1291
    'P6/51_outdoor_dancing/P6_51_outdoor_dancing_data.pkl',  # 1427
    'P2/23_outdoor_hug_tree/P2_23_outdoor_hug_tree_data.pkl',  # 1086
    'P6/49_outdoor_big_stairs_down/P6_49_outdoor_big_stairs_down_data.pkl',  # DUPLICATE 1559

    'P7/59_outdoor_rom/P7_59_outdoor_rom_data.pkl', # 1839
    'P3/31_outdoor_workout/P3_31_outdoor_workout_data.pkl',  # 1216
    'P3/33_outdoor_soccer_warmup_b/P3_33_outdoor_soccer_warmup_b_data.pkl',  # 1433
    'P7/57_outdoor_rock_chair/P7_57_outdoor_rock_chair_data.pkl', # DUPLICATE 1558

    'P3/32_outdoor_soccer_warmup_a/P3_32_outdoor_soccer_warmup_a_data.pkl',  # 1084
    'P8/64_outdoor_skateboard/P8_64_outdoor_skateboard_data.pkl',   # DUPLICATE 1704
    'P7/60_outdoor_workout/P7_60_outdoor_workout_data.pkl',  # 1693
    'P6/50_outdoor_workout/P6_50_outdoor_workout_data.pkl', # 1532

    'P8/68_outdoor_handstand/P8_68_outdoor_handstand_data.pkl',  # 1606
    'P9/76_outdoor_sitting/P9_76_outdoor_sitting_data.pkl',  # 1768
    'P1/14_outdoor_climb/P1_14_outdoor_climb_data.pkl',  # 1284
    'P5/44_indoor_rom/P5_44_indoor_rom_data.pkl', # 1381
]
EMDB1_NAMES = ["_".join(p.split("/")[:2]) for p in EMDB1_LIST]

EMDB2_LIST = [
    'P2/19_indoor_walk_off_mvs/P2_19_indoor_walk_off_mvs_data.pkl',  # 1299
    'P3/29_outdoor_stairs_up/P3_29_outdoor_stairs_up_data.pkl', # 1205
    'P4/35_indoor_walk/P4_35_indoor_walk_data.pkl',  # 1226
    'P7/55_outdoor_walk/P7_55_outdoor_walk_data.pkl',  # 2179
    'P9/80_outdoor_walk_big_circle/P9_80_outdoor_walk_big_circle_data.pkl',  # 2240
    'P9/77_outdoor_stairs_up/P9_77_outdoor_stairs_up_data.pkl',  # DUPLICATE 728
    'P9/79_outdoor_walk_rectangle/P9_79_outdoor_walk_rectangle_data.pkl',  # 1917

    'P7/57_outdoor_rock_chair/P7_57_outdoor_rock_chair_data.pkl',  # DUPLICATE 1558
    'P2/24_outdoor_long_walk/P2_24_outdoor_long_walk_data.pkl',  # 3280
    'P3/30_outdoor_stairs_down/P3_30_outdoor_stairs_down_data.pkl',  #1137
    'P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl',  # 2160
    'P6/49_outdoor_big_stairs_down/P6_49_outdoor_big_stairs_down_data.pkl',   # DUPLICATE 1559
    'P9/78_outdoor_stairs_up_down/P9_78_outdoor_stairs_up_down_data.pkl',  # 1083

    'P7/56_outdoor_stairs_up_down/P7_56_outdoor_stairs_up_down_data.pkl', # 1120
    'P2/20_outdoor_walk/P2_20_outdoor_walk_data.pkl',  # 2713
    'P3/27_indoor_walk_off_mvs/P3_27_indoor_walk_off_mvs_data.pkl',  # 1448
    'P4/37_outdoor_run_circle/P4_37_outdoor_run_circle_data.pkl', # 881
    'P5/40_indoor_walk_big_circle/P5_40_indoor_walk_big_circle_data.pkl', # 2661
    'P6/48_outdoor_walk_downhill/P6_48_outdoor_walk_downhill_data.pkl',  # 1959

    'P0/09_outdoor_walk/P0_09_outdoor_walk_data.pkl',  # 2009
    'P3/28_outdoor_walk_lunges/P3_28_outdoor_walk_lunges_data.pkl',  # 1836
    'P7/58_outdoor_parcours/P7_58_outdoor_parcours_data.pkl',  # 1332
    'P7/61_outdoor_sit_lie_walk/P7_61_outdoor_sit_lie_walk_data.pkl', # 1914
    'P8/64_outdoor_skateboard/P8_64_outdoor_skateboard_data.pkl',   # DUPLICATE 1704
    'P8/65_outdoor_walk_straight/P8_65_outdoor_walk_straight_data.pkl',  # 1981
]

EMDB2_NAMES = ["_".join(p.split("/")[:2]) for p in EMDB2_LIST]
EMDB_NAMES = {1: EMDB1_NAMES, 2: EMDB2_NAMES}
EMDB_LIST = {1: EMDB1_LIST, 2: EMDB2_LIST}


def load_pkl(fp):
    annot = pickle.load(open(fp, "rb"))
    # ['gender', 'name', 'emdb1', 'emdb2', 'n_frames', 'good_frames_mask', 'camera', 'smpl', 'kp2d', 'bboxes', 'subfolder']
    data = {}

    F = annot["n_frames"]
    data["smpl_params"] = {
        "smpl_body_pose": annot["smpl"]["poses_body"],  # (F, 69)
        "smpl_shape": annot["smpl"]["betas"][None].repeat(F, axis=0),  # (F, 10)
        "smpl_root_pose_w": annot["smpl"]["poses_root"],  # (F, 3)
        "smpl_transl_w": annot["smpl"]["trans"],  # (F, 3)
    }

    data["name"] = annot["name"].replace('_', '/', 1)
    data["gender"] = annot["gender"]
    data["mask"] = annot["good_frames_mask"]   # (L,)
    data["K_fullimg"] = annot["camera"]["intrinsics"]  # (3, 3)
    data["T_w2c"] = annot["camera"]["extrinsics"]  # (L, 4, 4)

    return data

def to_tensor(ndarray):
    tensor = torch.from_numpy(ndarray).float()
    return tensor
