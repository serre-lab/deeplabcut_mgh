# coding: utf-8
import glob, os
import pandas as pd
import numpy as np

header1 = "scorer, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit, Kalpit"
header2 = "bodyparts, Nose, Nose, RightShoulder, RightShoulder, RightElbow, RightElbow, RightWrist, RightWrist, LeftShoulder, LeftShouler, LeftElbow, LeftElbow, LeftWrist, LeftWrist"
header3 = "coords, x, y, x, y, x, y, x, y, x, y, x, y, x, y"
prefix = "labeled-data"
label_path = "/media/ssd_storage/work_repos/MGH/bootstrap_round3_2019_08_14_labels"

subs = ['BW46', 'MG51b', 'MG117', 'MG118', 'MG120b']
exp_dirs = [
    'mgh_pose_dlc_BW46_2-Kalpit-2019-08-14',
    'mgh_pose_dlc_MG51b_2-Kalpit-2019-08-14',
    'mgh_pose_dlc_MG117_2-Kalpit-2019-08-14',
    'mgh_pose_dlc_MG118_2-Kalpit-2019-08-14',
    'mgh_pose_dlc_MG120b_2-Kalpit-2019-08-14'
]

for s, e in zip(subs, exp_dirs):
    print("Processing {}'s directory {}".format(s, e))
    lab_path = os.path.join(label_path, s)
    label_vid_dirs = os.listdir(lab_path)
    pref = os.path.join(e, prefix)
    vid_dirs = os.listdir(pref)

    for v in vid_dirs:
        path_v = os.path.join(pref, v)
        suffix_v = '_'.join(v.split('_')[1:])
        assert 'pose_' + suffix_v + '.csv' in label_vid_dirs
        path_l = os.path.join(lab_path, 'pose_' + suffix_v + '.csv')
        img_string = "img%04d"
        with open(path_l, 'r') as f:
            labels = f.readlines()
        headers = labels[0]
        with open(os.path.join(path_v, 'frame_numbers.txt'), 'r') as f:
            chosen_frames = f.readlines()[0].strip().split(',')
        chosen_frames = [int(x) for x in chosen_frames]
        updated_frames = []
        with open(os.path.join(path_v, 'CollectedData_Kalpit.csv'), 'w') as f:
            f.write(header1 + "\n")
            f.write(header2 + "\n")
            f.write(header3 + "\n")
            for i in range(1, len(labels)):
                if i in chosen_frames:
                    line = labels[i].strip().split(',')
                    qual = int(line[-1])
                    if qual == -1:
                        continue
                    updated_frames.append(i)
                    joints = line[3:10]
                    strtowrite = os.path.join(prefix, v, img_string % i + '.png')
                    for joint in joints:
                        x, y, _ = joint.split('-')
                        x = float(x); y = float(y)
                        strtowrite += ", " + str(x) + ", " + str(y)
                    strtowrite += "\n"
                    f.write(strtowrite)
            with open(os.path.join(path_v, 'frame_numbers_updated.txt'), 'w') as f:
                strtowrite = str(updated_frames[0])
                for n in updated_frames[1:]:
                    strtowrite += "," + str(n)
                f.write(strtowrite)

    for v in vid_dirs:
        path_v = os.path.join(pref, v)
        with open(os.path.join(path_v, 'CollectedData_Kalpit.csv'), 'r') as f:
            out = f.readlines()
        image_names = [x.strip().split(',')[0] for x in out[3:]]
        a = np.empty((len(image_names), 2,))
        a[:] = np.nan
        bparts = ['Nose', 'RightShoulder', 'RightElbow', 'RightWrist', 'LeftShoulder', 'LeftElbow', 'LeftWrist']
        scorer = 'Kalpit'
        dataFrame = None
        for bpart in bparts:
            index = pd.MultiIndex.from_product([[scorer], [bpart], ['x', 'y']], names=['scorer', 'bodyparts', 'coords'])
            frame = pd.DataFrame(a, columns = index, index = image_names)
            dataFrame = pd.concat([dataFrame, frame], axis=1)
        for it, line in enumerate(out[3:]):
            line = line.strip().split(',')[1:]
            for idx, bp in enumerate(bparts):
                dataFrame.loc[image_names[it]][scorer, bp, 'x'] = line[idx*2]
                dataFrame.loc[image_names[it]][scorer, bp, 'y'] = line[idx*2+1]
        dataFrame.to_hdf(os.path.join(path_v, 'CollectedData_Kalpit.h5'), key='df_with_missing', format='table', mode='w') 
