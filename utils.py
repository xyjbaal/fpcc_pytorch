import json
import numpy as np


def distance_matrix(vector1,vector2):
    '''
    vector1 : (N1,d)
    vector2 : (N2,d)
    return : (N1,N2)
    '''
    r_1 = np.sum(np.square(vector1), axis=-1)
    r_2 = np.sum(np.square(vector2), axis=-1)
    r_2 = r_2.reshape([1,-1])
    r_1 = np.reshape(r_1, [-1, 1])
    s = 2 * np.matmul(vector1, np.transpose(vector2, axes=[1, 0]))
    dis_martix = r_1 - s + r_2

    return dis_martix

def GroupMerging_fpcc(pts, pts_features, center_scores, center_socre_th=0.5, max_feature_dis=None, use_3d_mask=None, r_nms=1):
    """
    input:
        pts: xyz of point cloud
        pts_features: 128-dim feature of each point Nx128
        center_scores： center_score of each pint Nx1
    Returns:
    """

    validpts_index = np.where(center_scores > center_socre_th)[0]


    validpts = pts[validpts_index,:]
    validscore = center_scores[validpts_index]

    validscore = validscore.reshape(-1,1)
    validpts_index = validpts_index.reshape(-1,1)


    candidate_point_selected = np.concatenate((validpts,validscore, validpts_index),axis=1)

    heightest_point_selected = []

    validscore = validscore.reshape(-1)
    order = validscore.argsort()[::-1]
    center_points = []
    while order.size > 0:
        i = order[0]
        center_points.append(candidate_point_selected[i])
        distance = np.sqrt(np.sum((candidate_point_selected[order,:3]-candidate_point_selected[i,:3])**2,axis=1))
        remain_index = np.where(distance > r_nms)

        order = order[remain_index]

    center_points = np.concatenate(center_points, axis=0)
    center_points = center_points.reshape((-1,5))
    # print('number of instances',center_points.shape[0])

    center_index = np.array(center_points[:,-1]).astype(int)

    center_point_features = pts_features[center_index]

    pts_corr = distance_matrix(center_point_features,pts_features)

    if use_3d_mask is not None:
        pts_c = pts[center_index]

        dis_mask = distance_matrix(pts_c, pts)
        dis_mask = np.sqrt(dis_mask)
        pts_corr[np.where(dis_mask > use_3d_mask)] = 999

    groupid = np.argmin(pts_corr,axis=0)

    if max_feature_dis is not None:
        pts_corr_min = np.min(pts_corr, axis=0)
        over_threshold = np.where(pts_corr_min>max_feature_dis)
        groupid[over_threshold] = -1

    return groupid, center_index


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

color_map = json.load(open('part_color_mapping.json', 'r'))

def output_color_point_center_score(data, c_s, out_file):

    c_s = c_s.reshape((-1,1))
    point_c_s = np.hstack((data,c_s))

    np.savetxt(out_file, point_c_s, fmt='%0.6f')

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            # f.write('%f %f %f %d %d %d\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
            # color[0] = math.floor(color[0]*255)
            # color[1] = math.floor(color[1]*255)
            # color[2] = math.floor(color[2]*255)
            # f.write('%f %f %f %d\n' % (data[i][0], data[i][1], data[i][2],i))
            f.write('%f %f %f %d %d %d\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))