import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse, os, sys
import cv2 as cv

from pathlib import Path


def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


if __name__ == '__main__':
    from glob import glob

    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--gt', type=str, help='ground truth')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='/dataset/dtu_official/SampleSet/MVS_Data')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.002)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=0.1)
    parser.add_argument('--visualize_threshold', type=float, default=0.01)
    parser.add_argument('--log', type=str, default=None)
    args = parser.parse_args()

    method = 'colmap'

    # scans = [30, 92, 117, 133, 164, 204, 300, 320, 448, 522, 591, 598]
    scans = [300]

    for scan in scans:

        GT_DIR = f"/dataset/deepfashion3d_point_cloud/{scan}"

        base_dir = "/exp/udf/deepfashion3d/300/expname"

        print("processing scan%d" % scan)

        # args.data = os.path.join(base_dir, "udf_res512_step100000.ply")
        args.data = glob(os.path.join(base_dir, "*.ply"))[0]
        # args.data = glob(os.path.join(base_dir, "*trim7.ply"))[0]

        if not os.path.exists(args.data):
            continue

        args.gt = os.path.join(GT_DIR, "%d_pc_swap.ply" % scan)
        args.vis_out_dir = os.path.join(base_dir, "scan{}".format(scan))
        args.scan = scan
        os.makedirs(args.vis_out_dir, exist_ok=True)

        dist_thred1 = 0.001
        dist_thred2 = 0.002

        thresh = args.downsample_density

        if args.mode == 'mesh':
            pbar = tqdm(total=9)
            pbar.set_description('read data mesh')
            data_mesh = o3d.io.read_triangle_mesh(args.data)

            vertices = np.asarray(data_mesh.vertices)
            triangles = np.asarray(data_mesh.triangles)
            tri_vert = vertices[triangles]

            pbar.update(1)
            pbar.set_description('sample pcd from mesh')
            v1 = tri_vert[:, 1] - tri_vert[:, 0]
            v2 = tri_vert[:, 2] - tri_vert[:, 0]
            l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
            l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
            area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
            non_zero_area = (area2 > 0)[:, 0]
            l1, l2, area2, v1, v2, tri_vert = [
                arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
            ]
            thr = thresh * np.sqrt(l1 * l2 / area2)
            n1 = np.floor(l1 / thr)
            n2 = np.floor(l2 / thr)

            with mp.Pool() as mp_pool:
                new_pts = mp_pool.map(sample_single_tri,
                                      ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                                       range(len(n1))), chunksize=1024)

            new_pts = np.concatenate(new_pts, axis=0)
            data_pcd = np.concatenate([vertices, new_pts], axis=0)

        elif args.mode == 'pcd':
            pbar = tqdm(total=8)
            pbar.set_description('read data pcd')
            data_pcd_o3d = o3d.io.read_point_cloud(args.data)
            data_pcd = np.asarray(data_pcd_o3d.points)

        pbar.update(1)
        pbar.set_description('random shuffle pcd index')
        shuffle_rng = np.random.default_rng()
        shuffle_rng.shuffle(data_pcd, axis=0)

        pbar.update(1)
        # pbar.set_description('downsample pcd')
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]
        # data_down = data_pcd

        pbar.update(1)
        pbar.set_description('read STL pcd')
        stl_pcd = o3d.io.read_point_cloud(args.gt)
        stl = np.asarray(stl_pcd.points)

        pbar.update(1)
        pbar.set_description('compute data2stl')
        nn_engine.fit(stl)
        dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
        max_dist = args.max_dist
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        precision_1 = len(dist_d2s[dist_d2s < dist_thred1]) / len(dist_d2s)
        precision_2 = len(dist_d2s[dist_d2s < dist_thred2]) / len(dist_d2s)

        pbar.update(1)
        pbar.set_description('compute stl2data')

        nn_engine.fit(data_down)
        dist_s2d, idx_s2d = nn_engine.kneighbors(stl, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        recall_1 = len(dist_s2d[dist_s2d < dist_thred1]) / len(dist_s2d)
        recall_2 = len(dist_s2d[dist_s2d < dist_thred2]) / len(dist_s2d)

        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = args.visualize_threshold
        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 0, 1]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color = R * data_alpha + W * (1 - data_alpha)
        data_color[dist_d2s[:, 0] >= max_dist] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2gt.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color = R * stl_alpha + W * (1 - stl_alpha)
        stl_color[dist_s2d[:, 0] >= max_dist] = G
        write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_gt2d.ply', stl, stl_color)

        pbar.update(1)
        pbar.set_description('done')
        pbar.close()
        over_all = (mean_d2s + mean_s2d) / 2

        fscore_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-6)
        fscore_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-6)

        print(f'over_all: {over_all}; mean_d2gt: {mean_d2s}; mean_gt2d: {mean_s2d}.')
        print(f'precision_1mm: {precision_1};  recall_1mm: {recall_1};  fscore_1mm: {fscore_1}')
        print(f'precision_2mm: {precision_2};  recall_2mm: {recall_2};  fscore_2mm: {fscore_2}')

        pparent, stem, ext = get_path_components(args.data)
        if args.log is None:
            path_log = os.path.join(pparent, 'eval_result.txt')
        else:
            path_log = args.log
        with open(path_log, 'w+') as fLog:
            fLog.write(f'over_all {np.round(over_all, 6)} '
                       f'mean_d2gt {np.round(mean_d2s, 6)} '
                       f'mean_gt2d {np.round(mean_s2d, 6)} \n'
                       f'precision_1mm {np.round(precision_1, 6)} '
                       f'recall_1mm {np.round(recall_1, 6)} '
                       f'fscore_1mm {np.round(fscore_1, 6)} \n'
                       f'precision_2mm {np.round(precision_2, 6)} '
                       f'recall_2mm {np.round(recall_2, 6)} '
                       f'fscore_2mm {np.round(fscore_2, 6)} \n'
                       f'[{stem}] \n')
