import numpy as np
import cv2 as cv
import os
from glob import glob
import trimesh
import torch
import sys

sys.path.append("../")


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


def clean_points_by_mask(points, scan, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    cameras = np.load('{}/scan{}/cameras.npz'.format(DTU_DIR, scan))
    mask_lis = sorted(glob('{}/scan{}/mask/*.png'.format(DTU_DIR, scan)))
    n_images = 49 if scan < 83 else 64
    inside_mask = np.zeros(len(points))

    if imgs_idx is None:
        imgs_idx = [i for i in range(n_images)]

    for i in imgs_idx:
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel_size = mask_dilated_size  # default 101
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] > 128)

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        in_mask = (pts_image[:, 0] >= 0) * (pts_image[:, 0] <= 1600) * (pts_image[:, 1] >= 0) * (
                pts_image[:, 1] <= 1200) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        curr_mask = curr_mask.astype(np.float32) * in_mask

        inside_mask += curr_mask

    return inside_mask > minimal_vis


def clean_points_by_visualhull(points, scan, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    cameras = np.load('{}/scan{}/cameras.npz'.format(DTU_DIR, scan))
    mask_lis = sorted(glob('{}/scan{}/mask/*.png'.format(DTU_DIR, scan)))
    n_images = 49 if scan < 83 else 64
    outside_mask = np.zeros(len(points))

    if imgs_idx is None:
        imgs_idx = [i for i in range(n_images)]

    for i in imgs_idx:
        P = cameras['world_mat_{}'.format(i)]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32) + 1

        mask_image = cv.imread(mask_lis[i])
        kernel_size = mask_dilated_size  # default 101
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :, 0] < 128)  # * outside the mask

        mask_image = np.concatenate([np.ones([1, 1600]), mask_image, np.ones([1, 1600])], axis=0)
        mask_image = np.concatenate([np.ones([1202, 1]), mask_image, np.ones([1202, 1])], axis=1)

        border = 50
        in_mask = (pts_image[:, 0] >= (0 + border)) * (pts_image[:, 0] <= (1600 - border)) * (
                pts_image[:, 1] >= (0 + border)) * (
                          pts_image[:, 1] <= (1200 - border)) > 0
        curr_mask = mask_image[(pts_image[:, 1].clip(0, 1201), pts_image[:, 0].clip(0, 1601))]

        curr_mask = curr_mask.astype(np.float32) * in_mask

        outside_mask += curr_mask

    return outside_mask < 5


def clean_mesh_faces_by_mask(mesh_file, new_mesh_file, scan, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, scan, imgs_idx, minimal_vis, mask_dilated_size)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.long)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    # ! if colmap trim=7, comment these
    # meshes = new_mesh.split(only_watertight=False)
    # new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(new_mesh_file)


def clean_mesh_faces_by_visualhull(mesh_file, new_mesh_file, scan, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_visualhull(old_vertices, scan, imgs_idx, minimal_vis, mask_dilated_size)
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.long)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    # ! if colmap trim=7, comment these
    # meshes = new_mesh.split(only_watertight=False)
    # new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(new_mesh_file)


def clean_mesh_by_faces_num(mesh, faces_num=500):
    old_vertices = mesh.vertices[:]
    old_faces = mesh.faces[:]

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=faces_num)
    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[np.concatenate(cc)] = True

    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.long)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    return new_mesh


def clean_outliers(old_mesh_file, new_mesh_file, faces_num=500, keep_largest=True):
    new_mesh = trimesh.load(old_mesh_file)

    if keep_largest:
        meshes = new_mesh.split(only_watertight=False)
        new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    else:
        new_mesh = clean_mesh_by_faces_num(new_mesh, faces_num)

    new_mesh.export(new_mesh_file)


if __name__ == "__main__":

    # scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
    scans = [24]
    mask_kernel_size = 11

    DTU_DIR = "~/dataset/DTU_IDR/DTU"

    for scan in scans:
        base_path = f"exp/udf/dtu/mesh/ours"
        print("processing scan%d" % scan)
        dir_path = os.path.join(base_path, f'scan{scan}')

        old_mesh_file = glob(os.path.join(dir_path, "*.ply"))[0]

        clean_mesh_file = os.path.join(dir_path, "clean_%03d.ply" % scan)
        visualhull_mesh_file = os.path.join(dir_path, "visualhull_%03d.ply" % scan)
        final_mesh_file = os.path.join(dir_path, "final_%03d.ply" % scan)

        clean_mesh_faces_by_mask(old_mesh_file, clean_mesh_file, scan, None, minimal_vis=2,
                                 mask_dilated_size=mask_kernel_size)

        clean_mesh_faces_by_visualhull(clean_mesh_file, visualhull_mesh_file, scan, None, minimal_vis=2,
                                       mask_dilated_size=mask_kernel_size + 20)
        # optional; keep the largest mesh or remove small isolated meshes
        # clean_outliers(visualhull_mesh_file, visualhull_mesh_file, keep_largest=True)  # do not need
        print("finish processing scan%d" % scan)
