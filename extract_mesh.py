import argparse
import torch
import numpy as np
from scipy.sparse import coo_matrix
import trimesh
from torch.nn import functional as F
import sys
import os
import tqdm
from collections import defaultdict

import sys

sys.path.append('custom_mc')
from custom_mc._marching_cubes_lewiner import udf_mc_lewiner


def get_udf_normals_grid_slow(func, func_grad, N=56, max_batch=int(2 ** 20)):
    """
    Fills a dense N*N*N regular grid by querying the decoder network
    Inputs:
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        N: grid size
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """
    # network.eval()
    ################
    # 1: setting up the empty grid
    ################
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 7)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = torch.div(overall_index, N, rounding_mode='floor') % N
    samples[:, 0] = torch.div(torch.div(overall_index, N, rounding_mode='floor'), N, rounding_mode='floor') % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    num_samples = N ** 3
    samples.requires_grad = False
    # samples.pin_memory()
    ################
    # 2: Run forward pass to fill the grid
    ################
    head = 0
    ## FIRST: fill distance field grid without gradients
    while head < num_samples:
        # xyz coords
        sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].clone().cuda()
        # Create input
        xyz = sample_subset

        input = xyz.reshape(-1, xyz.shape[-1])
        # Run forward pass
        with torch.no_grad():
            df = func(input)
        # Store df
        samples[head: min(head + max_batch, num_samples), 3] = df.squeeze(-1).detach().cpu()
        # Next iter
        head += max_batch
    #
    ## THEN: compute gradients only where needed,
    # ie. where the predicted df value is small
    max_batch = max_batch // 4
    norm_mask = samples[:, 3] < 2 * voxel_size
    norm_idx = torch.where(norm_mask)[0]
    head, num_samples = 0, norm_idx.shape[0]
    while head < num_samples:
        # Find the subset of indices to compute:
        # -> a subset of indices where normal computations are needed
        sample_subset_mask = torch.zeros_like(norm_mask)
        sample_subset_mask[norm_idx[head]: norm_idx[min(head + max_batch, num_samples) - 1] + 1] = True
        sample_subset_mask = norm_mask * sample_subset_mask
        # xyz coords
        sample_subset = samples[sample_subset_mask, 0:3].clone().cuda()
        sample_subset.requires_grad = True
        # Create input
        xyz = sample_subset

        input = xyz.reshape(-1, xyz.shape[-1])

        grad = func_grad(input).detach()[:, 0]
        samples[sample_subset_mask, 4:] = - F.normalize(grad, dim=1)
        # Next iter
        head += max_batch
    #
    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)
    return df_values, vecs, samples


def get_udf_normals_grid_fast(func, func_grad, samples, indices, N=56, max_batch=int(2 ** 18)):
    """
    Updates the N*N*N regular grid by querying the decoder network ONLY AT INDICES
    Inputs:
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one.
        indices: tensor representing the coordinates that need updating
        N: grid size
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """
    # If there is no indicesm fallback to the slow version
    if indices is None or samples is None:
        return get_udf_normals_grid_slow(func, func_grad, N, max_batch)
    # network.eval()
    ################
    # 1: setting up the empty grid: no longer needed
    ################
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    num_samples = indices.shape[0]
    samples.requires_grad = False
    samples.pin_memory()
    ################
    # 2: Run forward pass to fill the grid, only where needed
    ################
    head = 0
    ## Fill distance field grid with gradients
    while head < num_samples:
        # xyz coords
        sample_subset = samples[indices[head: min(head + max_batch, num_samples)], 0:3].clone().cuda()
        sample_subset.requires_grad = True
        # Create input
        xyz = sample_subset

        input = xyz.reshape(-1, xyz.shape[-1])
        # Run forward pass
        with torch.no_grad():
            df = func(input)
        # Store df
        samples[indices[head: min(head + max_batch, num_samples)], 3] = df.squeeze(-1).detach().cpu()
        # Compute and store normalized vectors pointing towards the surface

        grad = func_grad(input).detach()[:, 0]
        samples[indices[head: min(head + max_batch, num_samples)], 4:] = - F.normalize(grad, dim=1).cpu()
        # Next iter
        head += max_batch
    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(N, N, N)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(N, N, N, 3)
    return df_values, vecs, samples


def get_mesh_udf_fast(func, func_grad, samples=None, indices=None, N_MC=128,
                      gradient=True, eps=0.1, border_gradients=False, smooth_borders=False, dist_threshold_ratio=1.):
    """
    Computes a triangulated mesh from a distance field network conditioned on the latent vector
    Inputs:
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one, if any
        indices: tensor representing the coordinates that need updating in the previous samples tensor (to speed
                    up iterations)
        N_MC: grid size
        fourier: are xyz coordinates encoded with fourier?
        gradient: do we need gradients?
        eps: length of the normal vectors used to derive gradients
        border_gradients: add a special case for border gradients?
        smooth_borders: do we smooth borders with a Laplacian?
        dist_threshold_ratio: remove the surfaces with large udf values (> voxel_size*dist_threshold_ratio);
    Returns:
        verts: vertices of the mesh
        faces: faces of the mesh
        mesh: trimesh object of the mesh
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
        indices: tensor representing the coordinates that need updating in the next iteration
    """
    ### 1: sample grid
    df_values, normals, samples = get_udf_normals_grid_fast(func, func_grad, samples, indices, N=N_MC)
    df_values[df_values < 0] = 0
    ### 2: run our custom MC on it
    N = df_values.shape[0]
    voxel_size = 2.0 / (N - 1)
    voxel_origin = [-1, -1, -1]
    verts, faces, _, _ = udf_mc_lewiner(df_values.cpu().detach().numpy(),
                                        normals.cpu().detach().numpy(),
                                        spacing=[voxel_size] * 3)
    verts = verts - 1  # since voxel_origin = [-1, -1, -1]
    ### 3: evaluate vertices DF, and remove the ones that are too far
    verts_torch = torch.from_numpy(verts).float().cuda()
    with torch.no_grad():
        xyz = verts_torch
        pred_df_verts = func(xyz)
    pred_df_verts = pred_df_verts.cpu().numpy()
    # Remove faces that have vertices far from the surface
    # dist_threshold = voxel_size / 6
    dist_threshold = voxel_size * dist_threshold_ratio
    filtered_faces = faces[np.max(pred_df_verts[faces], axis=1)[:, 0] < dist_threshold]
    filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
    ### 4: clean the mesh a bit
    # Remove NaNs, flat triangles, duplicate faces
    filtered_mesh = filtered_mesh.process(
        validate=False)  # DO NOT try to consistently align winding directions: too slow and poor results
    filtered_mesh.remove_duplicate_faces()
    filtered_mesh.remove_degenerate_faces()
    # Fill single triangle holes
    filtered_mesh.fill_holes()

    filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
    # Re-process the mesh until it is stable:
    n_verts, n_faces, n_iter = 0, 0, 0
    while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter < 10:
        filtered_mesh_2 = filtered_mesh_2.process(validate=False)
        filtered_mesh_2.remove_duplicate_faces()
        filtered_mesh_2.remove_degenerate_faces()
        (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
        n_iter += 1
        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    if smooth_borders:
        # Identify borders: those appearing only once
        border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

        # Build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
        neighbours = defaultdict(lambda: [])
        for (u, v) in filtered_mesh.edges_sorted[border_edges]:
            neighbours[u].append(v)
            neighbours[v].append(u)
        border_vertices = np.array(list(neighbours.keys()))

        # Build a sparse matrix for computing laplacian
        pos_i, pos_j = [], []
        for k, ns in enumerate(neighbours.values()):
            for j in ns:
                pos_i.append(k)
                pos_j.append(j)

        sparse = coo_matrix((np.ones(len(pos_i)),  # put ones
                             (pos_i, pos_j)),  # at these locations
                            shape=(len(border_vertices), len(filtered_mesh.vertices)))

        # Smoothing operation:
        lambda_ = 0.3
        for _ in range(5):
            border_neighbouring_averages = sparse @ filtered_mesh.vertices / sparse.sum(axis=1)
            laplacian = border_neighbouring_averages - filtered_mesh.vertices[border_vertices]
            filtered_mesh.vertices[border_vertices] = filtered_mesh.vertices[border_vertices] + lambda_ * laplacian

    if not gradient:
        return torch.tensor(filtered_mesh.vertices).float().cuda(), torch.tensor(
            filtered_mesh.faces).long().cuda(), filtered_mesh
    else:
        ### 5: use the mesh to compute normals
        normals = trimesh.geometry.weighted_vertex_normals(vertex_count=len(filtered_mesh.vertices),
                                                           faces=filtered_mesh.faces,
                                                           face_normals=filtered_mesh.face_normals,
                                                           face_angles=filtered_mesh.face_angles)
        ### 6: evaluate the DF around each vertex, based on normals
        normals = torch.tensor(normals).float().cuda()
        verts = torch.tensor(filtered_mesh.vertices).float().cuda()

        xyz_s1 = verts + eps * normals
        xyz_s2 = verts - eps * normals

        with torch.no_grad():
            s1 = func(xyz_s1)
            s2 = func(xyz_s2)
        # Re-plug differentiability here, by this rewriting trick
        new_verts = verts - eps * s1 * normals + eps * s2 * normals

        ## Compute indices needed for re-evaluation at the next iteration
        # fetch bins that are activated
        k = ((new_verts[:, 2].detach().cpu().numpy() - voxel_origin[2]) / voxel_size).astype(int)
        j = ((new_verts[:, 1].detach().cpu().numpy() - voxel_origin[1]) / voxel_size).astype(int)
        i = ((new_verts[:, 0].detach().cpu().numpy() - voxel_origin[0]) / voxel_size).astype(int)
        # find points around
        next_samples = i * N_MC * N_MC + j * N_MC + k
        next_samples_ip = np.minimum(i + 1, N_MC - 1) * N_MC * N_MC + j * N_MC + k
        next_samples_jp = i * N_MC * N_MC + np.minimum(j + 1, N_MC - 1) * N_MC + k
        next_samples_kp = i * N_MC * N_MC + j * N_MC + np.minimum(k + 1, N - 1)
        next_samples_im = np.maximum(i - 1, 0) * N_MC * N_MC + j * N_MC + k
        next_samples_jm = i * N_MC * N_MC + np.maximum(j - 1, 0) * N_MC + k
        next_samples_km = i * N_MC * N_MC + j * N_MC + np.maximum(k - 1, 0)
        # Concatenate
        next_indices = np.concatenate((next_samples, next_samples_ip, next_samples_jp,
                                       next_samples_kp, next_samples_im, next_samples_jm, next_samples_km))

        if border_gradients:
            ### 7: Add gradients at the surface borders?
            # Identify borders
            border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

            # Build a dictionnary of (u,v) edges, such that each vertex on the border
            # gets associated to exactly one border edge
            border_edges_dict = {}
            for (u, v) in filtered_mesh.edges_sorted[border_edges]:
                border_edges_dict[u] = v
                border_edges_dict[v] = u
            u_v_border = np.array(list(border_edges_dict.items()))
            u_border = u_v_border[:, 0]  # split border edges (u,v) into u and v arrays
            v_border = u_v_border[:, 1]

            # For each vertex on the border, take the cross product between
            # its normal and the border's edge
            normals_border = normals[u_border]
            edge_border = filtered_mesh.vertices[v_border] - filtered_mesh.vertices[u_border]
            edge_border = torch.tensor(edge_border).float().cuda()
            out_vec = torch.cross(edge_border, normals_border, dim=1)
            out_vec = out_vec / (torch.norm(out_vec, dim=1, keepdim=True) + 1e-6)  # make it unit length

            # Then we need to orient the out_vec such that they point outwards
            # To do so, we evaluate at +- their offset, and take the corresponding max DF value
            border_verts = torch.tensor(filtered_mesh.vertices[u_border]).float().cuda()

            xyz_s1_border = border_verts + 3 * eps * out_vec
            xyz_s2_border = border_verts - 3 * eps * out_vec

            with torch.no_grad():
                s1_border = func(xyz_s1_border)
                s2_border = func(xyz_s2_border)
            s1s2 = torch.stack((s1_border, s2_border))
            sign_out_vec = -torch.argmax(s1s2, dim=0) * 2 + 1
            out_vec = sign_out_vec * out_vec

            # Filter out the verts borders for which a displacement of out_vec
            # still evaluates at < eps DF, ie. verts classified as borders which are not really so
            u_border_filtered = u_border[((s1_border + s2_border)[:, 0] > eps).detach().cpu().numpy()]
            out_vec_filtered = out_vec[(s1_border + s2_border)[:, 0] > eps]
            out_df_filtered = torch.max(s1_border, s2_border)[(s1_border + s2_border) > eps]

            # Plug gradients to verts positions
            s_border = (eps * (out_df_filtered - out_df_filtered.detach())).unsqueeze(
                -1)  # Fake zero, just to pass grads
            new_verts[u_border_filtered] = new_verts[u_border_filtered] - s_border * out_vec_filtered

        return new_verts, torch.tensor(filtered_mesh.faces).long().cuda(), filtered_mesh, samples, next_indices


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Reconstruct training shapes with a trained DeepSDF autodecoder (+latents)")
    arg_parser.add_argument(
        "--experiment", "-e", dest="experiment_directory", required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue", "-c", dest="continue_from", default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
             + "from the latest running snapshot, or an integer corresponding to "
             + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--MC", type=int, default=128,
        help="What is marching cubes resolution?",
    )
    arg_parser.add_argument(
        "--A", type=int, default=0,
        help="Starting shape id?",
    )
    arg_parser.add_argument(
        "--B", type=int, default=2,
        help="Target shape id?",
    )
    arg_parser.add_argument(
        "--no_border_gradients", dest="no_border_gradients", action="store_true",
        help="Do we skip the special case for gradients on the borders (to allow surface extension/shrinking?)",
    )
    arg_parser.add_argument(
        "--no_smooth_borders", dest="no_smooth_borders", action="store_true",
        help="Do we skip the border smoothing step?",
    )
    arg_parser.add_argument(
        "--iters", type=int, default=800,
        help="Optimization steps",
    )

    args = arg_parser.parse_args()

    out_dir = os.path.join(args.experiment_directory, f'optim_{args.A}_to_{args.B}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('======================================')
    print(f'Meshing the UDF directly')
    print(f'And optimizing CHD loss from shape {args.A} to shape {args.B}')
    print(f'MC resolution: {args.MC}')
    print(f'Results will be stored in: {out_dir}')
    print('======================================')
