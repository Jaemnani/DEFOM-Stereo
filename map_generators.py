import numpy as np
import cv2

def generate_geometric_costmap(pcd, res=0.05, max_cost=255.0, bev_width_m=20.0, bev_depth_m=20.0):
    """
    Optimized Geometric Costmap Generator (>60FPS target)
    """
    if len(pcd.shape) == 3:
        pcd = pcd.reshape(-1, 3)
        
    # Pre-allocate indices
    min_x, max_x = -bev_width_m / 2.0, bev_width_m / 2.0
    min_z, max_z = 0.0, bev_depth_m
    width = int((max_x - min_x) / res)
    height = int((max_z - min_z) / res)
    
    # 1. Vectorized Filtering
    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    valid = (Z > min_z) & (Z < max_z) & (X > min_x) & (X < max_x) & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[valid], Y[valid], Z[valid]
    
    if len(X) == 0:
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool), (min_x, min_z, res)

    # 2. Vectorized Grid Mapping
    grid_x = np.clip(np.floor((X - min_x) / res).astype(np.int32), 0, width - 1)
    grid_z = np.clip(np.floor((max_z - Z) / res).astype(np.int32), 0, height - 1)
    
    # Linearize indices for fast 1D bincount/reduceat operations
    flat_indices = grid_z * width + grid_x
    
    # 3. Fast Accumulation using bincount for counts and sums
    count_flat = np.bincount(flat_indices, minlength=height*width)
    y_sum_flat = np.bincount(flat_indices, weights=Y, minlength=height*width)
    y_sq_sum_flat = np.bincount(flat_indices, weights=Y**2, minlength=height*width)
    
    mask_flat = count_flat > 0
    mean_y_flat = np.zeros_like(y_sum_flat, dtype=np.float32)
    mean_y_flat[mask_flat] = y_sum_flat[mask_flat] / count_flat[mask_flat]
    
    var_y_flat = np.zeros_like(y_sum_flat, dtype=np.float32)
    var_y_flat[mask_flat] = (y_sq_sum_flat[mask_flat] / count_flat[mask_flat]) - mean_y_flat[mask_flat]**2
    var_y_flat = np.clip(var_y_flat, 0.0, None)
    
    # Fast Min/Max per cell using lexsort
    sort_idx = np.argsort(flat_indices)
    sorted_flat_indices = flat_indices[sort_idx]
    sorted_Y = Y[sort_idx]
    
    # Find bin boundaries
    bin_starts = np.searchsorted(sorted_flat_indices, np.arange(height*width))
    bin_ends = np.append(bin_starts[1:], len(sorted_Y))
    
    valid_bins = mask_flat
    
    # For speed, use bincounttrick for min/max if possible, but loop on valid bins is faster than np.add.at
    # Actually, a much faster way for min/max is using np.minimum.at/maximum.at, but on the flattened array.
    max_y_flat = np.full(height*width, -np.inf, dtype=np.float32)
    min_y_flat = np.full(height*width, np.inf, dtype=np.float32)
    np.maximum.at(max_y_flat, flat_indices, Y)
    np.minimum.at(min_y_flat, flat_indices, Y)
    
    height_diff_flat = np.zeros_like(y_sum_flat, dtype=np.float32)
    height_diff_flat[mask_flat] = max_y_flat[mask_flat] - min_y_flat[mask_flat]
    
    # 4. Reshape back
    mean_y = mean_y_flat.reshape(height, width)
    var_y = var_y_flat.reshape(height, width)
    height_diff = height_diff_flat.reshape(height, width)
    mask = mask_flat.reshape(height, width)
    
    # 5. Gradient and Cost
    grad_z, grad_x = np.gradient(mean_y)
    slope = np.sqrt(grad_x**2 + grad_z**2)
    
    cost = slope * 50. + var_y * 500. + height_diff * 200.
    cost = np.clip(cost, 0, max_cost).astype(np.uint8)
    cost[~mask] = int(max_cost)
    
    map_info = (min_x, min_z, res)
    return cost, mean_y, mask, map_info


def generate_pure_terrain_map(pcd, res=0.05, inlier_thresh=0.1, bev_width_m=20.0, bev_depth_m=20.0, downsample_step=50):
    """
    Optimized Terrain Map Generator (>60FPS target)
    Uses vectorized operations and downsampling for fast ground plane extraction.
    """
    if len(pcd.shape) == 3:
        pcd = pcd.reshape(-1, 3)
        
    min_x, max_x = -bev_width_m / 2.0, bev_width_m / 2.0
    min_z, max_z = 0.0, bev_depth_m
    width = int((max_x - min_x) / res)
    height = int((max_z - min_z) / res)
    
    X, Y, Z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    valid = (Z > min_z) & (Z < max_z) & (X > min_x) & (X < max_x) & np.isfinite(Y) 
    pts = pcd[valid]
    
    if len(pts) < 10:
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=bool), (min_x, min_z, res)
        
    # 1. Fast RANSAC on downsampled points for massive speedup
    pts_down = pts[::downsample_step]
    if len(pts_down) < 3: pts_down = pts # fallback 
    
    num_pts = len(pts_down)
    iters = 30 # Reduced iterations for speed
    
    # Vectorized random choice
    idx1 = np.random.randint(0, num_pts, iters)
    idx2 = np.random.randint(0, num_pts, iters)
    idx3 = np.random.randint(0, num_pts, iters)
    
    p1 = pts_down[idx1]
    p2 = pts_down[idx2]
    p3 = pts_down[idx3]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Filter valid normals
    valid_norms = norms.squeeze() > 1e-6
    if not np.any(valid_norms):
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=bool), (min_x, min_z, res)
        
    normals = normals[valid_norms] / norms[valid_norms]
    p1 = p1[valid_norms]
    
    # Prefer vertical normals (Y axis dominant)
    vertical_mask = np.abs(normals[:, 1]) > 0.6
    if np.any(vertical_mask):
        normals = normals[vertical_mask]
        p1 = p1[vertical_mask]
        
    # Broadcast distance calculation: (N_pts, 3) dot (N_planes, 3)^T -> (N_pts, N_planes)
    # To save memory, evaluate on the downsampled points first
    distances_sample = np.abs(np.dot(pts_down, normals.T) - np.einsum('ij,ij->i', p1, normals)[None, :])
    inlier_counts = np.sum(distances_sample < inlier_thresh, axis=0)
    
    best_idx = np.argmax(inlier_counts)
    best_normal = normals[best_idx]
    best_p1 = p1[best_idx]
    
    # 2. Apply best plane to ALL points
    distances_full = np.dot(pts - best_p1, best_normal)
    is_ground = np.abs(distances_full) < inlier_thresh * 2 
    ground_pts = pts[is_ground]
    
    if len(ground_pts) == 0:
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=bool), (min_x, min_z, res)
        
    # 3. Vectorized Terrain Height Mapping (same bincount technique)
    X_g = ground_pts[:, 0]
    Y_g = ground_pts[:, 1]
    Z_g = ground_pts[:, 2]
    
    grid_x = np.clip(np.floor((X_g - min_x) / res).astype(np.int32), 0, width - 1)
    grid_z = np.clip(np.floor((max_z - Z_g) / res).astype(np.int32), 0, height - 1)
    flat_indices = grid_z * width + grid_x
    
    count_flat = np.bincount(flat_indices, minlength=height*width)
    y_sum_flat = np.bincount(flat_indices, weights=Y_g, minlength=height*width)
    
    mask_flat = count_flat > 0
    terrain_map_flat = np.zeros_like(y_sum_flat, dtype=np.float32)
    terrain_map_flat[mask_flat] = y_sum_flat[mask_flat] / count_flat[mask_flat]
    
    terrain_map = terrain_map_flat.reshape((height, width))
    mask = mask_flat.reshape((height, width))
    
    terrain_vis = np.zeros_like(terrain_map, dtype=np.uint8)
    if np.any(mask):
        t_min = np.min(terrain_map[mask])
        t_max = np.max(terrain_map[mask])
        if t_max > t_min:
            terrain_vis[mask] = 255.0 * (terrain_map[mask] - t_min) / (t_max - t_min)
        else:
            terrain_vis[mask] = 127
            
    map_info = (min_x, min_z, res)
    return terrain_vis, mask, map_info


def generate_drivable_region(costmap, mask, map_info, seed_x=0.0, seed_z=0.0, cost_thresh=50, max_z=20.0, method='cca'):
    """
    Extracts the drivable region.
    method: 'cca' (Connected Components, robust to seed issues) OR 'floodfill' (Original, exact seed expansion)
    return: drivable_mask (uint8 array, 255 if drivable)
    """
    min_x, min_z, res = map_info
    height, width = costmap.shape
    
    if method == 'cca':
        # 1. Define free space based on cost threshold and valid data mask
        free_space = ((costmap <= cost_thresh) & mask).astype(np.uint8) * 255
        
        # Optional: Light morphological opening to remove noise/narrow gaps before component analysis
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
        
        # 2. Find Connected Components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=4)
        
        # If no free space found (only background label 0 exists)
        if num_labels <= 1:
            return np.zeros_like(costmap, dtype=np.uint8)
            
        # 3. Filter components logic
        # Instead of taking the absolute largest area, we want the area that connects to the camera/ego vehicle.
        # Find the label that contains the bottom-center region (where the robot/camera is positioned)
        
        # Mapping seed to the orthogonal top-down grid (Z=0 is at the bottom, Z=max_z is at top)
        # Default seed is usually near (0,0) in world coordinates.
        grid_z = int((max_z - seed_z) / res)
        grid_x = int((seed_x - min_x) / res)
        
        target_label = 0
        # Check if the seed point itself is in a valid component
        if 0 <= grid_x < width and 0 <= grid_z < height:
            label_at_seed = labels[grid_z, grid_x]
            if label_at_seed > 0:
                target_label = label_at_seed
                
        # If seed is blocked, search for the nearest valid component at the bottom center.
        if target_label == 0:
            found = False
            start_z = min(height - 1, max(0, grid_z))
            start_x = min(width - 1, max(0, grid_x))
            for r in range(1, max(width, height)):
                for dz in range(-r, r + 1):
                    for dx in [-r, r]:
                        z, x = start_z + dz, start_x + dx
                        if 0 <= z < height and 0 <= x < width:
                            candidate_label = labels[z, x]
                            if candidate_label > 0:
                                target_label = candidate_label
                                found = True
                                break
                    if found: break
                if found: break
                
        # Fallback to largest if we completely failed to find a component near the seed
        if target_label == 0:
            areas = stats[1:, cv2.CC_STAT_AREA]
            target_label = np.argmax(areas) + 1  
        
        drivable = (labels == target_label)
        drivable_mask = (drivable.astype(np.uint8) * 255)
        drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
        
        return drivable_mask

    elif method == 'floodfill':
        # Mapping seed to the orthogonal top-down grid (Z=0 is at the bottom, Z=max_z is at top)
        grid_z = int((max_z - seed_z) / res)
        grid_x = int((seed_x - min_x) / res)
        
        if not (0 <= grid_x < width and 0 <= grid_z < height) or not mask[grid_z, grid_x] or costmap[grid_z, grid_x] > cost_thresh:
            # Search for closest valid seed near the requested point
            found = False
            start_z = min(height - 1, max(0, grid_z))
            start_x = min(width - 1, max(0, grid_x))
            # Spiral search could be better, but we do simple box expansion
            for r in range(1, max(width, height)):
                for dz in range(-r, r + 1):
                    for dx in [-r, r]:
                        z, x = start_z + dz, start_x + dx
                        if 0 <= z < height and 0 <= x < width and mask[z, x] and costmap[z, x] <= cost_thresh:
                            grid_z, grid_x = z, x
                            found = True
                            break
                    if found: break
                if found: break
                for dx in range(-r + 1, r):
                    for dz in [-r, r]:
                        z, x = start_z + dz, start_x + dx
                        if 0 <= z < height and 0 <= x < width and mask[z, x] and costmap[z, x] <= cost_thresh:
                            grid_z, grid_x = z, x
                            found = True
                            break
                    if found: break
                if found: break
                    
            if not found:
                return np.zeros_like(costmap, dtype=np.uint8) 
                
        # Flood Fill setup
        flood_img = costmap.copy()
        flood_mask = np.zeros((height + 2, width + 2), np.uint8)
        
        obstacle_mask = (costmap > cost_thresh) | (~mask)
        flood_img[obstacle_mask] = 255
        
        cv2.floodFill(flood_img, flood_mask, (grid_x, grid_z), 255, loDiff=cost_thresh, upDiff=cost_thresh, flags=4 | (255 << 8))
        
        drivable = flood_mask[1:-1, 1:-1] == 255
        drivable_mask = (drivable.astype(np.uint8) * 255)
        
        return drivable_mask


def generate_occupancy_voxel_map(pcd, voxel_size=0.1, max_depth=20.0):
    """
    Maps depth to a 3D Voxel Grid indicating occupancy.
    return: voxel_map, map_info(min_x, min_y, min_z, voxel_size)
    """
    if len(pcd.shape) == 3:
        pcd = pcd.reshape(-1, 3)
        
    X = pcd[:, 0]
    Y = pcd[:, 1]
    Z = pcd[:, 2]
    
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > 0.1) & (Z < max_depth)
    pts = pcd[valid]
    
    if len(pts) == 0:
        return np.zeros((1, 1, 1), dtype=np.uint8), (0,0,0, voxel_size)
    
    # Include camera origin conceptually for indexing consistency
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    min_z, max_z = np.min(pts[:, 2]), np.max(pts[:, 2])
    
    min_x = min(min_x, 0)
    min_y = min(min_y, 0)
    min_z = min(min_z, 0)
    
    dim_x = int(np.ceil((max_x - min_x) / voxel_size)) + 1
    dim_y = int(np.ceil((max_y - min_y) / voxel_size)) + 1
    dim_z = int(np.ceil((max_z - min_z) / voxel_size)) + 1
    
    # 0 for Unknown/Empty, 1 for Occupied
    voxel_map = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    
    idx_x = np.floor((pts[:, 0] - min_x) / voxel_size).astype(np.int32)
    idx_y = np.floor((pts[:, 1] - min_y) / voxel_size).astype(np.int32)
    idx_z = np.floor((pts[:, 2] - min_z) / voxel_size).astype(np.int32)
    
    voxel_map[idx_x, idx_y, idx_z] = 1 # Occupied
    
    map_info = (min_x, min_y, min_z, voxel_size)
    return voxel_map, map_info

def save_voxel_as_ply(voxel_map, map_info, filename, color_mode='height'):
    """
    Exports the occupied voxels in voxel_map to a .ply file for visualization.
    color_mode: 'height' (colors by Y axis) or 'distance' (colors by Z axis)
    """
    from plyfile import PlyData, PlyElement
    
    min_x, min_y, min_z, voxel_size = map_info
    
    # Find occupied voxel indices
    occupied_indices = np.argwhere(voxel_map == 1)
    
    if len(occupied_indices) == 0:
        print(f"No occupied voxels to save for {filename}")
        return
        
        
    # To make voxels appear larger, we will create 8 corner points for each voxel
    # instead of just 1 center point.
    cx = occupied_indices[:, 0] * voxel_size + min_x + (voxel_size / 2)
    cy = occupied_indices[:, 1] * voxel_size + min_y + (voxel_size / 2)
    cz = occupied_indices[:, 2] * voxel_size + min_z + (voxel_size / 2)
    
    half_v = voxel_size * 0.2 # slightly less than half to leave a tiny gap
    
    # 8 corners offset combinations
    offsets = np.array([
        [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1]
    ]) * half_v
    
    pts_x = (cx[:, None] + offsets[:, 0]).flatten()
    pts_y = (cy[:, None] + offsets[:, 1]).flatten()
    pts_z = (cz[:, None] + offsets[:, 2]).flatten()
    
    vertices = np.empty(len(pts_x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = pts_x
    vertices['y'] = pts_y
    vertices['z'] = pts_z
    
    # Assign a visible color mapping
    if color_mode == 'height':
        # Color by voxel center Y (height)
        cy_rep = np.repeat(cy, 8)
        norm_val = (cy_rep - np.min(cy)) / (np.max(cy) - np.min(cy) + 1e-6)
    elif color_mode == 'distance':
        # Color by voxel center Z (distance)
        cz_rep = np.repeat(cz, 8)
        norm_val = (cz_rep - np.min(cz)) / (np.max(cz) - np.min(cz) + 1e-6)
    else:
        norm_val = np.zeros(len(pts_x))
        
    vertices['red'] = (norm_val * 255).astype(np.uint8)
    vertices['green'] = 128
    vertices['blue'] = ((1.0 - norm_val) * 255).astype(np.uint8)
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(filename)

def save_pcd_and_voxel_as_ply(pcd, rgb, voxel_map, map_info, filename, color_mode='height'):
    """
    Exports the original point cloud AND the occupied voxels in voxel_map 
    to a single .ply file for visualization overlay.
    color_mode: 'height' (colors by Y axis) or 'distance' (colors by Z axis)
    """
    from plyfile import PlyData, PlyElement
    
    # 1. Prepare original point cloud
    if rgb is None:
        rgb = np.full((pcd.shape[0], 3), 128, dtype=np.uint8)
    if np.max(rgb) <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
        
    pts_x_pcd = pcd[:, 0]
    pts_y_pcd = pcd[:, 1]
    pts_z_pcd = pcd[:, 2]
    
    # 2. Prepare voxel points (8 points per voxel)
    min_x, min_y, min_z, voxel_size = map_info
    occupied_indices = np.argwhere(voxel_map == 1)
    
    if len(occupied_indices) > 0:
        cx = occupied_indices[:, 0] * voxel_size + min_x + (voxel_size / 2)
        cy = occupied_indices[:, 1] * voxel_size + min_y + (voxel_size / 2)
        cz = occupied_indices[:, 2] * voxel_size + min_z + (voxel_size / 2)
        
        half_v = voxel_size * 0.2
        offsets = np.array([
            [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1]
        ]) * half_v
        
        pts_x_vox = (cx[:, None] + offsets[:, 0]).flatten()
        pts_y_vox = (cy[:, None] + offsets[:, 1]).flatten()
        pts_z_vox = (cz[:, None] + offsets[:, 2]).flatten()
        
        # Color voxels based on height or distance
        if color_mode == 'height':
            cy_rep = np.repeat(cy, 8)
            norm_val = (cy_rep - np.min(cy)) / (np.max(cy) - np.min(cy) + 1e-6)
        elif color_mode == 'distance':
            cz_rep = np.repeat(cz, 8)
            norm_val = (cz_rep - np.min(cz)) / (np.max(cz) - np.min(cz) + 1e-6)
        else:
            norm_val = np.zeros(len(pts_x_vox))
            
        vox_r = (norm_val * 255).astype(np.uint8)
        vox_g = np.full(len(pts_x_vox), 128, dtype=np.uint8)
        vox_b = ((1.0 - norm_val) * 255).astype(np.uint8)
    else:
        pts_x_vox, pts_y_vox, pts_z_vox = [], [], []
        vox_r, vox_g, vox_b = [], [], []
        
    # 3. Combine
    all_x = np.concatenate([pts_x_pcd, pts_x_vox])
    all_y = np.concatenate([pts_y_pcd, pts_y_vox])
    all_z = np.concatenate([pts_z_pcd, pts_z_vox])
    all_r = np.concatenate([rgb[:, 0], vox_r])
    all_g = np.concatenate([rgb[:, 1], vox_g])
    all_b = np.concatenate([rgb[:, 2], vox_b])
    
    vertices = np.empty(len(all_x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = all_x
    vertices['y'] = all_y
    vertices['z'] = all_z
    vertices['red'] = all_r
    vertices['green'] = all_g
    vertices['blue'] = all_b
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(filename)

def get_voxel_points_and_colors(voxel_map, map_info, color_mode='height'):
    min_x, min_y, min_z, voxel_size = map_info
    occupied_indices = np.argwhere(voxel_map == 1)
    
    if len(occupied_indices) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        
    cx = occupied_indices[:, 0] * voxel_size + min_x + (voxel_size / 2)
    cy = occupied_indices[:, 1] * voxel_size + min_y + (voxel_size / 2)
    cz = occupied_indices[:, 2] * voxel_size + min_z + (voxel_size / 2)
    
    half_v = voxel_size * 0.2
    offsets = np.array([
        [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1]
    ]) * half_v
    
    pts_x = (cx[:, None] + offsets[:, 0]).flatten()
    pts_y = (cy[:, None] + offsets[:, 1]).flatten()
    pts_z = (cz[:, None] + offsets[:, 2]).flatten()
    
    if color_mode == 'height':
        cy_rep = np.repeat(cy, 8)
        norm_val = (cy_rep - np.min(cy)) / (np.max(cy) - np.min(cy) + 1e-6)
    elif color_mode == 'distance':
        cz_rep = np.repeat(cz, 8)
        norm_val = (cz_rep - np.min(cz)) / (np.max(cz) - np.min(cz) + 1e-6)
    else:
        norm_val = np.zeros(len(pts_x))
        
    vox_r = (norm_val * 255).astype(np.uint8)
    vox_g = np.full(len(pts_x), 128, dtype=np.uint8)
    vox_b = ((1.0 - norm_val) * 255).astype(np.uint8)
    
    pts = np.stack([pts_x, pts_y, pts_z], axis=1).astype(np.float32)
    colors = np.stack([vox_r, vox_g, vox_b], axis=1).astype(np.uint8)
    return pts, colors

def get_combined_pcd_and_voxel_points_and_colors(pcd, rgb, voxel_map, map_info, color_mode='height'):
    if rgb is None:
        rgb = np.full((pcd.shape[0], 3), 128, dtype=np.uint8)
    if np.max(rgb) <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)
        
    vox_pts, vox_colors = get_voxel_points_and_colors(voxel_map, map_info, color_mode)
    
    if len(vox_pts) > 0:
        all_pts = np.vstack([pcd, vox_pts])
        all_colors = np.vstack([rgb, vox_colors])
    else:
        all_pts = pcd
        all_colors = rgb
        
    return all_pts.astype(np.float32), all_colors.astype(np.uint8)
