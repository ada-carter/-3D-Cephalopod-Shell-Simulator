import numpy as np

def generate_septa(
    centers, tangents, normals, binormals,
    r_arr, Sw_fun, Sh_fun, thetas,
    septa_count, septum_shape, phi_steps,
    septa_indices=None,         # specific septa indices to place
    bend_exponent: float = 0.5   # controls asymptotic septa bending
):
    """
    Generate realistic septa surfaces as domed partitions.
    """
    # use high radial resolution for smoother septa domes
    radial_steps = phi_steps  # one radial division per angular step
    sep_verts = []
    sep_faces = []
    # determine septa positions
    if septa_indices is not None:
        angles = np.array(septa_indices, dtype=int)
    else:
        angles = np.linspace(0, len(thetas)-1, septa_count, endpoint=False, dtype=int)
    for idx in angles:
        center = centers[idx]
        tangent = tangents[idx]
        N = normals[idx]
        B = binormals[idx]
        # cross-section radii at this septum
        Sw_loc = Sw_fun(thetas[idx]) * r_arr[idx]
        Sh_loc = Sh_fun(thetas[idx]) * r_arr[idx]
        # use the smaller semiaxis to prevent septa protruding
        S_limit = min(Sw_loc, Sh_loc)
        # dome height
        H = S_limit * 0.3
        base_idx = len(sep_verts)
        for r_i in range(radial_steps+1):
            alpha = r_i / radial_steps  # 0 at center, 1 at shell wall
            rad = alpha * S_limit  # radial bound by shell cross-section
            # dome profile: asymptotic cap with bend_exponent
            height = H * (max(0, 1 - alpha) ** bend_exponent)
            if septum_shape.lower() == 'anticlastic':
                height = -height
            for p in range(phi_steps):
                ang = 2*np.pi * p / phi_steps
                # compute septal dome vertex within shell wall boundary
                pos = (
                    center +
                    N * (rad * np.cos(ang)) +
                    B * (rad * np.sin(ang)) +
                    tangent * height
                )
                sep_verts.append(tuple(pos))
        # build faces
        for r_i in range(radial_steps):
            for p in range(phi_steps):
                i0 = base_idx + r_i*phi_steps + p
                i1 = base_idx + r_i*phi_steps + (p+1)%phi_steps
                i2 = base_idx + (r_i+1)*phi_steps + p
                i3 = base_idx + (r_i+1)*phi_steps + (p+1)%phi_steps
                sep_faces.append((i0, i2, i1))
                sep_faces.append((i1, i2, i3))
    return sep_verts, sep_faces

def generate_shell(
    W: float = 1.3,      # Whorl expansion rate per revolution
    D: float = 0.5,      # Distance from coiling axis
    T: float = 0.0,      # Translation per revolution
    Sw: float = 0.1,     # Aperture cross-section radius (normal)
    Sh: float = 0.1,     # Aperture cross-section radius (binormal)
    turns: int = 5,      # Number of revolutions
    theta_steps: int = 400,
    phi_steps: int = 30,
    r0: float = 1.0,     # Initial radius
    C: float = 0.0,      # Curvature (Okamoto’s parameter)
    Tau: float = 0.0,    # Torsion (Okamoto’s parameter)
    cross_exp: float = 1.0,     # exponent for cross-section scaling (1=linear scaling)
    septa_count: int = 0,         # number of internal septa
    septum_shape: str = 'synclastic',  # 'synclastic', 'anticlastic', or 'none'
    shell_coil: str = 'none',  # 'involute', 'evolute', or 'none'
    siphuncle_pos: str = 'none',      # 'none', 'central', or 'marginal'
    siphuncle_radius: float = 0.02,   # radius of siphuncle tube
    cross_shape: str = 'circular',     # 'circular', 'elliptical', or 'lobate'
    lobate_freq: int = 0,             # frequency of lobes around aperture
    lobate_amp: float = 0.0,          # amplitude of lobes
    septa_indices: list = None,   # list of septa indices for custom spacing
    bend_exponent: float = 0.5    # controls asymptotic septa bending
):
    """
    Returns (vertices, faces) for a shell mesh.
    C and Tau are accepted but not yet applied to centerline geometry (stub).
    - vertices: (N, 3) array
    - faces:    (M, 3) array of integer indices
    """
    # generate theta array skipping the zero angle to avoid central circle
    full = np.linspace(0, 2 * np.pi * turns, theta_steps + 1)
    thetas = full[1:]  # skip theta=0 ring
    phis = np.linspace(0, 2 * np.pi, phi_steps, endpoint=False)

    # prepare dynamic parameters: float or callable
    W_fun = W if callable(W) else (lambda θ: W)
    D_fun = D if callable(D) else (lambda θ: D)
    # coil type override removed - slider 'Tightness (D)' always controls D
    T_fun = T if callable(T) else (lambda θ: T)
    Sw_fun = Sw if callable(Sw) else (lambda θ: Sw)
    Sh_fun = Sh if callable(Sh) else (lambda θ: Sh)
    # discretize variation: compute r(θ) and cz(θ) via cumulative integration
    dθ = thetas[1] - thetas[0]
    k_arr = np.array([np.log(W_fun(th)) / (2 * np.pi) for th in thetas])
    cum_k = np.cumsum(k_arr * dθ)
    # include initial radius r0 in coil growth
    r_curve = r0 * np.exp(cum_k)        # spiral growth for centerline scaled by r0
    r_arr = r_curve                     # cross-section growth uses same scale
    t_rate = np.array([T_fun(th) / (2 * np.pi) for th in thetas])
    cz_arr = np.cumsum(t_rate * dθ)
    # build centerline points array
    centers = np.vstack([[D_fun(theta) * r_curve[i] * np.cos(theta),
                          D_fun(theta) * r_curve[i] * np.sin(theta),
                          cz_arr[i]] for i, theta in enumerate(thetas)])
    # compute tangents
    dp = np.diff(centers, axis=0)
    tangents = np.vstack([dp, dp[-1:]])
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    # initialize normals via arbitrary perpendicular to first tangent
    n0 = np.cross(tangents[0], [0, 0, 1.0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.cross(tangents[0], [0, 1.0, 0])
    normals = [n0 / np.linalg.norm(n0)]
    binormals = []
    # parallel transport frame
    for i in range(1, len(tangents)):
        v_prev = tangents[i-1]
        v = tangents[i]
        rot_axis = np.cross(v_prev, v)
        if np.linalg.norm(rot_axis) < 1e-6:
            normals.append(normals[-1])
        else:
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.arccos(np.clip(np.dot(v_prev, v), -1.0, 1.0))
            # Rodrigues' rotation for normal
            n_prev = normals[-1]
            normals.append((n_prev * np.cos(angle) +
                            np.cross(rot_axis, n_prev) * np.sin(angle) +
                            rot_axis * np.dot(rot_axis, n_prev) * (1 - np.cos(angle))))
        # binormal
        binormals.append(np.cross(tangents[i], normals[-1]))
    # ensure equal length
    binormals.insert(0, binormals[0])

    # apply curvature and torsion via Frenet-Serret if specified
    C_fun = C if callable(C) else (lambda θ: C)
    Tau_fun = Tau if callable(Tau) else (lambda θ: Tau)
    if any([C_fun(th) != 0 for th in thetas]) or any([Tau_fun(th) != 0 for th in thetas]):
        # preserve default centers for arc-length spacing
        centers_dft = centers.copy()
        ds_arr = np.linalg.norm(np.diff(centers_dft, axis=0), axis=1)
        ds_arr = np.append(ds_arr, ds_arr[-1])
        # initialize Frenet frame
        centers_fs = np.zeros_like(centers_dft)
        centers_fs[0] = centers_dft[0]
        T_prev = tangents[0]
        N_prev = normals[0]
        B_prev = binormals[0]
        tangents_fs = [T_prev]
        normals_fs = [N_prev]
        binormals_fs = [B_prev]
        # integrate
        for i in range(1, len(thetas)):
            ds = ds_arr[i-1]
            k = C_fun(thetas[i])
            tau = Tau_fun(thetas[i])
            # update tangent
            T_i = T_prev + k * N_prev * ds
            T_i /= np.linalg.norm(T_i)
            # update normal
            N_i = N_prev + (-k * T_prev + tau * B_prev) * ds
            N_i /= np.linalg.norm(N_i)
            # update binormal
            B_i = np.cross(T_i, N_i)
            # update position
            centers_fs[i] = centers_fs[i-1] + T_i * ds
            # store
            tangents_fs.append(T_i)
            normals_fs.append(N_i)
            binormals_fs.append(B_i)
            T_prev, N_prev, B_prev = T_i, N_i, B_i
        # replace frames
        centers = centers_fs
        tangents = np.vstack(tangents_fs)
        normals = normals_fs
        binormals = binormals_fs

    # build vertices by sweeping cross-section using normals and binormals, scaling cross-section with shell growth
    verts = []
    for i, theta in enumerate(thetas):
        r_scale = r_arr[i] ** cross_exp
        Sx_base = Sw_fun(theta) * r_scale
        Sy_base = Sh_fun(theta) * r_scale
        N = normals[i]
        B = binormals[i]
        center = centers[i]
        for phi in phis:
            # determine local cross-section radii based on shape
            if cross_shape.lower() == 'elliptical':
                Sx = Sx_base
                Sy = Sy_base * 0.6  # aspect ratio, adjust as needed
            else:
                Sx = Sx_base
                Sy = Sy_base
            # apply lobate modulation
            if cross_shape.lower() == 'lobate' and lobate_freq > 0:
                factor = 1 + lobate_amp * np.cos(lobate_freq * phi)
                Sx *= factor
                Sy *= factor
            offset = N * (Sx * np.cos(phi)) + B * (Sy * np.sin(phi))
            verts.append(tuple(center + offset))

    faces = []
    # skip first ring to avoid central hole
    for i in range(1, theta_steps - 1):
        for j in range(phi_steps):
            i0 = i * phi_steps + j
            i1 = i * phi_steps + (j + 1) % phi_steps
            i2 = (i + 1) * phi_steps + j
            i3 = (i + 1) * phi_steps + (j + 1) % phi_steps
            faces.append((i0, i2, i1))
            faces.append((i1, i2, i3))

    # add septa if requested
    sep_verts_out = None
    sep_faces_out = None
    if septa_count and septum_shape.lower() != 'none':
        # generate local septa geometry
        sep_verts, sep_faces = generate_septa(
            centers, tangents, normals, binormals,
            r_arr, Sw_fun, Sh_fun, thetas,
            septa_count, septum_shape, phi_steps,
            septa_indices,
            bend_exponent
        )
        offset = len(verts)
        verts.extend(sep_verts)
        # compute offset faces and store for separate output
        sep_faces_out = [(f[0]+offset, f[1]+offset, f[2]+offset) for f in sep_faces]
        for f_off in sep_faces_out:
            faces.append(f_off)
        sep_verts_out = sep_verts

    # add siphuncle tube if requested
    if siphuncle_pos.lower() != 'none':
        tube_verts = []
        tube_faces = []
        n_seg = 12
        # build path along centers
        path = centers.copy()
        # offset for marginal siphuncle before truncation
        if siphuncle_pos.lower() == 'marginal':
            path = path + np.array(normals) * (siphuncle_radius * 2)
        # truncate siphuncle at last septum if septa exist
        if septa_count and septum_shape.lower() != 'none':
            angles = np.linspace(0, len(thetas)-1, septa_count, endpoint=False, dtype=int)
            last_idx = int(angles[-1])
            path = path[: last_idx+1]
        # generate ring vertices
        base_index = len(verts)
        for i, pt in enumerate(path):
            N = normals[i]
            B = binormals[i]
            for k in range(n_seg):
                ang = 2 * np.pi * k / n_seg
                off = N * (siphuncle_radius * np.cos(ang)) + B * (siphuncle_radius * np.sin(ang))
                tube_verts.append(tuple(pt + off))
        # connect rings
        for i in range(len(path)-1):
            for k in range(n_seg):
                i0 = base_index + i*n_seg + k
                i1 = base_index + i*n_seg + (k+1)%n_seg
                i2 = base_index + (i+1)*n_seg + k
                i3 = base_index + (i+1)*n_seg + (k+1)%n_seg
                tube_faces.append((i0, i2, i1))
                tube_faces.append((i1, i2, i3))
        # append
        verts.extend(tube_verts)
        faces.extend(tube_faces)

    return np.array(verts, dtype=float), np.array(faces, dtype=int)
