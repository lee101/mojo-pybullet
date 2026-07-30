"""Primitive collision detection and rigid-body stepping exposed through a C ABI."""

from std.algorithm import sync_parallelize
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def rot_x(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
          x: Float64, y: Float64, z: Float64) -> Float64:
    return (1.0 - 2.0 * (qy * qy + qz * qz)) * x + \
        2.0 * (qx * qy - qz * qw) * y + 2.0 * (qx * qz + qy * qw) * z


def rot_y(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
          x: Float64, y: Float64, z: Float64) -> Float64:
    return 2.0 * (qx * qy + qz * qw) * x + \
        (1.0 - 2.0 * (qx * qx + qz * qz)) * y + 2.0 * (qy * qz - qx * qw) * z


def rot_z(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
          x: Float64, y: Float64, z: Float64) -> Float64:
    return 2.0 * (qx * qz - qy * qw) * x + 2.0 * (qy * qz + qx * qw) * y + \
        (1.0 - 2.0 * (qx * qx + qy * qy)) * z


def irot_x(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
           x: Float64, y: Float64, z: Float64) -> Float64:
    return (1.0 - 2.0 * (qy * qy + qz * qz)) * x + \
        2.0 * (qx * qy + qz * qw) * y + 2.0 * (qx * qz - qy * qw) * z


def irot_y(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
           x: Float64, y: Float64, z: Float64) -> Float64:
    return 2.0 * (qx * qy - qz * qw) * x + \
        (1.0 - 2.0 * (qx * qx + qz * qz)) * y + 2.0 * (qy * qz + qx * qw) * z


def irot_z(qx: Float64, qy: Float64, qz: Float64, qw: Float64,
           x: Float64, y: Float64, z: Float64) -> Float64:
    return 2.0 * (qx * qz + qy * qw) * x + 2.0 * (qy * qz - qx * qw) * y + \
        (1.0 - 2.0 * (qx * qx + qy * qy)) * z


def closest_sphere_sphere(data: FPtr, pos: FPtr, a: Int, b: Int, result: FPtr):
    var dx = pos[a * 3] - pos[b * 3]
    var dy = pos[a * 3 + 1] - pos[b * 3 + 1]
    var dz = pos[a * 3 + 2] - pos[b * 3 + 2]
    var length = sqrt(dx * dx + dy * dy + dz * dz)
    var nx = 1.0
    var ny = 0.0
    var nz = 0.0
    if length > 1e-15:
        nx = dx / length
        ny = dy / length
        nz = dz / length
    var ra = data[a * 4]
    var rb = data[b * 4]
    result[0] = pos[a * 3] - nx * ra
    result[1] = pos[a * 3 + 1] - ny * ra
    result[2] = pos[a * 3 + 2] - nz * ra
    result[3] = pos[b * 3] + nx * rb
    result[4] = pos[b * 3 + 1] + ny * rb
    result[5] = pos[b * 3 + 2] + nz * rb
    result[6] = nx
    result[7] = ny
    result[8] = nz
    result[9] = length - ra - rb


def sphere_box(data: FPtr, pos: FPtr, orn: FPtr, sphere: Int, box: Int, result: FPtr):
    var qx = orn[box * 4]
    var qy = orn[box * 4 + 1]
    var qz = orn[box * 4 + 2]
    var qw = orn[box * 4 + 3]
    var dx = pos[sphere * 3] - pos[box * 3]
    var dy = pos[sphere * 3 + 1] - pos[box * 3 + 1]
    var dz = pos[sphere * 3 + 2] - pos[box * 3 + 2]
    var lx = irot_x(qx, qy, qz, qw, dx, dy, dz)
    var ly = irot_y(qx, qy, qz, qw, dx, dy, dz)
    var lz = irot_z(qx, qy, qz, qw, dx, dy, dz)
    var margin = 0.001
    var hx = data[box * 4] - margin
    var hy = data[box * 4 + 1] - margin
    var hz = data[box * 4 + 2] - margin
    var cx = min(max(lx, -hx), hx)
    var cy = min(max(ly, -hy), hy)
    var cz = min(max(lz, -hz), hz)
    var nx = lx - cx
    var ny = ly - cy
    var nz = lz - cz
    var length = sqrt(nx * nx + ny * ny + nz * nz)
    var signed_surface = length - margin
    if length > 1e-15:
        nx /= length
        ny /= length
        nz /= length
        cx += nx * margin
        cy += ny * margin
        cz += nz * margin
    else:
        var mx = hx - abs(lx)
        var my = hy - abs(ly)
        var mz = hz - abs(lz)
        nx = 1.0 if lx >= 0.0 else -1.0
        ny = 0.0
        nz = 0.0
        signed_surface = -(mx + margin)
        if my < mx and my <= mz:
            nx = 0.0
            ny = 1.0 if ly >= 0.0 else -1.0
            signed_surface = -(my + margin)
        elif mz < mx and mz < my:
            nx = 0.0
            nz = 1.0 if lz >= 0.0 else -1.0
            signed_surface = -(mz + margin)
        cx = lx - nx * signed_surface
        cy = ly - ny * signed_surface
        cz = lz - nz * signed_surface
    var wnx = rot_x(qx, qy, qz, qw, nx, ny, nz)
    var wny = rot_y(qx, qy, qz, qw, nx, ny, nz)
    var wnz = rot_z(qx, qy, qz, qw, nx, ny, nz)
    var radius = data[sphere * 4]
    result[0] = pos[sphere * 3] - wnx * radius
    result[1] = pos[sphere * 3 + 1] - wny * radius
    result[2] = pos[sphere * 3 + 2] - wnz * radius
    result[3] = pos[box * 3] + rot_x(qx, qy, qz, qw, cx, cy, cz)
    result[4] = pos[box * 3 + 1] + rot_y(qx, qy, qz, qw, cx, cy, cz)
    result[5] = pos[box * 3 + 2] + rot_z(qx, qy, qz, qw, cx, cy, cz)
    result[6] = wnx
    result[7] = wny
    result[8] = wnz
    result[9] = signed_surface - radius


def sphere_plane(data: FPtr, pos: FPtr, orn: FPtr, sphere: Int, plane: Int, result: FPtr):
    var qx = orn[plane * 4]
    var qy = orn[plane * 4 + 1]
    var qz = orn[plane * 4 + 2]
    var qw = orn[plane * 4 + 3]
    var nx = rot_x(qx, qy, qz, qw, data[plane * 4], data[plane * 4 + 1], data[plane * 4 + 2])
    var ny = rot_y(qx, qy, qz, qw, data[plane * 4], data[plane * 4 + 1], data[plane * 4 + 2])
    var nz = rot_z(qx, qy, qz, qw, data[plane * 4], data[plane * 4 + 1], data[plane * 4 + 2])
    var dx = pos[sphere * 3] - pos[plane * 3]
    var dy = pos[sphere * 3 + 1] - pos[plane * 3 + 1]
    var dz = pos[sphere * 3 + 2] - pos[plane * 3 + 2]
    var signed = dx * nx + dy * ny + dz * nz
    if signed < 0.0:
        signed = -signed
        nx = -nx
        ny = -ny
        nz = -nz
    var radius = data[sphere * 4]
    result[0] = pos[sphere * 3] - nx * radius
    result[1] = pos[sphere * 3 + 1] - ny * radius
    result[2] = pos[sphere * 3 + 2] - nz * radius
    result[3] = pos[sphere * 3] - nx * signed
    result[4] = pos[sphere * 3 + 1] - ny * signed
    result[5] = pos[sphere * 3 + 2] - nz * signed
    result[6] = nx
    result[7] = ny
    result[8] = nz
    result[9] = signed - radius


def swap_contact(result: FPtr):
    for k in range(3):
        var tmp = result[k]
        result[k] = result[3 + k]
        result[3 + k] = tmp
        result[6 + k] = -result[6 + k]


def closest(types: IPtr, data: FPtr, pos: FPtr, orn: FPtr,
            a: Int, b: Int, result: FPtr) -> Bool:
    var ta = types[a]
    var tb = types[b]
    if ta == 2 and tb == 2:
        closest_sphere_sphere(data, pos, a, b, result)
        return True
    if ta == 2 and tb == 3:
        sphere_box(data, pos, orn, a, b, result)
        return True
    if ta == 3 and tb == 2:
        sphere_box(data, pos, orn, b, a, result)
        swap_contact(result)
        return True
    if ta == 2 and tb == 6:
        sphere_plane(data, pos, orn, a, b, result)
        return True
    if ta == 6 and tb == 2:
        sphere_plane(data, pos, orn, b, a, result)
        swap_contact(result)
        return True
    return False


@export("mpb_closest_points")
def mpb_closest_points(types_addr: Int, data_addr: Int, pos_addr: Int, orn_addr: Int,
                       a: Int, b: Int, result_addr: Int) abi("C") -> Int:
    return 1 if closest(ip(types_addr), fp(data_addr), fp(pos_addr), fp(orn_addr),
                        a, b, fp(result_addr)) else 0


@export("mpb_ray_test_batch")
def mpb_ray_test_batch(starts_addr: Int, ends_addr: Int, nrays: Int,
                       types_addr: Int, data_addr: Int, pos_addr: Int, orn_addr: Int,
                       ids_addr: Int, active_addr: Int, n: Int, hit_ids_addr: Int,
                       fractions_addr: Int, points_addr: Int, normals_addr: Int) abi("C"):
    var starts = fp(starts_addr)
    var ends = fp(ends_addr)
    var types = ip(types_addr)
    var data = fp(data_addr)
    var pos = fp(pos_addr)
    var orn = fp(orn_addr)
    var ids = ip(ids_addr)
    var active = ip(active_addr)
    var hit_ids = ip(hit_ids_addr)
    var fractions = fp(fractions_addr)
    var points = fp(points_addr)
    var normals = fp(normals_addr)
    for r in range(nrays):
        var sx = starts[r * 3]
        var sy = starts[r * 3 + 1]
        var sz = starts[r * 3 + 2]
        var dx = ends[r * 3] - sx
        var dy = ends[r * 3 + 1] - sy
        var dz = ends[r * 3 + 2] - sz
        var best = 1.0
        var best_id = Int64(-1)
        var bnx = 0.0
        var bny = 0.0
        var bnz = 0.0
        for body in range(n):
            if active[body] == 0:
                continue
            var candidate = 2.0
            var nx = 0.0
            var ny = 0.0
            var nz = 0.0
            var typ = types[body]
            if typ == 2:
                var ox = sx - pos[body * 3]
                var oy = sy - pos[body * 3 + 1]
                var oz = sz - pos[body * 3 + 2]
                var aa = dx * dx + dy * dy + dz * dz
                var bb = ox * dx + oy * dy + oz * dz
                var radius = data[body * 4]
                var cc = ox * ox + oy * oy + oz * oz - radius * radius
                var disc = bb * bb - aa * cc
                if aa > 0.0 and disc >= 0.0:
                    candidate = (-bb - sqrt(disc)) / aa
                    if candidate >= 0.0 and candidate <= best:
                        var hx = sx + candidate * dx - pos[body * 3]
                        var hy = sy + candidate * dy - pos[body * 3 + 1]
                        var hz = sz + candidate * dz - pos[body * 3 + 2]
                        var invr = 1.0 / radius
                        nx = hx * invr
                        ny = hy * invr
                        nz = hz * invr
            elif typ == 3:
                var qx = orn[body * 4]
                var qy = orn[body * 4 + 1]
                var qz = orn[body * 4 + 2]
                var qw = orn[body * 4 + 3]
                var wx = sx - pos[body * 3]
                var wy = sy - pos[body * 3 + 1]
                var wz = sz - pos[body * 3 + 2]
                var lsx = irot_x(qx, qy, qz, qw, wx, wy, wz)
                var lsy = irot_y(qx, qy, qz, qw, wx, wy, wz)
                var lsz = irot_z(qx, qy, qz, qw, wx, wy, wz)
                var ldx = irot_x(qx, qy, qz, qw, dx, dy, dz)
                var ldy = irot_y(qx, qy, qz, qw, dx, dy, dz)
                var ldz = irot_z(qx, qy, qz, qw, dx, dy, dz)
                var near = 0.0
                var far = best
                var axis = -1
                var sign = 0.0
                var valid = True
                for k in range(3):
                    var sv = lsx
                    var dv = ldx
                    if k == 1:
                        sv = lsy
                        dv = ldy
                    elif k == 2:
                        sv = lsz
                        dv = ldz
                    var half = data[body * 4 + k]
                    if abs(dv) < 1e-15:
                        if sv < -half or sv > half:
                            valid = False
                    else:
                        var t1 = (-half - sv) / dv
                        var t2 = (half - sv) / dv
                        var face_sign = -1.0
                        if t1 > t2:
                            var tmp = t1
                            t1 = t2
                            t2 = tmp
                            face_sign = 1.0
                        if t1 > near:
                            near = t1
                            axis = k
                            sign = face_sign
                        far = min(far, t2)
                        if near > far:
                            valid = False
                if valid and axis >= 0 and near >= 0.0 and near <= best:
                    candidate = near
                    var lnx = sign if axis == 0 else 0.0
                    var lny = sign if axis == 1 else 0.0
                    var lnz = sign if axis == 2 else 0.0
                    nx = rot_x(qx, qy, qz, qw, lnx, lny, lnz)
                    ny = rot_y(qx, qy, qz, qw, lnx, lny, lnz)
                    nz = rot_z(qx, qy, qz, qw, lnx, lny, lnz)
            elif typ == 6:
                var qx = orn[body * 4]
                var qy = orn[body * 4 + 1]
                var qz = orn[body * 4 + 2]
                var qw = orn[body * 4 + 3]
                nx = rot_x(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
                ny = rot_y(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
                nz = rot_z(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
                var denom = dx * nx + dy * ny + dz * nz
                if abs(denom) > 1e-15:
                    candidate = -((sx - pos[body * 3]) * nx +
                                   (sy - pos[body * 3 + 1]) * ny +
                                   (sz - pos[body * 3 + 2]) * nz) / denom
                    if denom > 0.0:
                        nx = -nx
                        ny = -ny
                        nz = -nz
            if candidate >= 0.0 and candidate <= best:
                best = candidate
                best_id = ids[body]
                bnx = nx
                bny = ny
                bnz = nz
        hit_ids[r] = best_id
        fractions[r] = best
        if best_id >= 0:
            points[r * 3] = sx + best * dx
            points[r * 3 + 1] = sy + best * dy
            points[r * 3 + 2] = sz + best * dz
            normals[r * 3] = bnx
            normals[r * 3 + 1] = bny
            normals[r * 3 + 2] = bnz
        else:
            for k in range(3):
                points[r * 3 + k] = 0.0
                normals[r * 3 + k] = 0.0


def ray_test_gpu_kernel(
    starts: FPtr, ends: FPtr, nrays: Int, types: IPtr, data: FPtr,
    pos: FPtr, orn: FPtr, ids: IPtr, active: IPtr, n: Int,
    hit_ids: IPtr, fractions: FPtr, points: FPtr, normals: FPtr,
):
    var r = block_idx.x * block_dim.x + thread_idx.x
    if r >= nrays:
        return
    var sx = starts[r * 3]
    var sy = starts[r * 3 + 1]
    var sz = starts[r * 3 + 2]
    var dx = ends[r * 3] - sx
    var dy = ends[r * 3 + 1] - sy
    var dz = ends[r * 3 + 2] - sz
    var best = 1.0
    var best_id = Int64(-1)
    var bnx = 0.0
    var bny = 0.0
    var bnz = 0.0
    for body in range(n):
        if active[body] == 0:
            continue
        var candidate = 2.0
        var nx = 0.0
        var ny = 0.0
        var nz = 0.0
        var typ = types[body]
        if typ == 2:
            var ox = sx - pos[body * 3]
            var oy = sy - pos[body * 3 + 1]
            var oz = sz - pos[body * 3 + 2]
            var aa = dx * dx + dy * dy + dz * dz
            var bb = ox * dx + oy * dy + oz * dz
            var radius = data[body * 4]
            var cc = ox * ox + oy * oy + oz * oz - radius * radius
            var disc = bb * bb - aa * cc
            if aa > 0.0 and disc >= 0.0:
                candidate = (-bb - sqrt(disc)) / aa
                if candidate >= 0.0 and candidate <= best:
                    var hx = sx + candidate * dx - pos[body * 3]
                    var hy = sy + candidate * dy - pos[body * 3 + 1]
                    var hz = sz + candidate * dz - pos[body * 3 + 2]
                    var invr = 1.0 / radius
                    nx = hx * invr
                    ny = hy * invr
                    nz = hz * invr
        elif typ == 3:
            var qx = orn[body * 4]
            var qy = orn[body * 4 + 1]
            var qz = orn[body * 4 + 2]
            var qw = orn[body * 4 + 3]
            var wx = sx - pos[body * 3]
            var wy = sy - pos[body * 3 + 1]
            var wz = sz - pos[body * 3 + 2]
            var lsx = irot_x(qx, qy, qz, qw, wx, wy, wz)
            var lsy = irot_y(qx, qy, qz, qw, wx, wy, wz)
            var lsz = irot_z(qx, qy, qz, qw, wx, wy, wz)
            var ldx = irot_x(qx, qy, qz, qw, dx, dy, dz)
            var ldy = irot_y(qx, qy, qz, qw, dx, dy, dz)
            var ldz = irot_z(qx, qy, qz, qw, dx, dy, dz)
            var near = 0.0
            var far = best
            var axis = -1
            var sign = 0.0
            var valid = True
            for k in range(3):
                var sv = lsx
                var dv = ldx
                if k == 1:
                    sv = lsy
                    dv = ldy
                elif k == 2:
                    sv = lsz
                    dv = ldz
                var half = data[body * 4 + k]
                if abs(dv) < 1e-15:
                    if sv < -half or sv > half:
                        valid = False
                else:
                    var t1 = (-half - sv) / dv
                    var t2 = (half - sv) / dv
                    var face_sign = -1.0
                    if t1 > t2:
                        var tmp = t1
                        t1 = t2
                        t2 = tmp
                        face_sign = 1.0
                    if t1 > near:
                        near = t1
                        axis = k
                        sign = face_sign
                    far = min(far, t2)
                    if near > far:
                        valid = False
            if valid and axis >= 0 and near >= 0.0 and near <= best:
                candidate = near
                var lnx = sign if axis == 0 else 0.0
                var lny = sign if axis == 1 else 0.0
                var lnz = sign if axis == 2 else 0.0
                nx = rot_x(qx, qy, qz, qw, lnx, lny, lnz)
                ny = rot_y(qx, qy, qz, qw, lnx, lny, lnz)
                nz = rot_z(qx, qy, qz, qw, lnx, lny, lnz)
        elif typ == 6:
            var qx = orn[body * 4]
            var qy = orn[body * 4 + 1]
            var qz = orn[body * 4 + 2]
            var qw = orn[body * 4 + 3]
            nx = rot_x(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
            ny = rot_y(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
            nz = rot_z(qx, qy, qz, qw, data[body * 4], data[body * 4 + 1], data[body * 4 + 2])
            var denom = dx * nx + dy * ny + dz * nz
            if abs(denom) > 1e-15:
                candidate = -((sx - pos[body * 3]) * nx +
                               (sy - pos[body * 3 + 1]) * ny +
                               (sz - pos[body * 3 + 2]) * nz) / denom
                if denom > 0.0:
                    nx = -nx
                    ny = -ny
                    nz = -nz
        if candidate >= 0.0 and candidate <= best:
            best = candidate
            best_id = ids[body]
            bnx = nx
            bny = ny
            bnz = nz
    hit_ids[r] = best_id
    fractions[r] = best
    if best_id >= 0:
        points[r * 3] = sx + best * dx
        points[r * 3 + 1] = sy + best * dy
        points[r * 3 + 2] = sz + best * dz
        normals[r * 3] = bnx
        normals[r * 3 + 1] = bny
        normals[r * 3 + 2] = bnz
    else:
        for k in range(3):
            points[r * 3 + k] = 0.0
            normals[r * 3 + k] = 0.0


@export("mpb_ray_test_batch_gpu")
def mpb_ray_test_batch_gpu(starts_addr: Int, ends_addr: Int, nrays: Int,
                           types_addr: Int, data_addr: Int, pos_addr: Int, orn_addr: Int,
                           ids_addr: Int, active_addr: Int, n: Int, hit_ids_addr: Int,
                           fractions_addr: Int, points_addr: Int,
                           normals_addr: Int) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var starts = ctx.enqueue_create_buffer[DType.float64](nrays * 3)
        var ends = ctx.enqueue_create_buffer[DType.float64](nrays * 3)
        var types = ctx.enqueue_create_buffer[DType.int64](n)
        var data = ctx.enqueue_create_buffer[DType.float64](n * 4)
        var pos = ctx.enqueue_create_buffer[DType.float64](n * 3)
        var orn = ctx.enqueue_create_buffer[DType.float64](n * 4)
        var ids = ctx.enqueue_create_buffer[DType.int64](n)
        var active = ctx.enqueue_create_buffer[DType.int64](n)
        var hit_ids = ctx.enqueue_create_buffer[DType.int64](nrays)
        var fractions = ctx.enqueue_create_buffer[DType.float64](nrays)
        var points = ctx.enqueue_create_buffer[DType.float64](nrays * 3)
        var normals = ctx.enqueue_create_buffer[DType.float64](nrays * 3)
        ctx.enqueue_copy(starts, fp(starts_addr))
        ctx.enqueue_copy(ends, fp(ends_addr))
        ctx.enqueue_copy(types, ip(types_addr))
        ctx.enqueue_copy(data, fp(data_addr))
        ctx.enqueue_copy(pos, fp(pos_addr))
        ctx.enqueue_copy(orn, fp(orn_addr))
        ctx.enqueue_copy(ids, ip(ids_addr))
        ctx.enqueue_copy(active, ip(active_addr))
        var block_size = 256
        ctx.enqueue_function[ray_test_gpu_kernel](
            starts, ends, nrays, types, data, pos, orn, ids, active, n,
            hit_ids, fractions, points, normals,
            grid_dim=(nrays + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(ip(hit_ids_addr), hit_ids)
        ctx.enqueue_copy(fp(fractions_addr), fractions)
        ctx.enqueue_copy(fp(points_addr), points)
        ctx.enqueue_copy(fp(normals_addr), normals)
        ctx.synchronize()
        return 1
    except:
        return 0


def resolve_contact(pos: FPtr, vel: FPtr, inv_mass: FPtr, restitution: FPtr,
                    friction: FPtr, a: Int, b: Int, contact: FPtr) -> Bool:
    var penetration = -contact[9]
    if penetration <= 0.0:
        return False
    var ima = inv_mass[a]
    var imb = inv_mass[b]
    var total_inv = ima + imb
    if total_inv == 0.0:
        return False
    var nx = contact[6]
    var ny = contact[7]
    var nz = contact[8]
    var correction = max(penetration - 1e-6, 0.0) / total_inv
    pos[a * 3] += nx * correction * ima
    pos[a * 3 + 1] += ny * correction * ima
    pos[a * 3 + 2] += nz * correction * ima
    pos[b * 3] -= nx * correction * imb
    pos[b * 3 + 1] -= ny * correction * imb
    pos[b * 3 + 2] -= nz * correction * imb
    var rvx = vel[a * 3] - vel[b * 3]
    var rvy = vel[a * 3 + 1] - vel[b * 3 + 1]
    var rvz = vel[a * 3 + 2] - vel[b * 3 + 2]
    var vn = rvx * nx + rvy * ny + rvz * nz
    if vn >= 0.0:
        return True
    var e = min(restitution[a], restitution[b])
    var impulse = -(1.0 + e) * vn / total_inv
    vel[a * 3] += nx * impulse * ima
    vel[a * 3 + 1] += ny * impulse * ima
    vel[a * 3 + 2] += nz * impulse * ima
    vel[b * 3] -= nx * impulse * imb
    vel[b * 3 + 1] -= ny * impulse * imb
    vel[b * 3 + 2] -= nz * impulse * imb
    rvx = vel[a * 3] - vel[b * 3]
    rvy = vel[a * 3 + 1] - vel[b * 3 + 1]
    rvz = vel[a * 3 + 2] - vel[b * 3 + 2]
    var tangent_dot = rvx * nx + rvy * ny + rvz * nz
    var tx = rvx - tangent_dot * nx
    var ty = rvy - tangent_dot * ny
    var tz = rvz - tangent_dot * nz
    var tlen = sqrt(tx * tx + ty * ty + tz * tz)
    if tlen > 1e-15:
        tx /= tlen
        ty /= tlen
        tz /= tlen
        var jt = -(rvx * tx + rvy * ty + rvz * tz) / total_inv
        var mu = sqrt(friction[a] * friction[b])
        jt = min(max(jt, -impulse * mu), impulse * mu)
        vel[a * 3] += tx * jt * ima
        vel[a * 3 + 1] += ty * jt * ima
        vel[a * 3 + 2] += tz * jt * ima
        vel[b * 3] -= tx * jt * imb
        vel[b * 3 + 1] -= ty * jt * imb
        vel[b * 3 + 2] -= tz * jt * imb
    return True


def integrate_body(pos: FPtr, orn: FPtr, vel: FPtr, angvel: FPtr,
                   force: FPtr, torque: FPtr, inv_mass: FPtr,
                   inv_inertia: FPtr, linear_damping: FPtr,
                   angular_damping: FPtr, active: IPtr, i: Int,
                   gx: Float64, gy: Float64, gz: Float64, dt: Float64):
    if active[i] == 0 or inv_mass[i] == 0.0:
        return
    var speed = sqrt(vel[i * 3] * vel[i * 3] + vel[i * 3 + 1] * vel[i * 3 + 1] +
                     vel[i * 3 + 2] * vel[i * 3 + 2])
    var angular_speed = sqrt(angvel[i * 3] * angvel[i * 3] +
                             angvel[i * 3 + 1] * angvel[i * 3 + 1] +
                             angvel[i * 3 + 2] * angvel[i * 3 + 2])
    var ld = max(0.0, 1.0 - linear_damping[i] * dt * (1.0 + speed))
    var ad = max(0.0, 1.0 - angular_damping[i] * dt * (1.0 + angular_speed))
    vel[i * 3] = vel[i * 3] * ld + (gx + force[i * 3] * inv_mass[i]) * dt
    vel[i * 3 + 1] = vel[i * 3 + 1] * ld + (gy + force[i * 3 + 1] * inv_mass[i]) * dt
    vel[i * 3 + 2] = vel[i * 3 + 2] * ld + (gz + force[i * 3 + 2] * inv_mass[i]) * dt
    for k in range(3):
        angvel[i * 3 + k] = angvel[i * 3 + k] * ad + \
            torque[i * 3 + k] * inv_inertia[i * 3 + k] * dt
        pos[i * 3 + k] += vel[i * 3 + k] * dt
    var qx = orn[i * 4]
    var qy = orn[i * 4 + 1]
    var qz = orn[i * 4 + 2]
    var qw = orn[i * 4 + 3]
    var wx = angvel[i * 3]
    var wy = angvel[i * 3 + 1]
    var wz = angvel[i * 3 + 2]
    var halfdt = 0.5 * dt
    orn[i * 4] += halfdt * (wx * qw + wy * qz - wz * qy)
    orn[i * 4 + 1] += halfdt * (-wx * qz + wy * qw + wz * qx)
    orn[i * 4 + 2] += halfdt * (wx * qy - wy * qx + wz * qw)
    orn[i * 4 + 3] += halfdt * (-wx * qx - wy * qy - wz * qz)
    var qnorm = sqrt(orn[i * 4] * orn[i * 4] + orn[i * 4 + 1] * orn[i * 4 + 1] +
                     orn[i * 4 + 2] * orn[i * 4 + 2] + orn[i * 4 + 3] * orn[i * 4 + 3])
    for k in range(4):
        orn[i * 4 + k] /= qnorm


def resolve_pair(types: IPtr, data: FPtr, pos: FPtr, orn: FPtr, vel: FPtr,
                 inv_mass: FPtr, restitution: FPtr, friction: FPtr,
                 a: Int, b: Int, scratch: FPtr) -> Bool:
    if types[a] == 2 and types[b] == 2:
        var dx = pos[a * 3] - pos[b * 3]
        var dy = pos[a * 3 + 1] - pos[b * 3 + 1]
        var dz = pos[a * 3 + 2] - pos[b * 3 + 2]
        var radii = data[a * 4] + data[b * 4]
        if dx * dx + dy * dy + dz * dz >= radii * radii:
            return False
    if closest(types, data, pos, orn, a, b, scratch):
        return resolve_contact(
            pos, vel, inv_mass, restitution, friction, a, b, scratch
        )
    return False


@export("mpb_step")
def mpb_step(types_addr: Int, data_addr: Int, pos_addr: Int, orn_addr: Int,
             vel_addr: Int, angvel_addr: Int, force_addr: Int, torque_addr: Int,
             inv_mass_addr: Int, inv_inertia_addr: Int, restitution_addr: Int,
             friction_addr: Int, linear_damping_addr: Int, angular_damping_addr: Int,
             active_addr: Int, n: Int, gx: Float64, gy: Float64, gz: Float64,
             dt: Float64, iterations: Int, scratch_addr: Int) abi("C"):
    var types = ip(types_addr)
    var data = fp(data_addr)
    var pos = fp(pos_addr)
    var orn = fp(orn_addr)
    var vel = fp(vel_addr)
    var angvel = fp(angvel_addr)
    var force = fp(force_addr)
    var torque = fp(torque_addr)
    var inv_mass = fp(inv_mass_addr)
    var inv_inertia = fp(inv_inertia_addr)
    var restitution = fp(restitution_addr)
    var friction = fp(friction_addr)
    var linear_damping = fp(linear_damping_addr)
    var angular_damping = fp(angular_damping_addr)
    var active = ip(active_addr)
    var scratch = fp(scratch_addr)
    if n >= 1024:
        @parameter
        def integrate_one(i: Int):
            integrate_body(
                pos, orn, vel, angvel, force, torque, inv_mass, inv_inertia,
                linear_damping, angular_damping, active, i, gx, gy, gz, dt
            )

        sync_parallelize[integrate_one](n)
    else:
        for i in range(n):
            integrate_body(
                pos, orn, vel, angvel, force, torque, inv_mass, inv_inertia,
                linear_damping, angular_damping, active, i, gx, gy, gz, dt
            )
    comptime W = simdwidthof[DType.float64]()
    var total = n * 3
    var i = 0
    var zero = SIMD[DType.float64, W](0.0)
    while i + W <= total:
        force.store(i, zero)
        torque.store(i, zero)
        i += W
    while i < total:
        force[i] = 0.0
        torque[i] = 0.0
        i += 1
    for _ in range(iterations):
        var had_contact = False
        for a in range(n):
            if active[a] == 0:
                continue
            for b in range(a + 1, n):
                if active[b] == 0 or (inv_mass[a] == 0.0 and inv_mass[b] == 0.0):
                    continue
                if resolve_pair(
                    types, data, pos, orn, vel, inv_mass, restitution,
                    friction, a, b, scratch
                ):
                    had_contact = True
        if not had_contact:
            break
