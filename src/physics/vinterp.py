"""Vertical-interpolation
"""

from __future__ import annotations

import ctypes as ct
import os
from collections import namedtuple

import numpy as np
import xarray as xr

# -----------------------------------------------------------------------------
# Fortran library interface (from fvertintp_iface.py)
# -----------------------------------------------------------------------------
_F_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f_src", "fvertintp.so")
f_lib = ct.CDLL(_F_LIB_PATH)

c_real = ct.c_double
c_real_ptr = np.ctypeslib.ndpointer(c_real)
c_int = ct.c_int32
c_int_ptr = np.ctypeslib.ndpointer(c_int)

_REAL_DTYPE = np.float64
_INT_DTYPE = np.int32

f_lib.interp.argtypes = [c_real_ptr] * 2 + [c_int_ptr, c_real_ptr] + [c_int] * 6
f_lib.interp_fld.argtypes = [c_real_ptr] * 2 + [c_int_ptr, c_real_ptr] + [c_int] * 6


def _as_c_contig(arr, dtype):
    """Return a C-contiguous ndarray with the requested dtype.

    Uses a single conversion step to avoid the extra temporary created by
    calling ``ascontiguousarray`` and ``astype`` separately.
    """
    return np.asarray(arr, dtype=dtype, order="C")


def interp(psrc, ptgt, chunk_size_max=1000):
    """Interpolate ``psrc`` to ``ptgt``.

    psrc(ncom, nsrc, nlevsrc)
    ptgt(ncom, ntgt, nlevtgt)

    returns
    tgtlevs(ncom, nsrc, ntgt, nlevtgt), weights(ncom, nsrc, ntgt, nlevtgt)
    """
    ncom, ntgt, nlevtgt = ptgt.shape
    ncom2, nsrc, nlevsrc = psrc.shape

    if ncom != ncom2:
        raise ValueError(
            "Different common dimension size between source and target!"
            + f"({ncom2},{nsrc}) and ({ncom},{nsrc})"
        )

    outshape = (ncom, nsrc, ntgt, nlevtgt)

    tgtlevs = np.zeros(outshape, dtype=_INT_DTYPE)
    weights = np.zeros(outshape, dtype=_REAL_DTYPE)

    psrc_cont = _as_c_contig(psrc, _REAL_DTYPE)
    ptgt_cont = _as_c_contig(ptgt, _REAL_DTYPE)

    f_lib.interp(
        psrc_cont,
        ptgt_cont,
        tgtlevs,
        weights,
        c_int(ncom),
        c_int(nsrc),
        c_int(ntgt),
        c_int(nlevsrc),
        c_int(nlevtgt),
        c_int(chunk_size_max),
    )
    del psrc_cont, ptgt_cont

    return tgtlevs, weights


def interp_fld(fsrc, tgtlevs, weights, chunk_size_max=1000):
    """Interpolate fields according to weights.

    fsrc(ncom, nsrc, nlevsrc)
    tgtlevs(ncom, ntgt, nlevtgt)
    weights(ncom, ntgt, nlevtgt)

    returns
    fdst(ncom, nsrc, ntgt, nlevtgt)
    """
    ncom, nsrc, nlevsrc = fsrc.shape

    ncom2, ntgt, nlevtgt = tgtlevs.shape
    ncom3, ntgt2, nlevtgt2 = weights.shape

    if (ncom != ncom2) or (ncom2 != ncom3) or (ntgt != ntgt2) or (nlevtgt != nlevtgt2):
        raise ValueError(
            "Some dimensions are incompatible!!"
            + f"fsrc(ncom={ncom},nsrc={nsrc},nlevsrc{nlevsrc})"
            + f"tgtlevs(ncom={ncom2},ntgt={ntgt},nlevtgt{nlevtgt})"
            + f"tgtlevs(ncom={ncom3},ntgt={ntgt2},nlevtgt{nlevtgt2})"
        )

    fdstshape = (ncom, nsrc, ntgt, nlevtgt)
    fdst = np.zeros(fdstshape, dtype=_REAL_DTYPE)

    fsrc_cont = _as_c_contig(fsrc, _REAL_DTYPE)
    tgtlevs_cont = _as_c_contig(tgtlevs, _INT_DTYPE)
    weights_cont = _as_c_contig(weights, _REAL_DTYPE)

    f_lib.interp_fld(
        fsrc_cont,
        fdst,
        tgtlevs_cont,
        weights_cont,
        c_int(ncom),
        c_int(nsrc),
        c_int(ntgt),
        c_int(nlevsrc),
        c_int(nlevtgt),
        c_int(chunk_size_max),
    )
    del fsrc_cont, tgtlevs_cont, weights_cont

    return fdst


# -----------------------------------------------------------------------------
# xarray stacking helpers (from stack_tools.py)
# -----------------------------------------------------------------------------
StackTools = namedtuple(
    "StackTools",
    [
        "src_dim_order",
        "dst_dim_order",
        "out_dim_order",
        "src_stackshape",
        "dst_stackshape",
        "out_coords",
        "out_shape",
    ],
)


def tools_to_stack_xarrays(src_arr, dst_arr, intp_dim_name):
    """Return all tools to reorder and reshape arrays using numpy."""
    nonintp_dims_src = [d for d in src_arr.dims if d != intp_dim_name]
    nonintp_dims_dst = [d for d in dst_arr.dims if d != intp_dim_name]

    common_dim_names = list(set(nonintp_dims_src) & set(nonintp_dims_dst))
    unique_dim_names = list(set(nonintp_dims_src) ^ set(nonintp_dims_dst))
    onlysrc_dim_names = [d for d in nonintp_dims_src if d in unique_dim_names]
    onlydst_dim_names = [d for d in nonintp_dims_dst if d in unique_dim_names]

    src_dim_order = common_dim_names + onlysrc_dim_names
    dst_dim_order = common_dim_names + onlydst_dim_names
    if intp_dim_name:
        src_dim_order = src_dim_order + [intp_dim_name]
        dst_dim_order = dst_dim_order + [intp_dim_name]

    com_ndims = [len(src_arr[com_dim]) for com_dim in common_dim_names]
    src_ndims = [len(src_arr[src_dim]) for src_dim in onlysrc_dim_names]
    src_intp_ndims = [len(src_arr[intp_dim_name])] if intp_dim_name in src_arr.dims else []

    dst_ndims = [len(dst_arr[dst_dim]) for dst_dim in onlydst_dim_names]
    dst_intp_ndims = [len(dst_arr[intp_dim_name])] if intp_dim_name in dst_arr.dims else []

    src_stackshape = tuple(
        int(np.array(ndims).prod()) if len(ndims) > 0 else 1
        for ndims in [com_ndims, src_ndims, src_intp_ndims]
    )
    dst_stackshape = tuple(
        int(np.array(ndims).prod()) if len(ndims) > 0 else 1
        for ndims in [com_ndims, dst_ndims, dst_intp_ndims]
    )

    out_shape = tuple(com_ndims + src_ndims + dst_ndims + dst_intp_ndims)
    if len(out_shape) == 0:
        out_shape = None
        out_dim_order = None
    else:
        out_dim_order = common_dim_names + onlysrc_dim_names + onlydst_dim_names
        if intp_dim_name:
            out_dim_order = out_dim_order + [intp_dim_name]
    out_coords = {
        **{com_dim: src_arr.coords[com_dim] for com_dim in common_dim_names},
        **{src_dim: src_arr.coords[src_dim] for src_dim in onlysrc_dim_names},
        **{dst_dim: dst_arr.coords[dst_dim] for dst_dim in onlydst_dim_names},
    }
    if intp_dim_name:
        out_coords[intp_dim_name] = dst_arr.coords[intp_dim_name]

    arglist = [src_dim_order, dst_dim_order, out_dim_order] + [
        src_stackshape,
        dst_stackshape,
        out_coords,
        out_shape,
    ]

    return StackTools(*arglist)


def interp_xarray(
    p_src: xr.DataArray,
    p_tgt: xr.DataArray,
    intp_dim_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Call the Fortran ``interp`` routine and return results as xarray DataArrays.

    Parameters
    ----------
    p_src : xr.DataArray
        Source pressure coordinate, ascending along ``intp_dim_name``.
    p_tgt : xr.DataArray
        Target pressure values along ``intp_dim_name``.
    intp_dim_name : str
        Name of the interpolation dimension in both arrays.

    Returns
    -------
    tgtlevs : xr.DataArray (int)
        Lower-bracket source indices for each target level.
    weights : xr.DataArray (float)
        Linear interpolation weight towards the upper bracket (index + 1).

    The field interpolation can then be done in pure xarray::

        lower = field_src.isel({intp_dim_name: tgtlevs})
        upper = field_src.isel({intp_dim_name: tgtlevs + 1})
        result = (1 - weights) * lower + weights * upper
    """
    st = tools_to_stack_xarrays(src_arr=p_src, dst_arr=p_tgt, intp_dim_name=intp_dim_name)
    tgtlevs_np, weights_np = interp(
        psrc=p_src.transpose(*st.src_dim_order).values.reshape(st.src_stackshape),
        ptgt=p_tgt.transpose(*st.dst_dim_order).values.reshape(st.dst_stackshape),
    )
    tgtlevs = xr.DataArray(
        tgtlevs_np.reshape(st.out_shape),
        dims=st.out_dim_order,
        coords=st.out_coords,
    )
    weights = xr.DataArray(
        weights_np.reshape(st.out_shape),
        dims=st.out_dim_order,
        coords=st.out_coords,
    )
    return tgtlevs, weights


def interp_fld_xa(
    field_src: xr.DataArray,
    tgtlevs: xr.DataArray,
    weights: xr.DataArray,
    intp_dim_name: str,
) -> xr.DataArray:
    """Interpolate ``field_src`` using pre-computed indices and weights.

    Uses xarray ``isel`` to load only the two bracket slices for each target
    level rather than the full source field, reducing peak memory at the cost
    of some compute overhead vs the Fortran ``interp_fld``.

    Parameters
    ----------
    field_src : xr.DataArray
        Source field with dimension ``intp_dim_name``.
    tgtlevs : xr.DataArray
        Lower-bracket integer indices (from ``interp_xarray``).
    weights : xr.DataArray
        Interpolation weights towards the upper bracket (from ``interp_xarray``).
    intp_dim_name : str
        Name of the interpolation dimension in ``field_src``.

    Returns
    -------
    xr.DataArray
        Interpolated field on the target levels.
    """
    # tgtlevs from Fortran interp are 1-based; subtract 1 for 0-based isel.
    lower = field_src.isel({intp_dim_name: tgtlevs - 1})
    upper = field_src.isel({intp_dim_name: tgtlevs})
    return (1.0 - weights) * lower + weights * upper
