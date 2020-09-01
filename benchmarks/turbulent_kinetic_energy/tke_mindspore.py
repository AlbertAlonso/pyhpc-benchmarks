import mindspore
import mindspore.ops.functional as F
import mindspore.ops.operations as P

maximum = P.Maximum()

def integrate_tke(u, v, w, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, kbot, kappaM, mxl, forc, forc_tke_surface, tke, dtke):
    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1
    dt_mom = 1
    AB_eps = 0.1
    alpha_tke = 1.
    c_eps = 0.7
    K_h_tke = 2000.

    flux_east = F.zeros_like(maskU)
    flux_north = F.zeros_like(maskU)
    flux_top = F.zeros_like(maskU)

    sqrttke = F.sqrt(maximum(0., tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot[2:-2, 2:-2] - 1

    a_tri = F.zeros_like(maskU[2:-2, 2:-2])
    b_tri = F.zeros_like(maskU[2:-2, 2:-2])
    c_tri = F.zeros_like(maskU[2:-2, 2:-2])
    d_tri = F.zeros_like(maskU[2:-2, 2:-2])
    delta = F.zeros_like(maskU[2:-2, 2:-2])