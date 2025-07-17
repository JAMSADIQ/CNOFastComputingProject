from phi.torch.flow import *

from tqdm.notebook import trange
import matplotlib.pyplot as plt


##### Variables ############################
len_x = 10 # domain length
len_y = 10 # domain width
len_z = 10 # domain height

res_x = 20 # resolution grid x
res_y = 20 # resolution grid y
res_z = 20 # resolution grid z

N = 2000 # number of time steps
dt = 1e-2 # time step size

obstacles_ = {
    'obs0': [(9.29, 9.5), (3.93, 9.01), (7.94, 9.5)],
    'obs1': [(0.92, 2.73), (5.99, 9.5), (2.51, 5.27)],
}

ac_ = {
    'ac0': [(4.37, 4.91), (4.31, 5.07), (9.32, 9.5)],
}

ac_inflow = [
    (0.0, 0.0, -0.51),
]

t_ambient = 30
t_room = 25
t_ac = 16

viscosity = 1e-2
conductivity = 1e-3


############################################

def field_to_vtk(fields, filename):
    """
    Save a field to a VTK file.
    """
    import pyvista as pv


    x = np.linspace(0, len_x, res_x)
    y = np.linspace(0, len_y, res_y)
    z = np.linspace(0, len_y, res_z)
    X, Y, Z = np.meshgrid(x, y, z)

    grid = pv.StructuredGrid(X, Y, Z)
    for k, v in fields.items():
        resample = v.at(CenteredGrid(0, None, domain, x=res_x, y=res_y, z=res_z)).numpy()
        print(resample.shape)
        if k == 'v':
            grid.point_data['v'] = np.stack([
                resample[:, :, :, 0].flatten(),
                resample[:, :, :, 1].flatten(),
                resample[:, :, :, 2].flatten()
            ], axis=-1)

        elif k == 'p':
            grid.point_data['p'] = resample.flatten()
        elif k == 'T':
            grid.point_data['T'] = resample.flatten()

    grid.save(filename)




domain = Box(x=len_x, y=len_y, z=len_z)

acs = [ 
    Box(x=ac_[name][0], y=ac_[name][1], z=ac_[name][2])
    for name in ac_
]

obstacles = [
    Box(x=obstacles_[name][0], y=obstacles_[name][1], z=obstacles_[name][2])
    for name in obstacles_
]

from phi.geom import Box, Sphere, union, intersection


from phi.geom import  Box

boundary = {'x': 0, 'y': 0, 'z': 0}
v0 = StaggeredGrid(0, boundary, domain, x=res_x//2, y=res_y//2, z=res_z//2)
boundary_t= {'x': t_ambient, 'y': t_ambient, 'z': t_ambient}
t0 = CenteredGrid(t_room, boundary_t, domain, x=res_x, y=res_y, z=res_z)


vent_geom   = acs[0]

vent_mask = resample(acs[0], to=t0, soft=True)

staggered_jet = StaggeredGrid(
    ac_inflow[0],
    0.,
    domain,
    x=res_x,
    y=res_y,
    z=res_z
)

#@jit_compile
def step(v, p, t, dt=1., ii=0):


    print(f'Step {ii} with dt={dt}')

    t = advect.semi_lagrangian(t, v, dt)
    t = diffuse.explicit(t, conductivity, dt)

    # t = t * (1 - inflow_rate*dt) + 16.0 * (inflow_rate*dt * vent_mask)
    # t = t + (16.0 - t) * (inflow_rate * dt) * vent_mask
    t = vent_mask * t_ac + (1 - vent_mask) * t
    t = t.with_boundary(t_ambient)



    v = v + resample(staggered_jet, to=v) * dt * vent_mask
    v = v.with_boundary(0.0)

    v = advect.semi_lagrangian(v, v, dt)
    v = diffuse.explicit(v, viscosity, dt)

    solver = Solve('CG', 1e-3, x0=p) if (p is not None) else Solve('CG', 1e-3)
    v, p = fluid.make_incompressible(v, obstacles, solver)


    # plot(v[{'z': 8, 'vector': 'x'}])
    # plt.savefig(f'{ii}vx.jpg')
    # plot(v[{'z': 8, 'vector': 'y'}])
    # plt.savefig(f'{ii}vy.jpg')
    # plot(t[{'z': 18, 'vector': 'x,y'}])
    # plt.savefig(f'{ii}t.jpg')

    return v, p, t







    # # masks_v = [resample(ac, to=v) for ac in acs]
    masks_v = [
        CenteredGrid(0, 0, domain, x=res_x, y=res_y, z=res_z)
        for ac in acs
    ]
    # # print('masks_v', masks_v[0].shape)
    masks_v = [
        resample(ac, to=mask, soft=True)
        for mask, ac in zip(masks_v, acs)
    ]
    # # print('masks_v', masks_v[0].shape)
    # stacked = math.stack(masks_v, channel('mask'))
    # # print('stacked', stacked)
    # noac_mask = math.sum(stacked, channel('mask'))
    # # print(noac_mask)
    # # print(type(noac_mask))
    # # print('noac_mask', noac_mask.shape)

    # tmp = resample(noac_mask, to=v)
    # # noac_mask = 1. - union(*masks_v)
    # # hhhhhh
    # # for ac, flow in zip(acs, ac_inflow):
    # #     print(ac, flow)
    # #     v += flow * resample(ac, to=v, soft=True) * dt 
    ac_in = advect.mac_cormack(masks_v[0], v, dt) + ac_inflow[0] * masks_v[0]
    plot(ac_in[{'z': 27, 'vector': 'x,y'}])
    plt.savefig(f'{ii}ac_in.jpg')
    forcing_acs = resample(ac_in, to=v)
    v = advect.semi_lagrangian(v, v, dt) + forcing_acs * dt
    
    # v = StaggeredGrid(
    #     v.values, 
    #     v.extrapolation,  # keep the same boundary conditions
    #     v.bounds,         # same physical domain
    #     v.resolution      # same resolution
    # )
    plot(v[{'z': 27, 'vector': 'x,y'}])
    plt.savefig(f'{ii}dddd.jpg')
    
    print(type(v), v.shape)
    v, p = fluid.make_incompressible(v, obstacles, Solve('CG', 1e-3, x0=p))
    print(type(v), v.shape)
    print(v.values)
    print(p)

    # print('ac_in', ac_in)
    # gggg

    # v = v * tmp
    # for mask, flow, in zip(masks_v, ac_inflow):
    #     v = v + flow * resample(mask, to=v)
    # v.with_boundary(0.0)
    # print(v)
    # v = StaggeredGrid(v, 
    #                  v.extrapolation,  # keep the same boundary conditions
    #                  v.bounds,         # same physical domain
    #                  v.resolution) 
    # # plot(v[{'z': 28, 'vector': 'y'}])
    # # plot(v[{'z': 20, 'vector': 'x,y'}])
    # # plt.savefig('noac_mask.jpg')
    # # hhh
    # # plt.savefig('mask_v.jpg')
    # # hyhyhyh
    # # print(v * tmp)
    # # print('v', v.shape)
    # # print('masks_v', masks_v[0].shape)
    # # print('noac_mask', noac_mask.shape)

    # # v = v * noac_mask
    # # plot(v[{'z': 28, 'vector': 'x,y'}])
    # # plt.savefig('v0.jpg')
    # # ggg
    # # plot(v[{'z': 28, 'vector': 'y'}])
    # # plt.savefig('tt0.jpg')

    # # adv = advect.semi_lagrangian(v, v, dt)
    # # v = diffuse.explicit(v, viscosity, dt) 
    # adv = advect.finite_difference(v, v, order=2)
    # diff = diffuse.finite_difference(v, viscosity, order=2)
    # v = adv + diff

    # #v, p = fluid.make_incompressible(v, obstacles, Solve('CG', 1e-3, x0=p))
    # # v, p = fluid.make_incompressible(v, obstacles, CUDASolver())
    # v, p = fluid.make_incompressible(v, (), Solve('CG', rank_deficiency=0))

    # plot(v[{'z': 27, 'vector': 'x,y'}])
    # plt.savefig(f'v{ii}.jpg')

    # plot(v['x,y'][{'z': 27}])
    # mask_t = resample(acs[0], to=t)
    # for ac in acs[1:]:
    #     mask_t += resample(ac, to=t)

    # t = t * (1-mask_t)  + t_ac * mask_t # remove temperature in ac areas
    # t = t.with_boundary(30.0)

    # # plot(t[{'z': 27}])
    # # plt.savefig('tt1.jpg')
    # # for ac in acs:
    # #     t += t_ac
    # t = advect.semi_lagrangian(t, v, dt)
    # # plot(t[{'z': 27}])
    # # plt.savefig('tt2.jpg')
    # # gggg
    # t = diffuse.explicit(t, conductivity, dt) 
    return v, p, t


p, v, t = None, v0, t0
field_to_vtk({'v': v, 'T': t}, 'solution_0000.vtk')
for time in range(1, N+1):
    v1, p1, t1 = step(v, p, t, dt=dt, ii=time)
    print(f'Time {time}')

    if time % 10 == 0:
        field_to_vtk({'v': v1, 'p': p1, 'T': t1}, f'solution_{time:04d}.vtk')

    v, p, t = v1, p1, t1

print('Simulation complete.')

# print('v', v)
# print('p', p)
# print('t', t)

# plot(v[{'z': 27, 'vector': 'x,y'}])
# plt.savefig('v.jpg')

# plot(t[{'z': 27}])
# plt.savefig('t.jpg')

# hhh

# # v_sol_xy = v_sol[{'z': 2, 'vector': 'x, y'}]
# # v_sol_xz = v_sol[{'y': 2, 'vector': 'x, z'}]
# # v_sol_yz = v_sol[{'x': 1, 'vector': 'y, z'}]

# # plot(t_sol, animate='time', overlay='args')
# # plt.savefig('noh.jpg')
# plot(v_sol, *obstacles, *acs, animate='time', overlay='args')
# plt.show()

