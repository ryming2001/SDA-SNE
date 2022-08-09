import struct
import cv2
import numpy as np

## Parameter Selection and Initialization
# DATASET = {'Easy/android'     ,'Easy/cube'   ,'Easy/fish'     ,Easy/goldfish  ,'Easy/polyhedron' ,'Easy/teeth'     ,'Easy/torus'      ,'Easy/torusknot',
#            'Medium/airballoon','Medium/bird' ,'Medium/bunny'  ,'Medium/chair' ,'Medium/fairy'    ,'Medium/penguin' ,'Medium/sealion'  ,'Medium/Trex',
#            'Hard/batman'      ,'Hard/knight' ,'Hard/pineapple','Hard/plant'   ,'Hard/priest'     ,'Hard/room'      ,'Hard/temple'     ,'Hard/tree'}
DATASET = 'test_data/Easy/torusknot'
SNE  = 'SDA'   # SNE  = {'SDA': SNE introduced by us
               #         'OTW' : other SNEs}
BASE = 'CP2TV' # BASE = {'CP2TV','3F2N'}
MAX_ITER = 3   # MAX_ITER = {3,inf} : maximum iterations
UMAX = 640
VMAX = 480
PIC_NUM = 1    # serial number of an image



def get_dataset_params(rootPath):
    with open(rootPath + '/params.txt', 'r') as f:
        data = f.read()
        params = list(map(int, (data.split())))
    return params


def get_normal_gt(rootPath):
    # retVal: [-1,1]
    retVal = cv2.imread(rootPath + '/normal/{:0>6d}.png'.format(PIC_NUM), -1) / 65535 * 2 - 1
    retVal = -retVal[:, :, ::-1]
    return retVal


def get_depth(rootPath):
    with open(rootPath + '/depth/{:0>6d}.bin'.format(PIC_NUM), 'rb') as f:
        data_raw = struct.unpack('f' * UMAX * VMAX, f.read(4 * UMAX * VMAX))
        z = np.array(data_raw).reshape(VMAX, UMAX)
    x = u_map * z / cam_fx
    y = v_map * z / cam_fy
    # create mask
    mask = np.zeros((VMAX, UMAX))
    mask[z == 1] = 1

    return x, y, z, mask


def vector_normalization(normal):
    mag = np.linalg.norm(normal, axis=2)
    normal /= mag.reshape((VMAX,UMAX,1))
    return normal



def visualization_map_creation(nx, ny, nz, mask):
    # mask: 1 is background, 0 is obj
    # make background white, [1, 1, 1]
    n_vis = np.zeros((VMAX, UMAX, 3))
    n_vis[:, :, 0] = nx * (1 - mask) - mask
    n_vis[:, :, 1] = ny * (1 - mask) - mask
    n_vis[:, :, 2] = nz * (1 - mask) - mask
    n_vis = (1 - n_vis) / 2  # [-1, 1]区间变换到[0, 1]区间
    return n_vis


def angle_normalization(err_map):
    err_map[err_map > np.pi / 2] = np.pi - err_map[err_map > np.pi / 2]
    return err_map


def evaluation(n_gt, n_t, mask):
    scale = np.pi / 180
    errorMap = np.arccos(np.sum(n_gt*n_t, axis=2))
    errorMap = angle_normalization(errorMap) / scale
    errorMap[mask == 1] = 0
    ea = errorMap.sum() / (VMAX * UMAX - mask.sum())
    return errorMap, ea

# SDA
def SDA(Z, cam_fx, cam_fy, u_map, v_map):
    # kernel setting
    Gx = np.array([[0, 0, 0],
                      [-0.5, 0, 0.5],
                      [0, 0, 0]])
    GxL = np.array([[0, 0, 0],
                      [-1, 1, 0],
                      [0, 0, 0]])
    GxR = np.array([[0, 0, 0],
                      [0, -1, 1],
                      [0, 0, 0]])
    Gy = np.array([[0, -0.5, 0],
                      [0, 0, 0],
                      [0, 0.5, 0]])
    GyU = np.array([[0, -1, 0],
                      [0, 1, 0],
                      [0, 0, 0]])
    GyD = np.array([[0, 0, 0],
                      [0, -1, 0],
                      [0, 1, 0]])
    u_laplace =np.array([[-1, 2, -1]])
    v_laplace =np.array([[-1], [2], [-1]])
    
    # Initialization
    Z_ulaplace_0 = cv2.filter2D(Z,-1, u_laplace)
    Z_vlaplace_0 = cv2.filter2D(Z,-1, v_laplace)
    Z_ulaplace=abs(Z_ulaplace_0)
    Z_vlaplace=abs(Z_vlaplace_0)
    
    MATRIX_ONE = np.ones((VMAX, UMAX))
    H_ONE = np.ones((1, UMAX))
    V_ONE = np.ones((VMAX, 1))
    
    con_L = MATRIX_ONE-np.sign(np.hstack((V_ONE, Z_ulaplace_0[:, :-1]))*np.hstack((np.ones((VMAX, 2)), Z_ulaplace_0[:, :-2])))
    con_R = MATRIX_ONE-np.sign(np.hstack((Z_ulaplace_0[:, 1:], V_ONE))*np.hstack((Z_ulaplace_0[:, 2:], np.ones((VMAX, 2)))))
    con_U = MATRIX_ONE-np.sign(np.vstack((H_ONE, Z_vlaplace_0[:-1, :]))*np.vstack((np.ones((2, UMAX)), Z_vlaplace_0[:-2, :])))
    con_D = MATRIX_ONE-np.sign(np.vstack((Z_vlaplace_0[1:, :], H_ONE))*np.vstack((Z_vlaplace_0[2:, :], np.ones((2, UMAX)))))
   
    
    Gu_stack = np.array((
                        cv2.filter2D(Z,-1, Gx),
                        cv2.filter2D(Z,-1, GxR),
                        cv2.filter2D(Z,-1, GxL)
                        )).reshape(3,-1)
    Gv_stack = np.array((
                        cv2.filter2D(Z,-1, Gy),
                        cv2.filter2D(Z,-1, GyD),
                        cv2.filter2D(Z,-1, GyU)
                        )).reshape(3,-1)           
    Eu_stack = np.array((
                            Z_ulaplace,
                            np.hstack((Z_ulaplace[:, 1:], np.inf*V_ONE)),
                            np.hstack((np.inf*V_ONE, Z_ulaplace[:, :-1]))
                            ))
    Ev_stack = np.array((
                            Z_vlaplace,
                            np.vstack((Z_vlaplace[1:, :], np.inf*H_ONE)),
                            np.vstack((np.inf*H_ONE, Z_vlaplace[:-1, :]))
                            ))
    # smoothness energy initialization
    Eu = np.min(Eu_stack, axis=0)
    Ev = np.min(Ev_stack, axis=0)
    # state variable initialization
    umin_index = np.argmin(Eu_stack, axis=0)
    vmin_index = np.argmin(Ev_stack, axis=0)
    # depth gradient initialization
    Gu = Gu_stack[umin_index.reshape(-1), np.arange(VMAX*UMAX)].reshape(VMAX, UMAX)
    Gv = Gv_stack[vmin_index.reshape(-1), np.arange(VMAX*UMAX)].reshape(VMAX, UMAX)
    umin_index = 2 - umin_index
    vmin_index = 2 - vmin_index
    
    MAX = 1e10
    for i in range(1, MAX_ITER+1):
        # corresponding state variable:{[-1, 0],
        #                               [1, 0],
        #                               [0, 0],
        #                               [0, -1],
        #                               [0, 1],
        #                               [-1, -1],
        #                               [1, -1],
        #                               [-1, 1],
        #                               [1, 1]}
        Eu_stack = np.array((
                              np.hstack((np.inf*V_ONE, Z_ulaplace[:, :-1]))+(abs(umin_index-0*MATRIX_ONE)+abs(np.hstack((np.inf*V_ONE, umin_index[:, :-1]))-0*MATRIX_ONE)+con_L)*MAX,       
                              np.hstack((Z_ulaplace[:, 1:], np.inf*V_ONE))+(abs(umin_index-MATRIX_ONE)+abs(np.hstack((umin_index[:, 1:], np.inf*V_ONE))-MATRIX_ONE)+con_R)*MAX,
                              Eu,
                              2*(np.vstack((np.inf*H_ONE, Z_vlaplace[:-1, :]))+np.vstack((np.inf*H_ONE, Eu[:-1, :]))),   
                              2*(np.vstack((Z_vlaplace[1:, :],np.inf*H_ONE))+np.vstack((Eu[1:, :],np.inf*H_ONE))),
                              np.vstack((np.inf*H_ONE, Z_vlaplace[:-1, :]))+np.vstack((np.inf*H_ONE, Eu[:-1, :]))+np.hstack((np.inf*V_ONE, np.vstack((np.inf*H_ONE, Z_ulaplace[:-1, :]))[:, :-1]))+np.hstack((np.inf*V_ONE, np.vstack((np.inf*H_ONE, Ev[:-1, :]))[:, :-1])),   
                              np.vstack((np.inf*H_ONE, Z_vlaplace[:-1, :]))+np.vstack((np.inf*H_ONE, Eu[:-1, :]))+np.hstack((np.vstack((np.inf*H_ONE, Z_ulaplace[:-1, :]))[:, 1:],np.inf*V_ONE))+np.hstack((np.vstack((np.inf*H_ONE, Ev[:-1, :]))[:, 1:],np.inf*V_ONE)),        
                              np.vstack((Z_vlaplace[1:, :],np.inf*H_ONE))+np.vstack((Eu[1:, :],np.inf*H_ONE))+np.hstack((np.inf*V_ONE, np.vstack((Z_ulaplace[1:, :],np.inf*H_ONE))[:, :-1]))+np.hstack((np.inf*V_ONE, np.vstack((Ev[1:, :],np.inf*H_ONE))[:, :-1])),  
                              np.vstack((Z_vlaplace[1:, :],np.inf*H_ONE))+np.vstack((Eu[1:, :],np.inf*H_ONE))+np.hstack((np.vstack((Z_ulaplace[1:, :],np.inf*H_ONE))[:, 1:],np.inf*V_ONE))+np.hstack((np.vstack((Ev[1:, :],np.inf*H_ONE))[:, 1:],np.inf*V_ONE)),             
                            ))   
        # corresponding state variable:{[0, -1],
        #                               [0, 1],
        #                               [0, 0],
        #                               [-1, 0],
        #                               [1, 0],
        #                               [-1, -1],
        #                               [-1, 1],
        #                               [1, -1],
        #                               [1, 1]}
        Ev_stack = np.array((
                              np.vstack((np.inf*H_ONE, Z_vlaplace[:-1, :]))+(abs(vmin_index-0*MATRIX_ONE)+abs(np.vstack((np.inf*H_ONE, vmin_index[:-1, :]))-0*MATRIX_ONE)+con_U)*MAX,
                              np.vstack((Z_vlaplace[1:, :], np.inf*H_ONE))+(abs(vmin_index-MATRIX_ONE)+abs(np.vstack((vmin_index[1:, :], np.inf*H_ONE))-MATRIX_ONE)+con_D)*MAX,
                              Ev,
                              2*(np.hstack((np.inf*V_ONE, Z_ulaplace[:, :-1]))+np.hstack((np.inf*V_ONE, Ev[:, :-1]))),   
                              2*(np.hstack((Z_ulaplace[:, 1:],np.inf*V_ONE))+np.hstack((Ev[:, 1:],np.inf*V_ONE))),
                              np.hstack((np.inf*V_ONE, Z_ulaplace[:, :-1]))+np.hstack((np.inf*V_ONE, Ev[:, :-1]))+np.vstack((np.inf*H_ONE, np.hstack((np.inf*V_ONE, Z_vlaplace[:, :-1]))[:-1, :]))+np.vstack((np.inf*H_ONE, np.hstack((np.inf*V_ONE, Eu[:, :-1]))[:-1, :])),
                              np.hstack((np.inf*V_ONE, Z_ulaplace[:, :-1]))+np.hstack((np.inf*V_ONE, Ev[:, :-1]))+np.vstack((np.hstack((np.inf*V_ONE, Z_vlaplace[:, :-1]))[1:, :],np.inf*H_ONE))+np.vstack((np.hstack((np.inf*V_ONE, Eu[:, :-1]))[1:, :],np.inf*H_ONE)),
                              np.hstack((Z_ulaplace[:, 1:],np.inf*V_ONE))+np.hstack((Ev[:, 1:],np.inf*V_ONE))+np.vstack((np.inf*H_ONE, np.hstack((Z_vlaplace[:, 1:],np.inf*V_ONE))[:-1, :]))+np.vstack((np.inf*H_ONE, np.hstack((Eu[:, 1:],np.inf*V_ONE))[:-1, :])),
                              np.hstack((Z_ulaplace[:, 1:],np.inf*V_ONE))+np.hstack((Ev[:, 1:],np.inf*V_ONE))+np.vstack((np.hstack((Z_vlaplace[:, 1:],np.inf*V_ONE))[1:, :],np.inf*H_ONE))+np.vstack((np.hstack((Eu[:, 1:],np.inf*V_ONE))[1:, :],np.inf*H_ONE)),
                            )) 
        Gu_stack = np.array((
                              Gu+1/(i+1)*Z-1/(i+1)*(np.hstack((0*V_ONE, Z[:, :-1]))+np.hstack((0*V_ONE, Gu[:, :-1]))),
                              Gu-1/(i+1)*Z+1/(i+1)*(np.hstack((Z[:, 1:], 0*V_ONE))-np.hstack((Gu[:, 1:], 0*V_ONE))),
                              Gu,
                              np.vstack((0*H_ONE, Gu[:-1, :])),
                              np.vstack((Gu[1:, :],0*H_ONE)),
                              np.vstack((0*H_ONE, Gu[:-1, :]))-np.hstack((0*V_ONE,np.vstack((0*H_ONE, Gv[:-1, :]))[:, :-1]))+Gv,
                              np.vstack((0*H_ONE, Gu[:-1, :]))+np.hstack((np.vstack((0*H_ONE, Gv[:-1, :]))[:, 1:],0*V_ONE))-Gv,
                              np.vstack((Gu[1:, :],0*H_ONE))+np.hstack((0*V_ONE,np.vstack((Gv[1:, :],0*H_ONE))[:, :-1]))-Gv,
                              np.vstack((Gu[1:, :],0*H_ONE))-np.hstack((np.vstack((Gv[1:, :],0*H_ONE))[:, 1:],0*V_ONE))+Gv
                              )).reshape(9,-1)
             
        Gv_stack = np.array((
                              Gv+1/(i+1)*Z-1/(i+1)*(np.vstack((0*H_ONE, Z[:-1, :]))+np.vstack((0*H_ONE, Gv[:-1, :]))),
                              Gv-1/(i+1)*Z+1/(i+1)*(np.vstack((Z[1:, :], 0*H_ONE))-np.vstack((Gv[1:, :], 0*H_ONE))),
                              Gv,
                              np.hstack((0*V_ONE, Gv[:, :-1])),
                              np.hstack((Gv[:, 1:],0*V_ONE)),
                              np.hstack((0*V_ONE, Gv[:, :-1]))-np.vstack((0*H_ONE, np.hstack((0*V_ONE, Gu[:, :-1]))[:-1, :]))+Gu,
                              np.hstack((0*V_ONE, Gv[:, :-1]))+np.vstack((np.hstack((0*V_ONE, Gu[:, :-1]))[1:, :], 0*H_ONE))-Gu,
                              np.hstack((Gv[:, 1:],0*V_ONE))+np.vstack((0*H_ONE, np.hstack((Gu[:, 1:],0*V_ONE))[:-1, :]))-Gu,
                              np.hstack((Gv[:, 1:],0*V_ONE))-np.vstack((np.hstack((Gu[:, 1:],0*V_ONE))[1:, :], 0*H_ONE))+Gu
                              )).reshape(9,-1)
        # state trandfer
        umin_index = np.argmin(Eu_stack, axis=0)
        vmin_index = np.argmin(Ev_stack, axis=0)
        if (umin_index==2).all() and (vmin_index==2).all(): # if all state variables reach stable 
            print('epoch:',i) 
            break
        # smoothness energy minimization
        Eu = np.min(Eu_stack, axis=0)
        Ev = np.min(Ev_stack, axis=0)
        # depth gradient update
        Gu = Gu_stack[umin_index.reshape(-1), np.arange(VMAX*UMAX)].reshape(VMAX, UMAX)
        Gv = Gv_stack[vmin_index.reshape(-1), np.arange(VMAX*UMAX)].reshape(VMAX, UMAX)
         
    return Gu, Gv
    
# CP2TV
def CP2TV(Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map):
    Nx_t = Gu * cam_fx
    Ny_t = Gv * cam_fy
    Nz_t = -(Z + v_map * Gv + u_map * Gu)
    N_t = cv2.merge((Nx_t, Ny_t, Nz_t))
    return N_t

def delta_xyz_computation(X, Y, Z, pos):
    if pos == 1:
        kernel = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    elif pos == 2:
        kernel = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    elif pos == 3:
        kernel = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
    elif pos == 4:
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
    elif pos == 5:
        kernel = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif pos == 6:
        kernel = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
    elif pos == 7:
        kernel = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    else:
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

    X_d = cv2.filter2D(X, -1, kernel)
    Y_d = cv2.filter2D(Y, -1, kernel)
    Z_d = cv2.filter2D(Z, -1, kernel)

    X_d[Z_d == 0] = np.nan
    Y_d[Z_d == 0] = np.nan
    Z_d[Z_d == 0] = np.nan

    return X_d, Y_d, Z_d

# 3F2N
def TF2N(X, Y, Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map, third_filter):
    Nx_t = Gu * cam_fx
    Ny_t = Gv * cam_fy
    nz_t_volume = np.zeros((VMAX, UMAX, 8))

    for j in range(8):
        X_d, Y_d, Z_d = delta_xyz_computation(X, Y, Z, j)
        nz_t_volume[:, :, j] = -(Nx_t * X_d + Ny_t * Y_d) / Z_d

    if third_filter == 'median':
        Nz_t = np.nanmedian(nz_t_volume, 2)
    else:
        Nz_t = np.nanmean(nz_t_volume, 2)

    Nx_t[np.isnan(Nz_t)] = 0
    Ny_t[np.isnan(Nz_t)] = 0
    Nz_t[(Nx_t == 0) * (Ny_t == 0)] = -1
    N_t = cv2.merge((Nx_t, Ny_t, Nz_t))
    return N_t

if __name__ == '__main__':
    # get camera parameters
    cam_fx, cam_fy, u0, v0, _ = get_dataset_params(DATASET)
    K = np.array([[cam_fx, 0, u0], [0, cam_fy, v0], [0, 0, 1]])
    u_map = np.ones((VMAX, 1)) * np.arange(1, UMAX + 1) - u0  # u-u0
    v_map = np.arange(1, VMAX + 1).reshape(VMAX, 1) * np.ones((1, UMAX)) - v0  # v-v0

    # get ground truth normal [-1,1]
    N_gt = get_normal_gt(DATASET)
    N_gt = vector_normalization(N_gt)
    Nx_gt = N_gt[:, :, 0]
    Ny_gt = N_gt[:, :, 1]
    Nz_gt = N_gt[:, :, 2]
    # Nx_gt, Ny_gt, Nz_gt = vector_normalization(Nx_gt, Ny_gt, Nz_gt)
    # N_gt = cv2.merge((Nx_gt, Ny_gt, Nz_gt))

    # get X, Y, depth, and mask
    X, Y, Z, mask = get_depth(DATASET)

    if SNE == 'SDA':
        if BASE == 'CP2TV':
        # SDA with CP2TV
            Gu, Gv = SDA(Z, cam_fx, cam_fy, u_map, v_map)
            N_t = CP2TV(Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map)
        elif BASE == '3F2N':
        # SDA with 3F2N
            D = 1./Z;  # inverse depth or disparity
            Gu, Gv = SDA(D, cam_fx, cam_fy, u_map, v_map)
            N_t = TF2N(X, Y, Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map, 'median')
    elif SNE == 'OTW':
        if BASE == 'CP2TV':
        # CP2TV
            kernel_Gx = np.array([[0, 0, 0],[-2, 2, 0],[0, 0, 0]])
            kernel_Gy = np.array([[0, -2, 0],[0, 2, 0],[0, 0, 0]])
            # get partial u, partial v
            Gu = cv2.filter2D(Z, -1, kernel_Gx) / 2
            Gv = cv2.filter2D(Z, -1, kernel_Gy) / 2
            # compute normal: (nx,ny,nz) using CP2TV
            N_t = CP2TV(Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map)
        elif BASE == '3F2N':
        # 3F2N
            kernel_Gx = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
            kernel_Gy = np.array([[0, -1, 0],[0, 0, 0],[0, 1, 0]])
            D = 1 / Z
            Gu = cv2.filter2D(D, -1, kernel_Gx)
            Gv = cv2.filter2D(D, -1, kernel_Gy)
            N_t = TF2N(X, Y, Z, Gu, Gv, cam_fx, cam_fy, u_map, v_map, 'median')

    # vector normalization
    N_t = vector_normalization(N_t)
    Nx_t = N_t[:, :, 0]
    Ny_t = N_t[:, :, 1]
    Nz_t = N_t[:, :, 2]

    # show the ground truth normal
    ngt_vis = visualization_map_creation(Nx_gt, Ny_gt, Nz_gt, mask)
    cv2.imshow('ngt_vis', ngt_vis[:, :, ::-1])

    # show the computed normal
    nt_vis = visualization_map_creation(Nx_t, Ny_t, Nz_t, mask)
    cv2.imshow('nt_vis', nt_vis[:, :, ::-1])

    # compute error and evaluate the model
    error_map, ea = evaluation(N_gt, N_t, mask)
    errMap = error_map.astype(np.uint8)
    errMap = cv2.applyColorMap(errMap, 2)

    large_err_pos = np.argwhere(error_map >= error_map.max() - 5)
    for x, y in large_err_pos:
        # 注意drawMarker的参数是(u,v)，不是(x,y)
        errMap = cv2.drawMarker(errMap, (y, x), (0, 0, 255), markerSize=5)
    cv2.imshow(f'errMap', errMap)

    # edge_to_opt = (graph_to_optimize * 255).astype('uint8')
    # cv2.imshow(f'edge_to_opt', edge_to_opt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("ea:", ea)

