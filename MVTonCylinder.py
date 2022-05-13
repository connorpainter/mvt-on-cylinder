###############################################################################
##############################    IMPORTATIONS   ##############################
###############################################################################



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import fmin
from sympy import Plane



###############################################################################
###############################    PARAMETERS   ###############################
###############################################################################



L = np.pi
R = 1
r = 1/100
C = 2*np.pi*R
z0, t0 = (L/2, C/4)
resolution = (175, 175)
R_guess = C/10

tau = 2
vTol = None
hGamma = max(L/resolution[0], C/resolution[1])
hTol = hGamma/2**6
maxEntNodes = None



###############################################################################
##############################    CONVENIENCE   ###############################
###############################################################################



pi = np.pi



###############################################################################
################################    ASSEMBLY   ################################
###############################################################################



def assembly(): 
    
    tri = initializeMesh()
    phis = get_phis(tri.points)
    D, dD, iDs, idDs, C, iCs = findBoundary(phis, tri)
    tri = conformMesh(phis, tri)
    U, V, D_, iD_s = FEM(tri, D, dD, iDs, idDs)
    phis = updatePhi(U, V, phis)
    
    return



###############################################################################
###############################    FORMULAS   #################################
###############################################################################



def G_cyl(z, t, z0=z0, t0=t0, terms=10):
    
    k = np.arange(-terms,terms)
    
    return -1/(4*pi)*np.sum(np.log((np.cosh(t-t0-2*pi*R*k) - np.cos(z-z0))/(np.cosh(t-t0-2*pi*R*k)-np.cos(z+z0))))



def paraboloid(z, t, z0=z0, t0=t0, R_guess=R_guess, A=10):
    
    return A*((z-z0)**2 + (t-t0)**2 - R_guess**2)



###############################################################################
##########################    COMPLEX FUNCTIONS    ############################
###############################################################################



def initializeMesh(plot=False):
    
    global tri
    
    pz, pt = resolution
    half_pt = np.ceil(pt/2).astype(np.int64)
    dz, dt = L/(pz-1), C/(2*half_pt-1)
    
    z1 = np.linspace(0, L-dz/2, pz)
    t1 = np.linspace(0, C-dt, half_pt)
    z2 = z1 + dz/2
    t2 = t1 + dt
    grid1 = np.meshgrid(z1, t1)
    grid2 = np.meshgrid(z2, t2)
    
    zs = np.append(np.ravel(grid1[0]), np.ravel(grid2[0]))
    ts = np.append(np.ravel(grid1[1]), np.ravel(grid2[1]))
    coords = np.transpose([zs, ts])
    
    tri = Delaunay(coords) 
    
    if plot: 
        plt.figure(figsize=(20,20))
        plt.triplot(zs, ts, tri.simplices)
        plt.xlabel(r"$z$", fontsize=20) and plt.ylabel(r"$\theta$", fontsize=20)
        
    return tri



def findBoundary(phis, tri, plot=False):
    
    global D
    global dD
    global iDs
    global idDs
    global C
    global iCs
    
    D, dD, idDs = [], [], []
    iCs = np.where(phis>=0)[0]
    iDs = np.where(phis<0)[0]
    C = tri.points[iCs]
    D = tri.points[iDs]
    
    for iD in iDs:
        iNeighbors = vnvFinder(tri, iD)
        iNeighborsOut = np.intersect1d(iNeighbors, iCs)
        idDs = np.append(idDs, iNeighborsOut)
        dD = tri.points[iNeighborsOut] if len(dD)==0 else np.concatenate((dD, tri.points[iNeighborsOut]))
    
    dD, idDs = unique(dD), unique(idDs)
    
    if plot:
        fig, ax = plt.subplots(figsize=(75,75))
        ax.scatter(C[:,0], C[:,1], c='r', s=100)
        ax.scatter(dD[:,0], dD[:,1], c='k', s=100)
        ax.scatter(D[:,0], D[:,1], c='g', s=100)
        ax.triplot(tri.points[:,0], tri.points[:,1], triangles=tri.simplices)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$z$") and ax.set_ylabel(r"$\theta$")
    
    return D, dD, iDs, idDs, C, iCs



def conformMesh(phis, tri): 
    
    ## Needs to be coded: tri is conformed to the boundary using phis.
    
    return tri



def FEM(tri, D, dD, iDs, idDs, plot=True): 
    
    global U
    global V
    global D_
    global iD_s
    
    D_ = np.concatenate((dD, D))
    iD_s = np.append(idDs, iDs).astype(np.int64)
    n = len(D_)
    Ku = np.zeros((n,n))
    Kv = np.zeros((n,n))
    Fu = np.zeros((n,))
    Fv = np.zeros((n,))
    
    for i in range(n):
        node = D_[i]
        
        if i<len(dD):
            Ku[i,i] = 1
            Kv[i,i] = Ku[i,i]
            Fu[i] = G_cyl(node[0], node[1])
            Fv[i] = Fu[i]
        
        else:
            iEs = np.where(tri.simplices == iD_s[i])[0]
            
            for e in iEs:
                nodePlace = np.where(tri.simplices[e] == iD_s[i])[0]
                (x1, y1), (x2, y2), (x3, y3) = tri.points[tri.simplices[e]]
                A = np.abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))/2
                nodePrev = tri.simplices[e, (nodePlace-1)%3]
                nodeNext = tri.simplices[e, (nodePlace+1)%3]
                
                for l in tri.simplices[e]: 
                    j = np.where(iD_s == l)[0]
                    lPlace = np.where(tri.simplices[e] == l)[0]
                    lPrev = tri.simplices[e, (lPlace-1)%3]
                    lNext = tri.simplices[e, (lPlace+1)%3]
                    b_i = tri.points[nodeNext,1] - tri.points[nodePrev,1]
                    b_j = tri.points[lNext, 1] - tri.points[lPrev, 1]
                    c_i = tri.points[nodePrev, 0] - tri.points[nodeNext, 0]
                    c_j = tri.points[lPrev, 0] - tri.points[lNext, 0]
                    
                    Ku[i,j] = Ku[i,j] + (b_i*b_j + c_i*c_j)/(4*A)
                    Kv[i,j] = Ku[i,j]
                
                Fu[i] = Fu[i] + 4*A/(6*r**2) 
        
    U = np.linalg.solve(Ku, Fu)
    V = np.linalg.solve(Kv, Fv)
    
    if plot:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(D_[:,0], D_[:,1], U, cmap='coolwarm', alpha=1)
        ax.set_xlabel(r"$z$", fontsize=20) 
        ax.set_ylabel(r"$\theta$", fontsize=20)
        ax.set_zlabel(r"$\tilde{u}$", fontsize=20)
        ax.set_title("FEM Solution", fontsize=30)
        
        fig2 = plt.figure(figsize=(12,12))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_trisurf(D_[:,0], D_[:,1], V, cmap='coolwarm', alpha=1)
        ax2.set_xlabel(r"$z$", fontsize=20) 
        ax2.set_ylabel(r"$\theta$", fontsize=20)
        ax2.set_zlabel(r"$\tilde{v}$", fontsize=20)
        ax2.set_title("Harmonic FEM Solution", fontsize=30)
    
    return U, V, D_, iD_s



def updatePhi(U, V, phis): 
    
    ## Needs to be coded: signed distance function phi is updated based on error in FEM solution.
    
    return phis




    


###############################################################################
###########################    HELPER FUNCTIONS    ############################
###############################################################################



def get_Gs(locs): return np.array([G_cyl(z,t) for z,t in locs])



def get_phis(locs): return np.array([paraboloid(z,t) for z,t in locs])



def unique(a): 
    
    if len(np.shape(a))==1:
        b = np.empty(0, dtype=a.dtype)
        for i in range(len(a)):
            if a[i] not in b:
                b = np.hstack((b,a[i]))
    
    if len(np.shape(a))==2:
        b = [tuple(row) for row in a]
        b = np.unique(b, axis=0)
    
    return b



def vnvFinder(tri, k):
    
    vnv = tri.vertex_neighbor_vertices
    
    return vnv[1][vnv[0][k]:vnv[0][k+1]]



def shoelaceArea(vertices):
    
    z,t = np.transpose(vertices)

    return 1/2*(np.dot(z,np.roll(t,-1)) - np.dot(np.roll(z,-1),t))



def sumSideSquare(vertices):
    
    return sum(np.sum((vertices - np.roll(vertices,1,axis=0))**2, axis=1))



def getQ(vertices):
    
    area = shoelaceArea(vertices)
    suml2 = sumSideSquare(vertices)
    C2 = 4*np.sqrt(3)
    
    return C2*area/suml2



def worstQ(iVertex, tri, direction, lamb, returnall=False):
    
    vnv = tri.points[vnvFinder(tri, iVertex)]                                                                           ## Get neighboring vertices of vertex at given index.
    vertex = tri.points[iVertex]
    vertex = vertex + np.array([lamb,0]) if direction=='z' else vertex + np.array([0,lamb])                             ## Perturb given vertex by lambda in given direction.
    
    ## Order 1-ring CCW for correctly signed area.
    centerpoint = np.mean(vnv, axis=0)
    angles = np.array([ np.angle((vnv[i,0]-centerpoint[0])+(vnv[i,1]-centerpoint[1])*1j) for i in range(len(vnv)) ])    ## Compute angles of each vertex in 1-ring from centerpoint.
    vnv = vnv[np.argsort(angles)]                                                                                       ## Resort 1-ring vertices CCW.
    
    Qs = np.empty(0)
    
    for i in range(len(vnv)):                                                                                           ## Loop through triangles in 1-ring.
        triangle = np.vstack((vnv[[i%len(vnv), (i+1)%len(vnv)]], vertex))                                               ## Build the triangle so that CCW ordered vertices imply positively signed area.
        Q = getQ(triangle)                                                                                              ## Get quality of the triangle.
        Qs = np.append(Qs, [Q])
    
    if returnall: return Qs
    
    return np.min(Qs)



def optimizeQ(iVertex, tri, direction, guess=0):
    
    func = lambda lamb : -1*worstQ(iVertex, tri, direction, lamb)
    
    return fmin(func, guess)



def QPlot1d(iVertex, tri, direction, lambs, includeall=True, includemin=True):
    
    Qs = []
    for lamb in lambs:
        Q = worstQ(iVertex, tri, direction, lamb, returnall=True)
        Qs = np.array([Q]) if len(Qs)==0 else np.vstack((Qs, Q))
    if includeall: [plt.plot(lambs, Qs[:,i]) for i in range(len(Qs[0]))]
    if includemin: plt.plot(lambs, np.min(Qs, axis=1), 'k', linewidth=4)
    plt.xlabel(r"Perturbation $\lambda$")
    plt.ylabel(r"Quality $Q$")
    plt.grid()
    
    return



def projectToZeroLine(points, i):
    
    plane = Plane(points[0], points[1], points[2])
    zero = Plane((0,0,0), normal_vector=(0,0,1))
    zeroLine = plane.intersection(zero)[0]
    closestPoint = zeroLine.projection(points[i])
    
    return np.array(closestPoint, dtype=np.float64)
























