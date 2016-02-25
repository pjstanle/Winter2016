import numpy as np
from math import sqrt
from cmath import sin ,cos
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt

global truss_calls
truss_calls = np.array([])
global con_crit
con_crit = np.array([])
global major_iter
major_iter = np.array([])



def obj(A):
    mass, _, dm_dA, _ = tenbartruss(A, 'FD')
    return mass, dm_dA


def con(A):
    yield_stress = np.array([25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 75000, 25000])
    constraints = np.zeros(len(A)*2)
    stress = tenbartruss(A, 'FD')[1]
    for i in range(len(A)):
        constraints[i] = yield_stress[i]-stress[i]
        constraints[i+10] = yield_stress[i]+stress[i]
    return constraints


def congrad(A):
    gradients1 = np.empty((10,10), dtype=complex)
    gradients2 = np.empty((10,10), dtype=complex)
    stress_grads = tenbartruss(A, 'FD')[3]
    for i in range(len(A)):
        gradients1[i] = -stress_grads[i]
        gradients2[i] = stress_grads[i]
    gradients1 = np.transpose(gradients1)
    gradients2 = np.transpose(gradients2)
    gradients = np.vstack((gradients1, gradients2))
    return gradients


def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element

    Parameters
    ----------
    E : float
       modulus of elasticity
    A : float
       cross-sectional area
    L : float
       length of element
    phi : float
       orientation of element

    Outputs
    -------
    K : 4 x 4 ndarray
       stiffness matrix
    S : 1 x 4 ndarray
       stress matrix

    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)
    k1 = np.hstack([k0, -k0])
    K = E*A/L*np.vstack([k1, -k1])

    # stress matrix
    S = E/L*np.array([[-c, -s, c, s]])

    return K, S


def node2idx(node, DOF):
   """Computes the appropriate indices in the global matrix for
   the corresponding node numbers.  You pass in the number of the node
   (either as a scalar or an array of locations), and the degrees of
   freedom per node and it returns the corresponding indices in
   the global matrices

   """

   idx = np.array([], dtype=np.int)

   for i in range(len(node)):

       n = node[i]
       start = DOF*(n-1)
       finish = DOF*n

       idx = np.concatenate((idx, np.arange(start, finish, dtype=np.int)))

   return idx


def truss(start, finish, phi, A, L, E, rho, Fx, Fy, rigid):

    global truss_calls
    truss_calls = np.append(truss_calls, len(truss_calls))
    """Computes mass and stress for an arbitrary truss structure

    Parameters
    ----------
    start : ndarray of length nbar
       index of start of bar (1-based indexing) start and finish can be in any order as long as consistent with phi
    finish : ndarray of length nbar
       index of other end of bar (1-based indexing)
    phi : ndarray of length nbar (radians)
       defines orientation or bar
    A : ndarray of length nbar
       cross-sectional areas of each bar
    L : ndarray of length nbar
       length of each bar
    E : ndarray of length nbar
       modulus of elasticity of each bar
    rho : ndarray of length nbar
       material density of each bar
    Fx : ndarray of length nnode
       force in the x-direction at each node
    Fy : ndarray of length nnode
       force in the y-direction at each node
    rigid : list(boolean) of length nnode
       True if node_i is rigidly constrained

    Outputs
    -------
    mass : float
       mass of the entire structure
    stress : ndarray of length nbar
       stress of each bar

    """

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom
    nbar = len(A)  # number of bars

    # mass
    mass = np.sum(rho*A*L)

    # stiffness and stress matrices
    K = np.zeros((DOF*n, DOF*n), dtype=complex)
    S = np.zeros((nbar, DOF*n), dtype=complex)

    for i in range(nbar):  # loop through each bar

       # compute submatrix for each element
       Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

       # insert submatrix into global matrix
       idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
       K[np.ix_(idx, idx)] += Ksub
       S[i, idx] = Ssub

    # applied loads
    F = np.zeros((n*DOF, 1))

    for i in range(n):
       idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
       F[idx[0]] = Fx[i]
       F[idx[1]] = Fy[i]


    # boundary condition
    idx = np.squeeze(np.where(rigid))
    remove = node2idx(idx+1, DOF)  # add 1 b.c. made indexing 1-based for convenience

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress, d, K, S


def tenbartruss(A, grad_type):
   """This is the subroutine for the 10-bar truss.  You will need to complete it.

   Parameters
   ----------
   A : ndarray of length 10
       cross-sectional areas of all the bars
   grad_type : string (optional)
       gradient type.  'FD' for finite difference, 'CS' for complex step,
       'AJ' for adjoint

   Outputs
   -------
   mass : float
       mass of the entire structure
   stress : ndarray of length 10
       stress of each bar
   dmass_dA : ndarray of length 10
       derivative of mass w.r.t. each A
   dstress_dA : 10 x 10 ndarray
       dstress_dA[i, j] is derivative of stress[i] w.r.t. A[j]

   """
   global major_iter
   major_iter = np.append(major_iter, len(major_iter)+1)
   global con_crit
   # --- setup 10 bar truss ----
   E = 1.*10**7 #modulus of elasticity (psi)
   rho = 0.1 #density (lb/in**3)
   P = 100000. #applied load (lb)
   L_short = 360. #length of square sides (in)
   L_long = L_short*sqrt(2) #length of diagonal sides (in)

   start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4])
   stop = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1])
   phi = np.array([0., 0., 0., 0., pi/2., pi/2., -pi/4., pi/4., -pi/4., pi/4.])

   L_vector = np.array([L_short, L_short, L_short, L_short, L_short, L_short, L_long, L_long, L_long, L_long])
   E_vector = np.array([E, E, E, E, E, E, E, E, E, E])
   rho_vector = np.array([rho, rho, rho, rho, rho, rho, rho, rho, rho, rho])
   Fx = np.array([0., 0., 0., 0., 0., 0.]) #Forces on each node in the x direction
   Fy = np.array([0., -P, 0., -P, 0., 0.]) #Forces on each node in the y direction
   rigid = [False, False, False, False, True, True] #True if the node is rigidly constrained

   # --- call truss function ----

   mass, stress, d, K, S = truss(start, stop, phi, A, L_vector, E_vector, rho_vector, Fx, Fy, rigid)
   # --- compute derivatives for provided grad_type ----

   if grad_type == 'FD':
       step = 1.*10**-6 #define a step size for finite difference

       p1 = ([step, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       p2 = ([0, step, 0, 0, 0, 0, 0, 0, 0, 0])
       p3 = ([0, 0, step, 0, 0, 0, 0, 0, 0, 0])
       p4 = ([0, 0, 0, step, 0, 0, 0, 0, 0, 0])
       p5 = ([0, 0, 0, 0, step, 0, 0, 0, 0, 0])
       p6 = ([0, 0, 0, 0, 0, step, 0, 0, 0, 0])
       p7 = ([0, 0, 0, 0, 0, 0, step, 0, 0, 0])
       p8 = ([0, 0, 0, 0, 0, 0, 0, step, 0, 0])
       p9 = ([0, 0, 0, 0, 0, 0, 0, 0, step, 0])
       p10 = ([0, 0, 0, 0, 0, 0, 0, 0, 0, step])
       p = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

       dmass_dA = np.empty((10))
       dstress_dA = np.empty((10, 10))
       for i in range(0, len(p)):
           new_A_forwards = A+p[i] #new A to evaluate function
           new_A_backwards = A-p[i]

           mass_forward, stress_forward, _, _, _  = truss(start, stop, phi, new_A_forwards, L_vector, E_vector, rho_vector, Fx,
                                                Fy, rigid) #new function values forward step
           mass_backwards, stress_backwards, _, _, _  = truss(start, stop, phi, new_A_backwards, L_vector, E_vector, rho_vector,
                                                    Fx, Fy, rigid) #new function values backwards step
           dmass_dA[i] = (mass_forward-mass_backwards).real/(2*step) #derivative of mass with respect to area
           dstress_dA[i] = (stress_forward-stress_backwards).real/(2*step) #derivative of stress with respect to area

   if grad_type == 'CS':
       h = 1.*10**-30
       step = complex(0, h)

       p1 = ([step, 0, 0, 0, 0, 0, 0, 0, 0, 0])
       p2 = ([0, step, 0, 0, 0, 0, 0, 0, 0, 0])
       p3 = ([0, 0, step, 0, 0, 0, 0, 0, 0, 0])
       p4 = ([0, 0, 0, step, 0, 0, 0, 0, 0, 0])
       p5 = ([0, 0, 0, 0, step, 0, 0, 0, 0, 0])
       p6 = ([0, 0, 0, 0, 0, step, 0, 0, 0, 0])
       p7 = ([0, 0, 0, 0, 0, 0, step, 0, 0, 0])
       p8 = ([0, 0, 0, 0, 0, 0, 0, step, 0, 0])
       p9 = ([0, 0, 0, 0, 0, 0, 0, 0, step, 0])
       p10 = ([0, 0, 0, 0, 0, 0, 0, 0, 0, step])
       p = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

       dmass_dA = np.empty((10))
       dstress_dA = np.empty((10, 10))
       for i in range(0, len(p)):
           A_complex = A+p[i]
           mass_complex, stress_complex, _, _, _ = truss(start, stop, phi, A_complex, L_vector, E_vector, rho_vector, Fx,
                                                         Fy, rigid)
           dmass_dA[i] = mass_complex.imag/h
           dstress_dA[i] = stress_complex.imag/h

   if grad_type == 'AJ':
       dK_dA = np.zeros((10,8,8), dtype=complex)
       for j in range(0, len(A)):
           n = len(Fx)  # number of nodes
           DOF = 2.  # number of degrees of freedom
           nbar = len(A)  # number of bars

           # dK/dA matrix
           temp = np.zeros((DOF*n, DOF*n), dtype=complex)


           # compute submatrix
           Ksub, _ = bar(E_vector[j], A[j], L_vector[j], phi[j])

           # insert submatrix into global matrix
           idx = node2idx([start[j], stop[j]], DOF)  # pass in the starting and ending node number for this element
           temp[np.ix_(idx, idx)] += Ksub

           for i in range(n):
               idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience

           # boundary condition
           idx = np.squeeze(np.where(rigid))
           remove = node2idx(idx+1, DOF)  # add 1 b.c. made indexing 1-based for convenience

           temp = np.delete(temp, remove, axis=0)
           temp = np.delete(temp, remove, axis=1)
           dK_dA[j] = temp/A[j]

       dstress_dA = np.empty((10,10,1), dtype=complex)
       SK = np.dot(-S,np.linalg.inv(K))
       for i in range(0,len(A)):
           dstress_dA[i] = np.dot(SK,np.dot(dK_dA[i],d))

       dstress_dA = np.reshape(dstress_dA, (10,10), order='C')
       dmass_dA = np.zeros(len(A))
       for i in range(0, len(A)):
           dmass_dA[i] = rho_vector[i]*L_vector[i]

   # if len(con_crit) == 0:
   #     con_crit = np.append(con_crit, mass.real)
   # elif len(con_crit) != 0:
   #     con_crit = np.append(con_crit, mass.real-con_crit[len(con_crit)-1])

   return mass.real, stress.real, dmass_dA.real, dstress_dA.real


if __name__ == '__main__':
    A0 = np.array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

    # FDmass, FDstress, FDdm_dA, FDds_dA = tenbartruss(A0, 'FD')
    # CSmass, CSstress, CSdm_dA, CSds_dA = tenbartruss(A0, 'CS')
    # AJmass, AJstress, AJdm_dA, AJds_dA = tenbartruss(A0, 'AJ')
    #
    # print "FINITE DIFFERENCE"
    # print "FD MASS: ", FDmass
    # print "FD STRESS: ", FDstress
    # print "FD dm/dA: ", FDdm_dA
    # print "FD ds/dA: ", FDds_dA
    #
    # print "COMPLEX STEP"
    # print "CS MASS: ", CSmass
    # print "CS STRESS: ", CSstress
    # print "CS dm/dA: ", CSdm_dA
    # print "CS ds/dA: ", CSds_dA
    #
    # print "ADJOINT"
    # print "AJ MASS: ", AJmass
    # print "AJ STRESS: ", AJstress
    # print "AJ dm/dA: ", AJdm_dA
    # print "AJ ds/dA: ", AJds_dA

    plt.figure(1)
    bounds = np.empty((10,2), dtype=complex)
    bounds[:] = (0.1, 100)
    options = {'disp': True, 'iprint': 2}
    constraints = {'type': 'ineq', 'fun': con, 'jac': congrad}
    res = minimize(obj, A0, method='SLSQP', jac=True, bounds=bounds, tol=1e-6, constraints=constraints, options=options)

    print res.x

    print "Truss Function Calls: ", len(truss_calls)

    print "Tenbartruss Function Calls: ", len(major_iter)

    mass_vector = np.array([8.392935E+02, 8.773779E+02, 1.108012E+03, 1.358784E+03, 1.475279E+03, 1.497036E+03, 1.497599E+03, 1.497600E+03])

    convergence_crit = np.array([mass_vector[0]])
    for i in range(1, len(mass_vector)):
            convergence_crit = np.append(convergence_crit, mass_vector[i]-mass_vector[i-1])


    print tenbartruss(res.x, 'AJ')[1]

    plt.semilogy([1,2,3,4,5,6,7,8], convergence_crit)
    plt.title('Convergence Plot for Constrained Optimization')
    plt.xlabel('Major Iterations')
    plt.ylabel('Function Difference')
    plt.show()