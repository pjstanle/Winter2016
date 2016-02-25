import numpy as np
from HW3_truss import *
from scipy.optimize import minimize

global truss_calls

truss_calls = np.array([])

def obj(A):
    mass, _, dm_dA, _ = tenbartruss(A, 'CS')
    return mass, dm_dA


def con(A):
    yield_stress = np.array([25000, 25000, 25000, 25000, 25000, 25000, 25000, 25000, 75000, 25000])
    constraints = np.zeros(len(A)*2)
    stress = tenbartruss(A, 'CS')[1]
    for i in range(len(A)):
        constraints[i] = yield_stress[i]-stress[i]
        constraints[i+10] = yield_stress[i]+stress[i]
    return constraints


def congrad(A):
    gradients1 = np.empty((10,10), dtype=complex)
    gradients2 = np.empty((10,10), dtype=complex)
    stress_grads = tenbartruss(A, 'CS')[3]
    for i in range(len(A)):
        gradients1[i] = -stress_grads[i]
        gradients2[i] = stress_grads[i]
    gradients1 = np.transpose(gradients1)
    gradients2 = np.transpose(gradients2)
    gradients = np.vstack((gradients1, gradients2))
    return gradients


if __name__ == "__main__":
    A0 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

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

    bounds = np.empty((10,2), dtype=complex)
    bounds[:] = (0.1, 100)
    options = {'disp': True}
    constraints = {'type': 'ineq', 'fun': con, 'jac': congrad}
    res = minimize(obj, A0, method='SLSQP', jac=True, bounds=bounds, tol=1e-6, constraints=constraints, options=options)

    print res.x

    print "Iterations: ", truss_calls