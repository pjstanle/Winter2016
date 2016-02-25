import numpy as np
import scipy as sp
from algopy import UTPM
from algopy import sin, cos, arccos, sqrt


#Determine how much of the turbine is in the wake of the other turbines
def overlap(xin, params):
    xup, xdown, yup, ydown = xin
    r, alpha = params
    overlap_fraction = np.zeros(np.size(x))

    #define dx as the upstream x coordinate - the downstream x coordinate then rotate according to wind direction
    dx = xdown - xup
    #define dy as the upstream y coordinate - the downstream y coordinate then rotate according to wind direction
    dy = ydown - yup
    R = r+dx*alpha #The radius of the wake depending how far it is from the turbine
    A = r**2*np.pi #The area of the turbine

    if dx > 0:
        if np.abs(dy) <= R-r:
            overlap_fraction = 1 #if the turbine is completely in the wake, overlap is 1, or 100%
        elif np.abs(dy) >= R+r:
            overlap_fraction = 0 #if none of it touches the wake, the overlap is 0
        else:
            #if part is in and part is out of the wake, the overlap fraction is defied by the overlap area/rotor area
            overlap_area = r**2.*arccos((dy**2.+r**2.-R**2.)/(2.0*dy*r))+R**2.*arccos((dy**2.+R**2.-r**2.)/(2.0*dy*R))-0.5*sqrt((-dy+r+R)*(dy+r-R)*(dy-r+R)*(dy+r+R))
            overlap_fraction = overlap_area/A
    else:
        overlap_fraction = 0 #turbines cannot be affected by any wakes that start downstream from them

    # print overlap_fraction
    return overlap_fraction #retrun the n x n matrix of how each turbine is affected by all of the others
                            #for example [0, 0.5]
                                        #[0, 0] means that the first turbine (row one) has half of its area in the
                                        #wake of the second turbine (row two). The overlap_fraction on the second
                                        #turbine is zero, so we can conclude that it is upstream of the first



def rotate(x, y, U_direction_radians):
    x_r = x*cos(U_direction_radians)-y*sin(U_direction_radians)
    y_r = x*sin(U_direction_radians)+y*cos(U_direction_radians)
    return x_r, y_r


if __name__ == '__main__':

    "Define Variables"
    theta = 0.1
    alpha = sp.tan(theta)
    x = np.array([1000, 1000, 1000, 2000, 2000, 2000, 3000, 3000, 3000]) #x coordinates of the turbines
    y = np.array([1500, 2000, 3000, 1000, 2000, 3000, 1000, 2000, 3000]) #y coordinates of the turbines
    rho = 1.1716
    a = 1. / 3.
    U_velocity = 8.
    "0 degrees is coming from due North. +90 degrees means the wind is coming from due East, -90 from due West"
    U_direction = -98.
    r_0 = 40

    U_direction_radians = (U_direction+90) * np.pi / 180.
    #print U_direction_radians
    Cp = 4.*a*(1-a)**2.
    #rotate position coordinates
    x_r, y_r = rotate(x, y, U_direction_radians)

    #define parameters
    params = np.array([r_0, alpha])

    #define values that we want to take a derivative with respect to
    xin = np.array([x_r[3], x_r[6], y_r[3], y_r[6]])

    #Finite Differencing
    step = 1e-6
    p1 = np.array([step, 0, 0, 0])
    p2 = np.array([0, step, 0, 0])
    p3 = np.array([0, 0, step, 0])
    p4 = np.array([0, 0, 0, step])
    p = np.array([p1, p2, p3, p4])

    derivative_FD = np.zeros(4)
    for i in range(len(p)):
        derivative_FD[i] = (overlap(xin+p[i], params)-overlap(xin, params))/step

    print "FD: ", derivative_FD

    #Automatic Differentiation

    x_algopy = UTPM.init_jacobian(xin)

    overlap_fraction = overlap(x_algopy, params)

    derivative_auto = UTPM.extract_jacobian(overlap_fraction)

    print "Automatic: ", derivative_auto