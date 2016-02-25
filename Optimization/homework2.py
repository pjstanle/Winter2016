import numpy as np
import scipy
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import minimize


global function_calls, mf, rf, bf
mf = np.array([])
rf = np.array([])
bf = np.array([])


def matyas(x):
    global function_calls, mf
    fx = 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    gx0 = 0.52*x[0]-0.48*x[1]
    gx1 = 0.52*x[1]-0.48*x[0]
    gradient = np.array([gx0, gx1])
    function_calls += 1
    mf = np.append(mf, function_calls)

    return fx, gradient


def rosenbrock(x):
    global function_calls, rf
    fx = (1-x[0])**2+100*(x[1]-x[0]**2)**2
    gx0 = -2+2*x[0]-400*x[1]*x[0]+400*x[0]**3
    gx1 = 200*x[1]-200*x[0]**2
    gradient = np.array([gx0, gx1])
    function_calls += 1
    rf = np.append(rf, function_calls)

    return fx, gradient


def uncon(func, x0, epsilon_g, options=None):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    func : function handle
        function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        starting point
    epsilon_g : float
        convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    options : dict
        a dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in, so if the input is None
        you should setup some defaults.

    Outputs
    -------
    xopt : ndarray
        the optimal solution
    fopt : float
        the corresponding function value
    outputs : list
        other miscelaneous outputs that you might want, for example an array
        containing a convergence metric at each iteration.
    """

    if options is None:
        # set defaults here for how you want me to run it.
    # Your code goes here!  You can (and should) call other functions, but make
    # sure you do not change the function signature for this file.  This is the
    # file I will call to test your algorithm.
        rho = 0.33
        alpha_0 = 1
        x1 = x0
        x_hist = [x1]
        I = np.identity(np.size(x1))
        mu = 10e-6
        f1, g1 = func(x1)
        V1 = I
        p = np.dot(-V1, g1)
        alpha = alpha_0
        while func(x1+p*alpha)[0] > f1+mu*alpha*np.dot(p,g1):
            alpha = alpha*rho

        x2 = x1+p*alpha
        x_hist.append(x2)
        iterations = 1
        major_iter = np.array([0, 1])
        f2, g2 = func(x2)
        max_grad = np.array([np.max(np.abs(g1)), np.max(np.abs(g2))])

        while np.max(np.abs(g2)) >= epsilon_g:
            y = g2-g1
            s = x2-x1
            st = np.transpose(s[np.newaxis])
            yt = np.transpose(y[np.newaxis])
            den = np.dot(s, yt)
            term1 = np.dot((I-np.outer(st,y)/den),V1)
            term2 = I-np.outer(yt,s)/den
            term3 = np.outer(st,s)/den
            V2 = np.dot(term1, term2)+term3
            x1 = x2
            f1 = f2
            g1 = g2
            V1 = V2

            alpha = alpha_0
            p = np.dot(-V2, g1)

            while func(x1+p*alpha)[0] > f1+mu*alpha*np.dot(p,g1):
                alpha = alpha*rho

            x2 = x1+p*alpha
            f2, g2 = func(x2)
            iterations += 1
            max_grad = np.append(max_grad, np.max(np.abs(g2)))
            major_iter = np.append(major_iter, iterations)
            #x_hist1.append(x2[0])
            #x_hist2.append(x2[1])
            x_hist.append(x2)
        #x_hist = ([x_hist1, x_hist2])
        outputs = (major_iter, max_grad, x_hist)
        xopt = x2
        fopt = func(x2)

    return xopt, fopt, outputs


def brachistochrone(yint):
    global function_calls, bf
    """brachistochrone problem.

    Parameters
    ----------
    yint : a vector of y location for all the interior points

    Outputs
    -------
    J : scalar proportion to the total time it takes the bead to traverse
        the wire
    g : dJ/dyint the derivatives of J w.r.t. each yint.
    """

    # fill in details

    #gradient
    start = np.array([0, 1])#starting point
    stop = np.array([1, 0])#stop point
    x = np.linspace(start[0], stop[0], num=(np.size(yint)+2))#define the x points
    y = np.insert(yint, 0, start[1])#add the first point onto the y vector
    y = np.append(y, stop[1])#add the last point onto the y vector
    mu_k = 0.3
    H = 1
    g = grad_brach(x, y, mu_k, H)  # note y is not the same as yint.  y should include the end points

    dx = x[1]-x[0]#x is evenly spaced so dx only needs to be evaluated once
    J = 0#initialize J
    for i in range(0, np.size(x)-1):
        dy = y[i+1]-y[i]#dy is the difference between y values
        num = sqrt(dx**2+dy**2)#the numerator of the function in the summation
        den = sqrt(H-y[i+1]-mu_k*x[i+1])+sqrt(H-y[i]-mu_k*x[i])#denominator of this same function
        J = J+num/den#increase the value of J with each iteration

    function_calls += 1
    bf = np.append(bf, function_calls)

    return J, g


def grad_brach(x, y, mu_k, H):
    """gradients of the brachistochrone function.  This function accepts
    as input the full x, and y vectors, but returns gradients only for the
    interior points.

    Parameters
    ----------
    x : array of length n
        an array of x locations including the end points
    y : array of length n
        corresponding heights including the end points
    mu_k : float
        coefficient of kinetic friction
    H : float
        initial height of bead

    Outputs
    -------
    g : array of length n-2
        dJ/dy for all interior points.  Note that the end points are
        fixed and thus are not design variables and so there gradients
        are not included.

    """

    n = len(x)
    g = np.zeros(n-2)

    for i in range(n-1):

        ds = sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        vbar = sqrt(H - y[i+1] - mu_k*x[i+1]) + sqrt(H - y[i] - mu_k*x[i])

        if i > 0:
            dsdyi = -(y[i+1] - y[i])/ds
            dvdyi = -0.5/sqrt(H - y[i] - mu_k*x[i])
            dtdyi = (vbar*dsdyi - ds*dvdyi)/(vbar**2)
            g[i-1] += dtdyi
        if i < n-2:
            dsdyip = (y[i+1] - y[i])/ds
            dvdyip = -0.5/sqrt(H - y[i+1] - mu_k*x[i+1])
            dtdyip = (vbar*dsdyip - ds*dvdyip)/(vbar**2)
            g[i] += dtdyip

    return g


if __name__ == '__main__':
    xin_maty= np.array([-3.76, 2.19])
    xin_ros = np.array([-3.76, 2.19])
    n = 16
    delta = 1./n
    yint = np.linspace(1-delta,delta,n)
    options = {'disp': True}
    alg = 'Newton-CG'
    function_calls = 0

    res =  scipy.optimize.minimize(matyas, xin_maty, method = alg, jac = True, tol=10**-5, options=options)
    print 'built in: ', res.x

    print 'Function Calls, Matyas: ', function_calls

    function_calls = 0
    res =  scipy.optimize.minimize(rosenbrock, xin_ros, method = alg, jac = True, tol=10**-5, options=options)
    print 'built in: ', res.x

    print 'Function Calls, Rosebrock: ', function_calls

    function_calls = 0
    res =  scipy.optimize.minimize(brachistochrone, yint, method = alg, jac = True, tol=10**-5, options=options)
    print 'built in: ', res.x

    print 'Function Calls, Brachristochrone: ', function_calls

    """function_calls = 0
    n_l = n/2
    delta_l = 1./n_l
    yint_l = np.linspace(1-delta,delta,n_l)
    yint_l_o = uncon(brachistochrone, yint_l, 10**-5, options=None)[0]
    yint_warmstart = np.zeros(np.size(n))
    yint_warmstart[np.size(yint_warmstart)] = yint_l_o[np.size(yint_l_o)]/2
    for j in range(0, (np.size(n)-1),2):
        yint_warmstart[j] = yint_l_o[(j/2)]
        yint_warmstart[j+1] = (yint_l_o[j/2]+yint_l_o[j/2+1])/2"""




    """n_test = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    for i in range(0, np.size(n_test)):
        function_calls = 0
        delta = 1./n_test[i]
        yint = np.linspace(1-delta,delta,n_test[i])
        calls = np.size(uncon(brachistochrone, yint, 10**-5, options=None)[2][0])
        print 'Function Calls ', n_test[i], ': ', function_calls
        print 'Major Iterations ', calls"""

    """#BRACHRISTOCHRONE

    brach = uncon(brachistochrone, yint, 10**-5, options=None)
    print brach[0]
    print 'BRACH MIN: ', brach[1]
    print 'brachistochrone function calls: ', np.size(bf)
    print 'brachistochrone score: ', 1./np.size(bf)

    #Plot the convergence of the Brachristochrone
    plt.figure(3)
    plt.yscale('log')
    plt.xlim(0, np.max(brach[2][0]))
    plt.plot(brach[2][0], brach[2][1])
    plt.scatter(brach[2][0], brach[2][1])
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity Norm of the Gradient')
    plt.title('Convergence Plot of the Brachistochrone Function Optimization (n=%s)'%n)
    plt.savefig('brach_convergence%s'%(n) + '.pdf')

    #Plot history of Brachristochrone
    plt.figure(6)
    x_brach = np.linspace(0, 1, num=(n+2))

    for i in range(0, 5):
        iteration_number = np.shape(brach[2][2])[0]*i/5
        y_plot = brach[2][2][iteration_number]
        y_plot = np.append(y_plot, 0)
        y_plot = np.insert(y_plot, 0, 1)
        plt.plot(x_brach, y_plot, label = 'Iteration #%s'%(iteration_number))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Iteration History for the Brachristochrone problem with %s Points'%(n))
    plt.savefig('brach_history' + '.pdf')

    #MATYAS
    function_calls = 0
    maty = uncon(matyas, xin_maty, 10**-6, options=None)
    #print maty
    print 'matyas function calls: ', np.size(mf)
    print 'matyas score: ', 1./np.size(mf)

    #plot the convergence of the Matyas
    plt.figure(1)
    plt.yscale('log')
    plt.xlim(0, np.max(maty[2][0]))
    plt.plot(maty[2][0], maty[2][1])
    plt.scatter(maty[2][0], maty[2][1])
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity Norm of the Gradient')
    plt.title('Convergence Plot of the Matyas Function Optimization')
    plt.savefig('maty_convergence' + '.pdf')

    #ROSENBROCK
    function_calls = 0
    ros = uncon(rosenbrock, xin_ros, 10**-6, options=None)
    #print ros
    print 'rosenbrock function calls: ', np.size(rf)
    print 'rosenbrock score: ', 1./np.size(rf)

    #Plot the convergence of the Rosenbrock
    plt.figure(2)
    plt.yscale('log')
    plt.xlim(0, np.max(ros[2][0]))
    plt.plot(ros[2][0], ros[2][1])
    plt.scatter(ros[2][0], ros[2][1])
    plt.xlabel('Major Iterations')
    plt.ylabel('Infinity Norm of the Gradient')
    plt.title('Convergence Plot of the Rosenbrock Function Optimization')
    plt.savefig('ros_convergence' + '.pdf')


    print 'total function calls: ', np.size(bf)+np.size(mf)+np.size(rf)
    print 'total score: ', 1./np.size(bf)+1./np.size(mf)+1./np.size(rf)


    #code from Dr. Ning to make a Contour plot with some small adjustments
    nx = 150  # number of points in x-direction
    ny = 150  # number of points in y-direction
    x = np.linspace(-4, 4, nx)  # nx points equally spaced between 13 and 27
    y = np.linspace(-4, 4, ny)  # ny points equally spaced between 13 and 26
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D array (matrix) of points across x and y
    Z = np.zeros((nx, ny))  # initialize output of size (nx, ny)

    # --- evaluate across grid ---
    for i in range(nx):
        for j in range(ny):
            Z[i, j] = matyas([X[i, j], Y[i, j]])[0]

    # --- contour plot ---
    plt.figure(4)  # start a new figure
    plt.contour(X, Y, Z, 300)  # using 300 contour lines.
    plt.colorbar()  # add a colorbar
    plt.xlabel('x')  # labels for axes
    plt.ylabel('y')
    plt.title('Iteration History for the Matyas Function Starting at %s'%(xin_maty))
    maty_plot1 = np.zeros(np.size(maty[2][0]))
    maty_plot2 = np.zeros(np.size(maty[2][0]))
    for i in range(0, np.size(maty[2][0])):
        maty_plot1[i]= maty[2][2][i][0]
        maty_plot2[i]= maty[2][2][i][1]
    plt.plot(maty_plot1, maty_plot2)
    plt.scatter(maty_plot1, maty_plot2)
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.savefig('maty_history' + '.pdf')


    #code from Dr. Ning to make a Contour plot with some small adjustments
    nx = 150  # number of points in x-direction
    ny = 150  # number of points in y-direction
    x = np.linspace(-4, 4, nx)  # nx points equally spaced between 13 and 27
    y = np.linspace(-4, 4, ny)  # ny points equally spaced between 13 and 26
    X, Y = np.meshgrid(x, y, indexing='ij')  # 2D array (matrix) of points across x and y
    Z = np.zeros((nx, ny))  # initialize output of size (nx, ny)

    # --- evaluate across grid ---
    for i in range(nx):
        for j in range(ny):
            Z[i, j] = rosenbrock([X[i, j], Y[i, j]])[0]
    plt.figure(5)  # start a new figure
    plt.contour(X, Y, Z, 300)  # using 30 contour lines.
    plt.colorbar()  # add a colorbar
    plt.xlabel('x')  # labels for axes
    plt.ylabel('y')
    plt.title('Iteration History for the Rosenbrock Function Starting at %s'%(xin_ros))
    ros_plot1 = np.zeros(np.size(ros[2][0]))
    ros_plot2 = np.zeros(np.size(ros[2][0]))
    for i in range(0, np.size(ros[2][0])):
        ros_plot1[i]= ros[2][2][i][0]
        ros_plot2[i]= ros[2][2][i][1]
    plt.plot(ros_plot1, ros_plot2)
    plt.scatter(ros_plot1, ros_plot2)
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.savefig('ros_history' + '.pdf')
    #plt.show()"""

