import numpy as np
import matplotlib.pyplot as plt

def monomialcoeffs(xi, fi):
    A = np.vander(xi,increasing=True)
    return  np.linalg.solve(A,fi), np.linalg.cond(A, np.inf)

def monomialinterpol(x,p):
    y = np.zeros(len(x))
    for i in range(len(p)):
        y = y + p[i]*np.power(x, i)
    return y

def Lk_x(k,xi,x):
    n = len(xi)
    y = np.ones(len(x))
    for i in range(n):
        if i != k:
            y = np.multiply(y,(x - np.ones(len(x))*xi[i])/(xi[k]-xi[i]))
    return y


def lagrangeinterpol(x,p,xi):
    y = np.zeros(len(x))
    for k in range(len(xi)):
        y = y + Lk_x(k,xi,x)*p[k]
    return y
    


def f(x,i):
    return sum([x**j for j in range(i)])

f = np.vectorize(f)

errors = []
vandermonde_conditions = []
for i in range(1,201):
    xi = np.linspace(0,1,num=i)
    fi = f(xi,i)
    p = monomialcoeffs(xi,fi)
    err_vec = p[0] - np.ones(i)
    errors.append(np.linalg.norm(err_vec, np.inf))
    vandermonde_conditions.append(p[1])
    
x = np.arange(1,201)
errors = np.array(errors)
vandermonde_conditions = np.array(vandermonde_conditions)

plt.figure()
plt.plot(x,errors)
plt.title('errors in coefficients')
plt.loglog()
plt.show()

plt.figure()
plt.plot(x,vandermonde_conditions)
plt.title('vandermonde matrix conditions')
plt.loglog()
plt.show()




xi_test = np.linspace(0,np.pi,1000)
sin_test = np.sin(xi_test)

ns = np.array([10,20,40,80,200])
error_monomial = [] 
error_lagrange = []

for i in [10,20,40,80,200]:
    xi = np.linspace(0,np.pi,num=i)
    sin_fi = np.sin(xi)
    sin_p = monomialinterpol(xi_test,monomialcoeffs(xi,sin_fi)[0])
    sin_lagrange = lagrangeinterpol(xi_test,sin_fi,xi)

    error_monomial.append(np.linalg.norm(sin_p - sin_test,np.inf))
    error_lagrange.append(np.linalg.norm(sin_lagrange - sin_test, np.inf))


    plt.figure()
    plt.plot(xi_test,sin_p, label="interpolating polynomial")
    #plt.plot(xi_test,sin_lagrange, label="lagrange interpolating poly")
    plt.plot(xi_test,sin_test, label="sine")
    plt.legend()
    plt.title('sin with i = ' + str(i))
    plt.show()


plt.figure()
plt.plot(ns, np.array(error_monomial))
plt.title('error monomial')
plt.loglog()
plt.show()


plt.figure()
plt.plot(ns, np.array(error_lagrange))
plt.title('error lagrange')
plt.loglog()
plt.show()


