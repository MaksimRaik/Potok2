import numpy as np
from numpy.linalg import inv as inv
import matplotlib.pyplot as plt

#константы
Q = 2.94 * 10 ** 6
Ea = 71.0 * 10 ** 3
A = 10 ** 9
mu = 0.029
gamma = 1.25
h = 10 ** -7
R = 8.31
eps = 10 ** -6
Number = 1000

#начальные значения
p0 = 10.0 ** 5
T0 = 280
mu_mass = np.asarray( [ 26.0, 32.0, 28.0, 44.0, 18.0 ] ) * 10 ** -3
w = np.asarray( [ 0.07, 0.22, 0.713, 0.0, 0.0 ] )
ro0 = p0 * 1.0 / ( np.sum( w / mu_mass ) * R * T0 )
v0 = 0.0
Z0 = 1.0

def f1( ro, u, p, Z, c ):

    global f2

    return - ro / u * f2( ro, u, p, Z, c )

def f2( ro, u, p, Z, c ):

    global f4

    global Q, gamma

    return - Q * u * ( gamma - 1.0 ) / c ** 2 / ( 1.0 - u ** 2 / c ** 2 ) * f4( ro, u, p, Z )

def f3( ro, u, p, Z, c ):

    global f2

    return - ro * u * f2( ro, u, p, Z, c )

def f4( ro, u, p, Z ):

    global Ea, mu, A

    return - A * ro * Z / u * np.exp( - ro * Ea / p / mu )

def Fi( ro, u, p, Z, c, y1, y0 ):

    global f1, f2, f3, f4, h

    return  np.asarray( [ ( 3 * ro - 4 * y1[ 0 ] + y0[ 0 ] ) / 2.0 / h - f1( ro, u, p, Z, c ),
                        ( 3 * ro - 4 * y1[ 1 ] + y0[ 1 ] ) / 2.0 / h - f2( ro, u, p, Z, c ),
                        ( 3 * ro - 4 * y1[ 2 ] + y0[ 2 ] ) / 2.0 / h - f3( ro, u, p, Z, c ),
                        ( 3 * ro - 4 * y1[ 3 ] + y0[ 3 ] ) / 2.0 / h - f4( ro, u, p, Z ) ] )

def J( ro, u, p, Z, c ):

    global Ea, mu, A, Q, gamma, h

    alpha4 = - A * ro * Z / u * np.exp( - ro * Ea / p / mu )

    alpha2 = - Q * u * ( gamma - 1.0 ) / c ** 2 / ( 1.0 - u ** 2 / c ** 2 ) * alpha4

    alpha1 = -ro / u * alpha2

    alpha3 = -ro * u * alpha2

    J = []

    J.append( np.asarray( [ 3.0 / 2.0 / h + alpha1 * ( Ea / p / mu - 2.0 / ro ), alpha1 * ( 1 / u - 2 * u / c ** 2 / ( 1 - u ** 2 / c ** 2 ) ),
                         - alpha1 * ro * Ea / p ** 2 / mu, - alpha1 / Z ] ) )

    J.append(np.asarray( [ alpha2 * ( Ea / p / mu - 1.0 / ro ), 3.0 / 2.0 / h - alpha2 * 2 * u / c ** 2,
                         - alpha2 * ro * Ea / p ** 2 / mu, - alpha2 / Z] ) )

    J.append( np.asarray( [ alpha3 * ( Ea / p / mu - 2.0 / ro ), alpha3 * ( 2*u / c ** 2 - 1 / u ),
                         3.0 / 2.0 / h - alpha3 * ro * Ea / p ** 2 / mu, - alpha3 / Z] ) )

    J.append( np.asarray( [ alpha4 * ( Ea / p / mu - 1.0 / ro ), - alpha4 / u, - alpha4 * ro * Ea / p ** 2 / mu, 3.0 / 2.0 / h - alpha4 / Z] ) )

    return np.asarray( J )

def newtoon():

    global J, Fi

    global Ea, mu, A, Q, gamma, eps, p0, ro0, v0, h

    D_cj = np.sqrt( ( gamma ** 2 - 1.0 ) / 2 * Q + gamma * p0 / ro0 ) + np.sqrt( ( gamma ** 2 - 1.0 ) / 2 * Q )

    ro = []
    p = []
    u = []
    Z = []

    p.append( 2 * ro0 / ( gamma + 1.0 ) * D_cj ** 2 - ( gamma - 1.0 ) / ( gamma + 1.0 ) * p0 )

    u.append( D_cj - ( p[0] - p0 ) / np.sqrt( ro0 * ( (gamma + 1.0 ) * p[0] / 2.0 + ( gamma - 1.0 ) / 2 * p0 ) ) )

    ro.append( ro0 * D_cj / u[0] )

    Z.append( 1.0 )

    y0 = np.asarray( [ ro0, v0, p0, 1.0 ] )

    yx = np.asarray( [ ro[0], u[0], p[0], Z[0] ] )

    yx_1 = y0
    yx_0 = yx

    delta = np.sqrt( sum( ( ( yx_0 - yx_1 ) / yx_0 ) ** 2 ) )

    c = np.sqrt( gamma * y0[2] / y0[0] )

    for i in range(Number):

        while delta >= eps and ( c - yx[1] ) > 0.0 :

            c = np.sqrt( gamma * yx[2] / yx[0] )

            yx = yx - inv( J( yx[0], yx[1], yx[2], yx[3], c ) ).dot( Fi( yx[0], yx[1], yx[2], yx[3] , c, yx_0, yx_1 ) )

            yx_1 = yx_0
            yx_0 = yx

            delta = np.sqrt( sum( ( ( yx_0 - yx_1 ) / yx_0 ) ** 2 ) )

            print(yx, i)

        ro.append( yx[0] )
        u.append( yx[1] )
        p.append( yx[2] )
        Z.append( yx[3] )

        y0 = np.asarray( [ yx[0], yx[1], yx[2], yx[3] ] )

        D_cj = np.sqrt( ( gamma ** 2 - 1.0 ) / 2 * Q + gamma * y0[2] / y0[0] ) + np.sqrt( ( gamma ** 2 - 1.0 ) / 2 * Q )

        #print( D_cj, 'DD' )

        p.append( 2 * y0[0] / ( gamma + 1.0 ) * D_cj ** 2 - ( gamma - 1.0 ) / ( gamma + 1.0 ) * y0[2] )

        u.append( D_cj - ( p[-1] - y0[2] ) / np.sqrt( y0[0] * ( ( gamma + 1.0 ) * p[-1] / 2.0 + ( gamma - 1.0 ) / 2 * y0[2] ) ) )

        #print( D_cj - u[-1], 'uu' )

        ro.append( y0[0] * D_cj / u[-1] )

        Z.append( Z[-1] )

        yx = np.asarray( [ ro[-1], u[-1], p[-1], Z[-1] ] )

        yx_1 = y0
        yx_0 = yx

        print(yx)

        delta = np.sqrt( sum( ( ( yx_0 - yx_1 ) / yx_0 ) ** 2 ) )

    return ro, u, p, Z

def newton():


    class BaseLinearSolver:
        def solve(self, eq, x0, args=None, n_iter=1e3):
            raise NotImplemented()
    class BaseDifferentialSolver:
        def solve(self, x0, n_iter=1e3):
            raise NotImplemented()
    class BaseTimeDESolver(BaseDifferentialSolver):
        def __init__(self, dx, right_part, stop):
            self.dx = dx
            self.right_part = right_part
            self.stop = stop

        # вычисляет следующий слой по предыдущим
        def next_step(self, prev_u):
            raise NotImplemented()

        # выполяет начальную инициализацию первых слоев решения
        def init_values(self, x0):
            return [x0], 1

        # выполяет проверку на то, что нужно останавливаться
        def sto_p(self, prev_u):
            raise NotImplemented()

        def solve(self, u0, n_iter=10):
            f_values, num_inited = self.init_values(u0)
            for i in range(int(num_inited), int(n_iter)):
                val = self.next_step(f_values)
                f_values.append(val)

                if self.stop(f_values):
                    break

            return f_values
    class Iter_L_Solver(BaseLinearSolver):
        def __init__(self, eps=1e-5):
            self.eps = eps

        def solve(self, eq, x0, args=None, n_iter=10000):
            x = x0
            lam = np.array([1, -1, -1, -1])
            for _ in range(int(n_iter)):
                dx = eq(x)
                x = dx
                if np.linalg.norm(dx) < self.eps:
                    break
            return x
    class Geer2DESolver(BaseTimeDESolver):
        def __init__(self, dx, right_part, le_solver, stop):
            super().__init__(dx, right_part, stop)
            self.le_solver = le_solver

        def init_values(self, u0):
            f = lambda u: u0 + self.dx * self.right_part(u)
            u1 = self.le_solver.solve(f, u0)
            return [
                       u0,
                       u1
                   ], 2

        def next_step(self, prev_u):
            f = lambda u: (4 * prev_u[-1] - prev_u[-2]) / 3 + (2 * self.dx) / 3 * self.right_part(u)
            return self.le_solver.solve(f, prev_u[-1])

        def sto_p(self, prev_u):
            return stop(prev_u)
    class GasTask:
        def __init__(self, gamma, mu, Q, Ea, A):
            self.gamma = gamma
            self.mu = mu
            self.Q = Q
            self.Ea = Ea
            self.A = A

        def f(self, params):
            ro, u, p, Z = params

            RT = self.mu * p / ro
            c2 = self.gamma * p / ro

            res_f = np.empty((4,), dtype=np.float64)
            res_f[0] = - ro * self.Q * (self.gamma - 1) / (c2 - u ** 2) * self.A * ro * Z / u * np.exp(
                -self.Ea * ro / (p * self.mu))
            res_f[1] = self.Q * (self.gamma - 1) / (c2 - u ** 2) * self.A * ro * Z * np.exp(
                -self.Ea * ro / (p * self.mu))
            res_f[2] = - ro * self.Q * (self.gamma - 1) / (c2 - u ** 2) * self.A * ro * Z * u * np.exp(
                -self.Ea * ro / (p * self.mu))
            res_f[3] = - self.A * ro * Z / u * np.exp(-self.Ea * ro / (p * self.mu))
            return res_f

        def f0(self, params0):
            ro0, u0, p0, Z0 = params0

            Dcj = np.sqrt((self.gamma ** 2 - 1) / 2 * self.Q + self.gamma * p0 / ro0) + np.sqrt(
                (self.gamma ** 2 - 1) / 2 * self.Q)

            res_f0 = np.empty((4,), dtype=np.float64)
            res_f0[2] = 2 * ro0 / (self.gamma + 1) * Dcj ** 2 - (self.gamma - 1) / (self.gamma + 1) * p0
            res_f0[1] = Dcj - (res_f0[2] - p0) / np.sqrt(
                ro0 * ((self.gamma + 1) / 2 * res_f0[2] + (self.gamma - 1) / 2 * p0))
            res_f0[0] = ro0 * Dcj / res_f0[1]
            res_f0[3] = 1

            return res_f0

        def stop(self, f_values, eps=1e-5):
            params = f_values[-1]
            ro, u, p, Z = params
            c2 = self.gamma * p / ro
            return np.abs(u ** 2 - c2) < eps

        def getCJparams(self, params0):
            ro0, u0, p0, Z0 = params0
            Dcj = np.sqrt((self.gamma ** 2 - 1) / 2 * self.Q + self.gamma * p0 / ro0) + np.sqrt(
                (self.gamma ** 2 - 1) / 2 * self.Q)

            p_cj = p0 / (self.gamma + 1) + ro0 / (self.gamma + 1) * Dcj ** 2
            ro_cj = (self.gamma + 1) / self.gamma * ro0 ** 2 * Dcj ** 2 / (p0 + ro0 * Dcj ** 2)
            T_cj = p_cj * self.mu / (ro_cj * R)
            return {
                'p_cj': p_cj,
                'ro_cj': ro_cj,
                'T_cj': T_cj,
                'D_cj': Dcj
            }

    global Ea, mu, A, Q, gamma, eps, p0, ro0, v0, h, Number

    le_solver = Iter_L_Solver()

    task = GasTask(gamma, mu, Q, Ea, A)

    fii = task.f0((ro0, v0, p0, 1.0))

    de_solver = Geer2DESolver( h, task.f, le_solver, task.stop )

    result = de_solver.solve( fii, n_iter = Number )

    ro = []
    u = []
    p = []
    Z = []

    for i in result:

        ro.append( i[0] )
        u.append( i[1] )
        p.append( i[2] )
        Z.append( i[3] )

    return np.asarray( ro ) ,np.asarray( u ), np.asarray( p ), np.asarray( Z )

def draw( name, size ):

    plt.figure( figsize = size )
    plt.grid()
    plt.xlabel( name[ 0 ] )
    plt.ylabel( name[ 1 ] )

def CJparametr(ro, u, p, Z):

    global gamma, mu,R, Q

    D_cj = np.sqrt( ( gamma ** 2 - 1 ) / 2 * Q + gamma * p / ro ) + np.sqrt( ( gamma ** 2 - 1 ) / 2 * Q )

    p_cj = p / ( gamma + 1  ) + ro0 / (gamma + 1) * D_cj ** 2

    ro_cj = (gamma + 1) / gamma * ro ** 2 * D_cj ** 2 / ( p + ro * D_cj ** 2)

    T_cj = p_cj * mu / (ro_cj * R)

    print( 'D_cj=',D_cj, 'p_cj=',p_cj, 'ro_cj=',ro_cj, 'T_cj=',T_cj )









