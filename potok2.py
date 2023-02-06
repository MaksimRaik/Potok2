import matplotlib.pyplot as plt
import numpy as np
import function_potok2 as f



#решение

ro, u, p, Z =  f.newtoon()

print( f.ro0 )

T = np.asarray( p ) * 1 / ( f.R * np.asarray( ro ) * sum( f.w / f.mu_mass ) )

x = np.asarray( range( f.Number ) ) * f.h

f.CJparametr( f.ro0, f.v0, f.p0, 1.0 )

#посторение графиков

f.draw([r'$ x $', r'$ \rho $'], (12, 10))

plt.plot( x, ro, 'b-', label = r'$ \rho (x) $' )



f.draw([ r'$ x $', r'$ p $'], (12, 10))

plt.plot( x, p, 'g-', label = r'$ p(x) $' )


f.draw([r'$ x $', r'$ p $'], (12, 10))

plt.plot( x, u, 'r-', label = r'$ u(x) $' )


f.draw([r'$ x $', r'$ T(x) $'], (12, 10))

plt.plot( x, T, 'k-', label = r'$ T(x) $' )



legend = plt.legend( loc = 'upper right', shadow = True, fontsize = 'x-large' )

plt.show()



