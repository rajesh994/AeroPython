# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:52:37 2018

@author: rajesh994
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 18 17:42:29 2018

@author: rajesh994
"""
import csv
import os
import numpy
import math
from scipy import integrate, linalg
from matplotlib import pyplot


x1, y1 = numpy.genfromtxt('MainFoil_N=100.csv', dtype=float, delimiter=',', unpack=True)


x2, y2 = numpy.genfromtxt('FlapFoil_N=100.csv', dtype=float, delimiter=',', unpack=True)

#x1 = numpy.loadtxt('mainfoil.txt', dtype=float, delimiter='\t', unpack=True)
#y1 = numpy.loadtxt('mainfoil_y.txt', dtype=float, delimiter='\t', unpack=True)

#x2 = numpy.loadtxt('flapfoil_x.txt', dtype=float, delimiter='\t', unpack=True)
#y2 = numpy.loadtxt('flapfoil_y.txt', dtype=float, delimiter='\t', unpack=True)

x=numpy.append(x1,x2)
y=numpy.append(y1,y2)

Na=100

Nb=100

# plot the geometry
width = 10
pyplot.figure(figsize=(width, width))
pyplot.grid()
pyplot.xlabel('x', fontsize=16)
pyplot.ylabel('y', fontsize=16)
pyplot.plot(x1, y1, color='k', linestyle='-', linewidth=2)
pyplot.plot(x2, y2, color='k', linestyle='-', linewidth=2)
pyplot.axis('scaled', adjustable='box')

class Panel:
    """
    Contains information related to a panel.
    """
    def __init__(self, xa, ya, xb, yb):
        """
        Initializes the panel.
        
        Sets the end-points and calculates the center-point, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is located on the upper or lower surface of the geometry.
        Initializes the source-strength, tangential velocity, and pressure coefficient
        of the panel to zero.
        
        Parameters
        ---------_
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya  # panel starting-point
        self.xb, self.yb = xb, yb  # panel ending-point
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # panel center
        self.length = numpy.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length
        
        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = numpy.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = numpy.pi + numpy.arccos(-(yb - ya) / self.length)
        
        # panel location
        if self.beta <= numpy.pi:
            self.loc = 'upper'  # upper surface
        else:
            self.loc = 'lower'  # lower surface
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient
        self.gamma=0.0
        
# create panels
panels = numpy.empty(Na+Nb, dtype=object)
for i in range(Na):
    panels[i] = Panel(x1[i], y1[i], x1[i + 1], y1[i + 1])
    
    

for i in range(Na,Na+Nb):
    panels[i] = Panel(x2[i-Na], y2[i-Na], x2[i + 1-Na], y2[i + 1-Na])
   

class Freestream:
    """
    Freestream conditions.
    """
    def __init__(self, u_inf=1.0, alpha=0.0):
        """
        Sets the freestream speed and angle (in degrees).
        
        Parameters
        ----------
        u_inf: float, optional
            Freestream speed;
            default: 1.0.
        alpha: float, optional
            Angle of attack in degrees;
            default 0.0.
        """
        self.u_inf = u_inf
        self.alpha = numpy.radians(alpha)  # degrees to radians
# define freestream conditions
freestream = Freestream(u_inf=1.0, alpha=0.0)

def integral(x, y, panel, dxdk, dydk):
    """
    Evaluates the contribution from a panel at a given point.
    
    Parameters
    ----------
    x: float
        x-coordinate of the target point.
    y: float
        y-coordinate of the target point.
    panel: Panel object
        Panel whose contribution is evaluated.
    dxdk: float
        Value of the derivative of x in a certain direction.
    dydk: float
        Value of the derivative of y in a certain direction.
    
    Returns
    -------
    Contribution from the panel at a given point (x, y).
    """
    def integrand(s):
        return (((x - (panel.xa - numpy.sin(panel.beta) * s)) * dxdk +
                 (y - (panel.ya + numpy.cos(panel.beta) * s)) * dydk) /
                ((x - (panel.xa - numpy.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + numpy.cos(panel.beta) * s))**2) )
    return integrate.quad(integrand, 0.0, panel.length)[0]


def source_contribution_normal(panels):
    """
    Builds the source contribution matrix for the normal velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        Source contribution matrix.
    """
    A = numpy.empty((panels.size, panels.size), dtype=float)
    # source contribution on a panel from itself
    numpy.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, 
                                                    panel_j,
                                                    numpy.cos(panel_i.beta),
                                                    numpy.sin(panel_i.beta))
    return A

def vortex_contribution_normal(panels):
    """
    Builds the vortex contribution matrix for the normal velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        Vortex contribution matrix.
    """
    A = numpy.empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    numpy.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / numpy.pi * integral(panel_i.xc, panel_i.yc, 
                                                     panel_j,
                                                     numpy.sin(panel_i.beta),
                                                     -numpy.cos(panel_i.beta))
    return A

A_source = source_contribution_normal(panels)
B_vortex = vortex_contribution_normal(panels)


def kutta_condition(A_source, B_vortex):
    """
    Builds the Kutta condition array.
    
    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    
    Returns
    -------
    b: 1D Numpy array of floats
        The left-hand side of the Kutta-condition equation.
    """
    b = numpy.empty((2,A_source.shape[0] + 2), dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    b[0,:-2] = B_vortex[0, :] + B_vortex[Na-1, :]
    b[1,:-2] = B_vortex[Na, :] + B_vortex[Na+Nb-1, :]
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    b[0,-2] = - numpy.sum(A_source[0, 0:Na] + A_source[Na-1, 0:Na])
    
    b[0,-1] = - numpy.sum(A_source[0, Na:Na+Nb] + A_source[Na-1, Na:Na+Nb])
    
    b[1,-2] = - numpy.sum(A_source[Na, 0:Na] + A_source[Na+Nb-1, 0:Na])
    
    b[1,-1] = - numpy.sum(A_source[Na, Na:Na+Nb] + A_source[Na+Nb-1, Na:Na+Nb])
    return b

def build_singularity_matrix(A_source, B_vortex):
    """
    Builds the left-hand side matrix of the system
    arising from source and vortex contributions.
    
    Parameters
    ----------
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    
    Returns
    -------
    A:  2D Numpy array of floats
        Matrix of the linear system.
    """
    A = numpy.empty((A_source.shape[0] + 2, A_source.shape[1] + 2), dtype=float)
    # source contribution matrix
    A[:-2, :-2] = A_source
    # vortex contribution array
    A[:-2, -2] = numpy.sum(B_vortex[:,0:Na], axis=1)
    A[:-2, -1] = numpy.sum(B_vortex[:,Na:Na+Nb], axis=1)
    # Kutta condition array
    A[-2:, :] = kutta_condition(A_source, B_vortex)
    return A


def build_freestream_rhs(panels, freestream):
    """
    Builds the right-hand side of the system 
    arising from the freestream contribution.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    
    Returns
    -------
    b: 1D Numpy array of floats
        Freestream contribution on each panel and on the Kutta condition.
    """
    b = numpy.empty(panels.size + 2, dtype=float)
    # freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    # freestream contribution on the Kutta condition
    b[-2] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels[0].beta) +
                                 numpy.sin(freestream.alpha - panels[Na-1].beta) )
    b[-1] = -freestream.u_inf * (numpy.sin(freestream.alpha - panels[Na].beta) +
                                 numpy.sin(freestream.alpha - panels[Na+Nb-1].beta) )
    return b

A = build_singularity_matrix(A_source, B_vortex)
b = build_freestream_rhs(panels, freestream)

strengths = numpy.linalg.solve(A, b)

for i , panel in enumerate(panels):
    panel.sigma = strengths[i]
    if i<Na:
        panel.gamma=strengths[-2]
    else:
        panel.gamma=strengths[-1]
    
gammaa = strengths[-2]
gammab =strengths[-1]

def compute_tangential_velocity(panels, freestream, strengths, A_source, B_vortex):
    """
    Computes the tangential surface velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    gamma: float
        Circulation density.
    A_source: 2D Numpy array of floats
        Source contribution matrix for the normal velocity.
    B_vortex: 2D Numpy array of floats
        Vortex contribution matrix for the normal velocity.
    """
    A = numpy.empty((panels.size, panels.size + 2), dtype=float)
    # matrix of source contribution on tangential velocity
    # is the same than
    # matrix of vortex contribution on normal velocity
    A[:, :-2] = B_vortex
    # matrix of vortex contribution on tangential velocity
    # is the opposite of
    # matrix of source contribution on normal velocity
    A[:, -2] = -numpy.sum(A_source[:,0:Na], axis=1)
    A[:, -1] = -numpy.sum(A_source[:,Na:Na+Nb], axis=1)
    # freestream contribution
    b = freestream.u_inf * numpy.sin([freestream.alpha - panel.beta 
                                      for panel in panels])
    
   # strengths = numpy.append([panel.sigma for panel in panels], gamma)
    
    tangential_velocities = numpy.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]
        
# tangential velocity at each panel center.
compute_tangential_velocity(panels, freestream, strengths, A_source, B_vortex)

def compute_pressure_coefficient(panels, freestream):
    """
    Computes the surface pressure coefficients.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    freestream: Freestream object
        Freestream conditions.
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2
# surface pressure coefficient
compute_pressure_coefficient(panels, freestream)




# plot surface pressure coefficient
pyplot.figure(figsize=(10, 6))
pyplot.grid()
pyplot.xlabel('$x$', fontsize=16)
pyplot.ylabel('$C_p$', fontsize=16)
pyplot.plot([panel.xc for panel in panels],
            [panel.cp for panel in panels],
            label='upper surface',
            color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
pyplot.plot([panel.xc for panel in panels if panel.loc == 'lower'],
            [panel.cp for panel in panels if panel.loc == 'lower'],
            label= 'lower surface',
            color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
pyplot.legend(loc='best', prop={'size':16})
pyplot.title('Number of panels: {}'.format(panels.size), fontsize=16);

# compute the chord and lift coefficient
panelmain=panels[0:Na]
panelflap=panels[Na:Na+Nb]

c = abs(max(panel.xa for panel in panelmain) -
        min(panel.xa for panel in panelmain))
cl = ( sum(panel.gamma*panel.length for panel in panels) /
      (0.5 * freestream.u_inf * c))
print('lift coefficient: CL = {:0.3f}'.format(cl))