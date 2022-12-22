# from decimal import Decimal
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors
import numpy as np
import ffmpeg

class BaseIntegrator:

    def __init__(self, dt) :
        self.dt = dt   # time step

    def integrate(self, system):
        """ Perform a single integration step """
        for b in system.bodies:
            self.timestep(system, b)

            # Calculate kinetic and total energy
            b.E_kin.append(system.kinetic_energy(b))
            b.E_pot.append(system.potential_energy(b))
            b.E_tot.append(b.E_kin[-1] + b.E_pot[-1])

            if b.name != 'Sun':
                b.dist_to_sun.append((b.x[-1]**2 + b.y[-1]**2) ** 0.5)

                if abs(b.y[-1]) < 0.5*b.v_y*self.dt and b.x[-1] > 0:
                    b.poincare_x.append(b.x[-1])
                    #b.poincare_p.append(b.mass * b.v_y)
                    b.poincare_p.append(b.mass * (b.v_x ** 2 + b.v_y ** 2) ** 0.5)
        system.t.append(system.t[-1] + self.dt)


    def timestep(self, system):
        """ Virtual method: implemented by the child classes """
        pass


class SymplecticEulerIntegrator(BaseIntegrator): # Euler-Cromer
    def timestep(self, system, b):
        f = system.force(b)

        # Update velocity and position
        a_x = f[0] / b.mass
        a_y = f[1] / b.mass
        b.v_x += a_x * self.dt
        b.v_y += a_y * self.dt
        b.x.append(b.x[-1] + b.v_x * self.dt)
        b.y.append(b.y[-1] + b.v_y * self.dt)


class VerletIntegrator(BaseIntegrator):
    def timestep(self, system, b):
        f1 = system.force(b)
        a_x1 = f1[0] / b.mass
        a_y1 = f1[1] / b.mass
        b.x.append(b.x[-1] + b.v_x * self.dt + 0.5 * a_x1 * self.dt ** 2)
        b.y.append(b.y[-1] + b.v_y * self.dt + 0.5 * a_y1 * self.dt ** 2)

        f2 = system.force(b)
        a_x2 = f2[0] / b.mass
        a_y2 = f2[1] / b.mass
        b.v_x += 0.5 * (a_x1 + a_x2) * self.dt
        b.v_y += 0.5 * (a_y1 + a_y2) * self.dt


class RK4Integrator(BaseIntegrator):
    def timestep(self, system, b):
        x = b.x[-1]
        y = b.y[-1]
        v_x = b.v_x
        v_y = b.v_y
        b.x.append(x)
        b.y.append(y)
        
        f1 = system.force(b)
        a_x1 = self.dt * f1[0] / b.mass
        a_y1 = self.dt * f1[1] / b.mass
        b_x1 = v_x * self.dt
        b_y1 = v_y * self.dt
        b.x[-1] = x + 0.5 * b_x1
        b.y[-1] = y + 0.5 * b_y1
        b.v_x = v_x + 0.5 * a_x1
        b.v_y = v_y + 0.5 * a_y1

        f2 = system.force(b)
        a_x2 = self.dt * f2[0] / b.mass
        a_y2 = self.dt * f2[1] / b.mass
        b_x2 = (v_x + 0.5 * a_x1) * self.dt
        b_y2 = (v_y + 0.5 * a_y1) * self.dt
        b.x[-1] = x + 0.5 * b_x2
        b.y[-1] = y + 0.5 * b_y2
        b.v_x = v_x + 0.5 * a_x2
        b.v_y = v_y + 0.5 * a_y2

        f3 = system.force(b)
        a_x3 = self.dt * f3[0] / b.mass
        a_y3 = self.dt * f3[1] / b.mass
        b_x3 = (v_x + 0.5 * a_x2) * self.dt
        b_y3 = (v_y + 0.5 * a_y2) * self.dt
        b.x[-1] = x + b_x3
        b.y[-1] = y + b_y3
        b.v_x = v_x + a_x3
        b.v_y = v_y + a_y3

        f4 = system.force(b)
        a_x4 = self.dt * f4[0] / b.mass
        a_y4 = self.dt * f4[1] / b.mass
        b_x4 = (v_x + a_x3) * self.dt
        b_y4 = (v_y + a_y3) * self.dt

        b.v_x = v_x + (1 / 6) * (a_x1 + 2 * a_x2 + 2 * a_x3 + a_x4)
        b.v_y = v_y + (1 / 6) * (a_y1 + 2 * a_y2 + 2 * a_y3 + a_y4)
        b.x[-1] = x + (1 / 6) * (b_x1 + 2 * b_x2 + 2 * b_x3 + b_x4)
        b.y[-1] = y + (1 / 6) * (b_y1 + 2 * b_y2 + 2 * b_y3 + b_y4)


class SolarSystem:

    def __init__(self, dt, nsteps):
        self.bodies = []
        self.AU = 1.496e11
        self.G = 6.67428e-11
        self.daytosec = 24*60*60

        self.t = [0]
        self.dt = dt*self.daytosec
        self.nsteps = nsteps

    def add_body(self, body):
        self.bodies.append(body)

    def remove_body(self, body):
        for i in range(len(self.bodies)):
            if self.bodies[i] == body:
                self.bodies.pop(i)

    def force(self, body):
        """Total force and potential energy for body (x and y component)"""
        f = [0, 0]
        for b in self.bodies:
            if b == body:
                continue

            r_x = body.x[-1] - b.x[-1]
            r_y = body.y[-1] - b.y[-1]
            modr3 = (r_x**2 + r_y**2)**1.5
            
            f[0] += -self.G * body.mass * b.mass * r_x / modr3
            f[1] += -self.G * body.mass * b.mass * r_y / modr3

        return f

    def kinetic_energy(self, body):
        return 0.5 * body.mass * (body.v_x ** 2 + body.v_y ** 2)

    def potential_energy(self, body):
        E_pot = 0
        for b in self.bodies:
            if b == body:
                continue
            r_x = body.x[-1] - b.x[-1]
            r_y = body.y[-1] - b.y[-1]
            r = (r_x **2 + r_y ** 2) ** 0.5

            E_pot += -self.G * body.mass * b.mass / r
        
        return E_pot

    def run(self, integrator):
        while self.t[-1] < self.dt * self.nsteps:
            integrator.integrate(self)
        print('Data ready')

    def update(self, i):
        for j in range(len(self.bodies)):
            self.xdata[j].append(self.bodies[j].x[i])
            self.ydata[j].append(self.bodies[j].y[i])

            if i > self.bodies[j].tail_length:
                self.lines[j].set_data(self.xdata[j][-int(self.bodies[j].tail_length):], self.ydata[j][-int(self.bodies[j].tail_length):])
            else:
                self.lines[j].set_data(self.xdata[j], self.ydata[j])
            self.points[j].set_data(self.bodies[j].x[i], self.bodies[j].y[i])
            if self.show_text or self.show_dist:
                if self.show_dist and self.bodies[j].name != 'Sun':
                    self.text[j].set_text(str(int(self.bodies[j].dist_to_sun[i] / 1000)) + ' km')
                self.text[j].set_position([self.bodies[j].x[i], self.bodies[j].y[i]])

    def animate(self, lim, show_text = True, show_dist = False, last = False, save = False, title = 'mymovie', fps = 10):
        matplotlib.rcParams['animation.embed_limit'] = 2**128
        # plt.style.use('dark_background')
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111, facecolor = 'k')
        plt.rcParams.update({'font.size': 16})
        ax.set_xlim(-lim*self.AU,lim*self.AU)
        ax.set_ylim(-lim*self.AU,lim*self.AU)
        ax.set_aspect('equal')
        #ax.grid()

        self.show_text = show_text
        self.show_dist = show_dist
        if last:
            end = 0
        else:
            end = 1

        self.lines = []
        self.points = []
        self.text = []
        self.xdata = []
        self.ydata = []
        for i in range(len(self.bodies)):
            line, = (ax.plot([],[],self.bodies[i].color,lw=1))
            point, = (ax.plot([self.bodies[i].x[0]], [self.bodies[i].y[0]], marker="o", markersize=self.bodies[i].size, markeredgecolor=self.bodies[i].color, markerfacecolor=self.bodies[i].color))
            if self.show_text or self.show_dist and self.bodies[i].name == 'Sun':
                text = (ax.text(self.bodies[i].x[0],self.bodies[i].y[0],self.bodies[i].name,c='w'))
                self.text.append(text)
            elif self.show_dist and self.bodies[i].name != 'Sun':
                text = (ax.text(self.bodies[i].x[0],self.bodies[i].y[0], str(int(self.bodies[i].dist_to_sun[0] / 1000)) + ' km',c='w'))
                self.text.append(text)
            self.lines.append(line)
            self.points.append(point)
            self.xdata.append([])
            self.ydata.append([])

        anim = animation.FuncAnimation(fig,func=self.update,frames=self.nsteps,interval=end*1,blit=False,repeat=False)
        if save:
            mywriter = animation.FFMpegWriter(fps=fps)
            anim.save('/home/elliot/simmod/final_project/graphics/' + title + '.mp4', writer=mywriter)
            print('Animation saved as ' + title + '.mp4')
        else:
            plt.show()

    def plot_energy(self, object):
        valid = False
        for b in self.bodies:
            if b.name == object:
                object = b
                valid = True

        if valid:
            plt.figure(figsize=(8,7))
            plt.rcParams.update({'font.size': 16})
            plt.plot(self.t[:-1], object.E_kin)
            plt.plot(self.t[:-1], object.E_pot)
            plt.plot(self.t[:-1], object.E_tot)
            plt.legend(('E_kin','E_pot','E_tot'), loc = 1)
            plt.tight_layout()
            plt.grid()
            plt.show()
        else:
            raise ValueError('Input astronomical object is not in the system')

    def poincare(self, body):
        valid = False
        for b in self.bodies:
            if b.name == body:
                body = b
                valid = True
       
        if valid:
            plt.figure()
            plt.rcParams.update({'font.size': 16})
            plt.xlabel('x')
            plt.ylabel('p')
            plt.plot(body.poincare_x, body.poincare_p, 'ko', markersize = 2)
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError('Input astronomical object is not in the system')


class Body:

    def __init__(self, name, mass, size, tail_length, color):
        self.x = []
        self.y = []
        self.E_kin = []
        self.E_pot = []
        self.E_tot = []

        self.poincare_x = []
        self.poincare_p = []

        self.name = name
        self.mass = mass
        self.tail_length = tail_length    # sidereal orbit period
        self.size = size
        self.color = color


class Sun(Body):

    def __init__(self, name = 'Sun', mass = 1.9885e30, size = 20, tail_length = 1000, color = 'y'):
        super().__init__(name, mass, size, tail_length, color)
        self.x.append(0)
        self.y.append(0)
        self.v_x = 0
        self.v_y = 0


class Planet(Body):

    def __init__(self, name, mass, size, tail_length, color, dist_to_sun, velocity):
        super().__init__(name, mass, size, tail_length, color)
        self.dist_to_sun = []
        self.dist_to_sun.append(dist_to_sun)
        self.x.append(dist_to_sun)
        self.y.append(0)
        self.v_x = 0
        self.v_y = velocity


class Asteroid(Body):

    def __init__(self, name, mass, size, tail_length, color, x, y, v_x, v_y):
        super().__init__(name, mass, size, tail_length, color)
        self.dist_to_sun = []
        self.dist_to_sun.append((x**2 + y**2)**0.5)
        self.x.append(x)
        self.y.append(y)
        self.v_x = v_x
        self.v_y = v_y


def main():
    # Controls
    nsteps = 1000000
    dt = 1 # multiple of a day
    integrator = 'RK4' # Choice of integrator ('Symplectic'/'Verlet'/'RK4')
    lim = 2 # limits of axis in AU
    show_text = False # display names in simulation
    show_dist = False # display distance to the Sun in simulation
    last = False # shows only last frame of simulation if True
    
    pl_energy = True # plot energies of object_to_study
    object_to_study = 'Earth'

    anim = False # show animation of simulation
    save = False # save animation as mp4
    title = 'mymovie' # name of saved mp4
    fps = 100 # frame rate (playback speed) of saved animation

    poincare = True # plot Poincaré map for poincare_body
    poincare_body = 'Earth'


    tail_scale = 1 / dt
    size_scale = 1 / (0.4*lim)
    system = SolarSystem(dt=dt, nsteps=nsteps)
    sun = Sun(size=size_scale*20)
    # Start in aphelion
    mercury = Planet('Mercury', 3.3010e23, size_scale*4, tail_scale*120, 'lightcoral', 0.459*system.AU, 42860)
    venus = Planet('Venus', 4.8673e24, size_scale*7, tail_scale*225, 'w', 0.716*system.AU, 34790)
    earth = Planet('Earth', 5.9722e24, size_scale*8, tail_scale*365, 'b', system.AU, 29290)
    mars = Planet('Mars', 6.4174e23, size_scale*5, tail_scale*750, 'r', 1.639*system.AU, 22970)
    jupiter = Planet('Jupiter', 1.8981e27, size_scale*10, tail_scale*4200, 'orange', 5.367*system.AU, 12440)
    saturn = Planet('Saturn', 5.6832e26, size_scale*9, tail_scale*11500, 'tan', 9.905*system.AU, 9490)
    uranus = Planet('Uranus', 8.6811e25, size_scale*9, tail_scale*32000, 'skyblue', 19.733*system.AU, 6740)
    neptune = Planet('Neptune', 1.0241e26, size_scale*8, tail_scale*60200, 'orchid', 29.973*system.AU, 5480)
    pluto = Planet('Pluto', 1.3030e22, size_scale*2, tail_scale*130000, 'goldenrod', 48.023*system.AU, 4310)

    # asteroid = Asteroid('Asteroid', 1.9885e19, 2, tail_scale*5000, 'gray', 2*system.AU, system.AU, -5000, -18000)

    system.add_body(sun)
    system.add_body(mercury)
    system.add_body(venus)
    system.add_body(earth)
    system.add_body(mars)
    system.add_body(jupiter)
    system.add_body(saturn)
    system.add_body(uranus)
    system.add_body(neptune)
    system.add_body(pluto)

    # system.add_body(asteroid)
    
    if integrator == 'Symplectic':
        integrator = SymplecticEulerIntegrator(system.dt)
    elif integrator == 'Verlet':
        integrator = VerletIntegrator(system.dt)
    elif integrator == 'RK4':
        integrator = RK4Integrator(system.dt)
    else:
        raise ValueError('Invalid integrator')
    system.run(integrator)

    if pl_energy:
        system.plot_energy(object_to_study)
    if anim or save:
        system.animate(lim, show_text, show_dist, last, save, title, fps)
    if poincare:
        system.poincare(poincare_body)

def color_test():
    print(mcolors.CSS4_COLORS)

def main2():
    sys = SolarSystem(1, 10000)
    sun1 = Planet('Sun1', 1.9885e30, 20, 1000, 'y', sys.AU, 15000)
    sun2 = Planet('Sun2', 1.9885e30, 20, 1000, 'y', -sys.AU, -15000)
    tatooine = Planet('Tatooine', 5.9722e24, 8, 1500, 'tan', 4*sys.AU, 21290)
    sys.add_body(sun1)
    sys.add_body(sun2)
    sys.add_body(tatooine)

    sys.run()

    sys.animate(5)

def poincare_main():
    nsteps = 1000000
    dt = 1 # multiple of a day
    lim = 2 # limits of axis in AU
    
    tail_scale = 1 / dt
    size_scale = 1 / (0.4*lim)
    system = SolarSystem(dt=dt, nsteps=nsteps)
    sun = Sun(size=size_scale*20)
    earth = Planet('Earth', 5.9722e24, size_scale*8, tail_scale*365, 'b', system.AU, 29290)
    system.add_body(sun)
    system.add_body(earth)

    integrator = RK4Integrator(system.dt)
    system.run(integrator)
    
    print(len(earth.poincare_x))
    system.plot_energy('Earth')
    system.poincare('Earth')

if __name__ == "__main__" :
    #main()
    #main2()
    poincare_main()
    #color_test()