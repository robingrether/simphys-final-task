import numpy
import scipy.stats
from matplotlib import pyplot, animation
from IPython.display import HTML
from tqdm.notebook import tqdm
%matplotlib inline

class MDBox:
    """Rectangular box for simulating Lennard-Jones particles."""
    # constants
    R = 8.314462618e3 #   g * (m/s)*^2 / (mol * K)
    N_A = 6.02214076e23 # 1 / mol
    C12 = 9.847044e-6 #   kJ * nm^12 / mol
    C6 = 6.2647225e-3 #   kJ * nm^6 / mol
    K_E = 138.9354576 #   kJ * nm / (mol * e^2)
    # units:
    # mass in:            g / mol
    # length in:          nm
    # time in:            ns
    # velocity in:        nm / ns
    # acceleration in:    nm / (ns^2)
    # force in:           nm / (ns^2) * g / mol = 10^(-6) * kJ / (nm * mol)
    # energy in:          kJ / mol
    # charge in:          e = 1.602 * 10^(-19) C
    
    def __init__(self, xsize=50, ysize=50):
        """Initialize the rectangular box with the given size."""
        self.xsize = xsize
        self.ysize = ysize
    
    # ========== Start: Initialize particles ==========
    
    def temperature_particles(self, N=1, m=1, q=0, T=300, min_dist=(1/3)):
        """Delete all existing particles and randomly sample N new particles with velocities according to the Maxwell-Boltzmann statistics."""
        # initialize N particles
        self.N = N
        self.xpos = numpy.zeros(N)
        self.ypos = numpy.zeros(N)
        
        # make sure the particles are no closer than min_dist
        i = 0
        while i < N:
            xpos = numpy.random.random() * self.xsize
            ypos = numpy.random.random() * self.ysize
            
            for j in range(i):
                if (self.xpos[j] - xpos)**2 + (self.ypos[j] - ypos)**2 < (min_dist**2):
                    break
            else:
                self.xpos[i] = xpos
                self.ypos[i] = ypos
                i += 1
        
        # assign velocities according to Maxwell-Boltzmann distribution with random direction
        vel = scipy.stats.chi.rvs(2, scale=numpy.sqrt(MDBox.R * T / m), size=N)
        angles = numpy.random.random(size=N) * 2 * numpy.pi
        self.xvel = vel * numpy.cos(angles)
        self.yvel = vel * numpy.sin(angles)
        self.m = numpy.ones(N) * m
        self.q = numpy.ones(N) * q
    
    def newton_pendulum(self, N=5):
        """Delete all existing particles and sample a row of N particles to simulate a Newton pendulum."""
        # initialize N particles
        self.N = N
        self.xpos = numpy.arange(1, N + 1) * (self.xsize / (N + 1))
        self.ypos = numpy.ones(N) * (self.ysize * 0.5)
        self.xvel = numpy.zeros(N)
        self.xvel[-1] = -10
        self.yvel = numpy.zeros(N)
        self.m = numpy.ones(N)
        self.q = numpy.zeros(N)
    
    def load_particles(self, filename):
        """Delete all existing particles and load the given particle file."""
        data = numpy.load(filename)
        self.N = data.shape[1]
        self.xpos = data[0]
        self.ypos = data[1]
        self.xvel = data[2]
        self.yvel = data[3]
        self.m = data[4]
        self.q = data[5]
    
    def save_particles(self, filename):
        """Save the current particles to the given filename."""
        data = numpy.zeros((6, self.N))
        data[0] = self.xpos
        data[1] = self.ypos
        data[2] = self.xvel
        data[3] = self.yvel
        data[4] = self.m
        data[5] = self.q
        numpy.save(filename, data)
    
    # ========== End: Initialize particles ==========
    
    # ========== Start: Simulation ==========
    
    def simulate(self, steps=2000, dt=1, thermostat=None):
        """Delete the previous trajectories and simulate new trajectories.
        Thermostat: None to disable, otherwise array [target temperature, relaxation time]"""
        N = self.N
        
        # store initial positions and velocities
        self.trajectories = numpy.zeros((N, steps + 1, 2))
        self.trajectories[:,0,0] = self.xpos
        self.trajectories[:,0,1] = self.ypos
        
        # calculate initial accelerations
        self.__calculate_acceleration()
        
        # initialize evaluation process
        self.__init_evaluation(steps)
        
        # do time steps
        for step in tqdm(range(steps)):
                        
            # ordinary movement
            self.__move(dt)
            
            if thermostat:
                self.__rescale_velocities(thermostat[0], thermostat[1], dt)
            
            # evaluate the current step
            self.__evaluate_step(step)
            
            # store current positions and velocities
            self.trajectories[:, step + 1, 0] = self.xpos
            self.trajectories[:, step + 1, 1] = self.ypos

    def __move(self, dt):
        # ordinary movement
        self.xpos += self.xvel * dt + 0.5 * self.xacc * dt**2
        self.ypos += self.yvel * dt + 0.5 * self.yacc * dt**2
        
        # periodic box condition
        self.xpos %= self.xsize
        self.ypos %= self.ysize
        
        # calculate forces/accelerations
        self.last_xacc, self.last_yacc = self.xacc, self.yacc
        self.__calculate_acceleration()
        
        # update velocities
        self.xvel += 0.5 * (self.xacc + self.last_xacc) * dt
        self.yvel += 0.5 * (self.yacc + self.last_yacc) * dt
    
    
    def __calculate_force_vectors(self):
        # calculate distance vectors and absolute distance
        xdist, ydist = self.__distance_vectors()
        dist_sq = xdist**2 + ydist**2
        
        # set diagonal to inf (otherwise we get error messages upon division)
        numpy.fill_diagonal(dist_sq, numpy.inf)
        
        # ===== Start: Lennard-Jones forces =====
        # calculate exponentials of distance
        dist_6 = dist_sq**3
        dist_8 = dist_6 * dist_sq
        dist_14 = dist_8 * dist_6
        
        # calculate force matrix
        lennard_abs = (12 * MDBox.C12 * 1e6) / dist_14 - (6 * MDBox.C6 * 1e6) / dist_8
        lennard_x = lennard_abs * xdist
        lennard_y = lennard_abs * ydist
        # ===== End: Lennard-Jones forces
        
        # ===== Start: Coulomb forces =====
        # calculate matrix of charge products
        charge_products = numpy.tile(self.q, (self.N, 1)) * numpy.tile(self.q, (self.N, 1)).T
        
        # calculate force matrix
        coulomb_abs = MDBox.K_E * 1e6 * charge_products / (dist_sq**1.5)
        coulomb_x = coulomb_abs * xdist
        coulomb_y = coulomb_abs * ydist
        # ===== End: Coulomb forces =====
        
        # sum up force matrix per particle and return
        return (numpy.sum(lennard_x + coulomb_x, axis=0), numpy.sum(lennard_y + coulomb_y, axis=0))
    
    def __calculate_acceleration(self):
        xforce, yforce = self.__calculate_force_vectors()
        
        # calculate acceleration
        self.xacc = xforce / self.m
        self.yacc = yforce / self.m
          
    def __distance_vectors(self):
        # compute x distances including PBC
        xdist0 = numpy.tile(self.xpos, (self.N, 1)) - numpy.tile(self.xpos, (self.N, 1)).T
        xdist1 = xdist0 + self.xsize
        xdist2 = xdist0 - self.xsize
        
        # find the smallest absolute distance
        xdist = numpy.where(numpy.abs(xdist0) < numpy.abs(xdist1), xdist0, xdist1)
        xdist = numpy.where(numpy.abs(xdist) < numpy.abs(xdist2), xdist, xdist2)
        
        # compute x distances including PBC
        ydist0 = numpy.tile(self.ypos, (self.N, 1)) - numpy.tile(self.ypos, (self.N, 1)).T
        ydist1 = ydist0 + self.ysize
        ydist2 = ydist0 - self.ysize
        
        # find the smallest absolute distance
        ydist = numpy.where(numpy.abs(ydist0) < numpy.abs(ydist1), ydist0, ydist1)
        ydist = numpy.where(numpy.abs(ydist) < numpy.abs(ydist2), ydist, ydist2)
        
        return (xdist, ydist)
    
    def __rescale_velocities(self, T, tau, dt):
        # calculate rescaling factor according to 'Berendsen' thermostat
        lamda = numpy.sqrt(1 + dt / tau * (T / self.__calculate_temperature() - 1))
        
        # rescale velocities
        self.xvel *= lamda
        self.yvel *= lamda
    
    def __calculate_temperature(self):
        return 0.5 * numpy.sum(self.m * (self.xvel**2 + self.yvel**2)) / MDBox.R
    
    # ========== End: Simulation ==========
    
    # ========== Start: SD Minimization ==========
    
    def minimize_potential(self, dr=0.05, max_steps=1e3):
        # calculate initial potential
        Epot = self.__calculate_potential_energy()
        
        # calculate initial forces
        xforce, yforce = self.__calculate_force_vectors()
        
        # set step counter
        total_steps = 0
        dir_steps = 0
        
        while total_steps < max_steps:
            # build unit vectors
            force_abs = numpy.sqrt(xforce**2 + yforce**2)
            xunit = xforce / force_abs
            yunit = yforce / force_abs
            
            # move particles
            self.xpos += xunit * dr
            self.ypos += yunit * dr
            
            total_steps += 1
            
            # calculate potential
            Epot_new = self.__calculate_potential_energy()
            
            if Epot_new < Epot:
                # continue in same direction if step was successful
                Epot = Epot_new
                dir_steps += 1
                
            elif dir_steps > 0:
                # continue in new direction if last step was unsuccessful
                Epot = Epot_new
                # calculate new forces
                xforce, yforce = self.__calculate_force_vectors()
                # reset step counter
                dir_steps = 0
            else:
                # break if the last direction did not bring any successful steps
                self.xpos -= xunit * dr
                self.ypos -= yunit * dr
                break
        
        return total_steps
    
    def __calculate_potential_energy(self):
        xdist, ydist = self.__distance_vectors()
        dist_sq = xdist**2 + ydist**2
        
        # set diagonal to inf (otherwise we get error messages upon division)
        numpy.fill_diagonal(dist_sq, numpy.inf)
        
        # ===== Start: Lennard-Jones potential =====
        # calculate exponentials of distance
        dist_6 = dist_sq**3
        dist_12 = dist_6**2
        
        # calculate potential
        lennard_pot = MDBox.C12 / dist_12 - MDBox.C6 / dist_6
        # ===== End: Lennard-Jones potential
        
        # ===== Start: Coulomb potential =====
        # calculate matrix of charge products
        charge_products = numpy.tile(self.q, (self.N, 1)) * numpy.tile(self.q, (self.N, 1)).T
        
        # calculate potential
        coulomb_pot = MDBox.K_E * charge_products / numpy.sqrt(dist_sq)
        # ===== End: Coulomb potential
        
        return 0.5 * numpy.sum(lennard_pot + coulomb_pot)
    
    # ========== End: SD Minimization ==========
    
    # ========== Start: Evaluation ==========
    
    def __init_evaluation(self, steps):
        self.Epot = numpy.zeros(steps + 1)
        self.Ekin = numpy.zeros(steps + 1)
        
        self.__evaluate_step(-1)
    
    def __evaluate_step(self, step):
        self.Epot[step + 1] = self.__calculate_potential_energy()
        self.Ekin[step + 1] = 0.5 * 1e-6 * numpy.sum(self.m * (self.xvel**2 + self.yvel**2))
    
    # ========== End: Evaluation ==========
    
    # ========== Other functions ==========
    
    def render_video(self, styles=[["bo", 5.0, -1]], time_interval=20, frame_interval=1):
        """Render a video from the trajectory.
        Styles: list of elements [format string, marker size, number of trajectories (-1 => all remaining)]"""
        fig, ax = pyplot.subplots(figsize=(10, 10))

        ax.set_xlim([0, self.xsize])
        ax.set_ylim([0, self.ysize])

        ax.set_xlabel("position x")
        ax.set_ylabel("position y")

        dots = []
        for i in range(len(styles)):
            dot, = ax.plot([], [], styles[i][0], ms=styles[i][1])
            dots.append(dot)

        def init():
            for i in range(len(styles)):
                dots[i].set_data([], [])

        def animate(k):
            j = 0
            for i in range(len(styles)):
                if styles[i][2] <= 0:
                    dots[i].set_data(self.trajectories[j:,k,0], self.trajectories[j:,k,1])
                    break
                dots[i].set_data(self.trajectories[j:(j+styles[i][2]),k,0], self.trajectories[j:(j+styles[i][2]),k,1])
                j += styles[i][2]

        return animation.FuncAnimation(fig, animate, init_func=init, frames=tqdm(range(0, self.trajectories.shape[1], frame_interval)), interval=time_interval)

