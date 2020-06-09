import numpy
import Queue
from amuse.units import units, constants


code_queue = Queue.Queue()

G0 = 1.6e-3 * units.erg / units.s / units.cm**2


class Disk:
    def __init__(self,
                 star_id,
                 disk_radius,
                 disk_gas_mass,
                 central_mass,
                 dispersed_mass_threshold,
                 dispersed_density_threshold,
                 grid,
                 mu=2.33,
                 #Tm=120.|units.K,
                 delta=1e-2,
                 rho_g=1.|units.g/units.cm**3,
                 a_min=1e-8|units.m,
                 internal_photoevap_flag=False,
                 external_photoevap_flag=False):
        """ Initialize Disk. VADER codes must be initialized separately.

        :param star_id: key of the star to which the disk belongs [long int]
        :param disk_radius: disk radius [au]
        :param disk_gas_mass: disk mass [mass units]
        :param central_mass: host star's mass [mass units]
        :param dispersed_mass_threshold: mass under which the disk is considered as dispersed [mass units]
        :param grid: vader grid
        :param mu: mean molecular mass [hydrogen masses]
        :param Tm: midplane temperature at 1 AU [K]
        :param delta: (dust mass)/(gas mass) ratio [adimensional]
        :param rho_g: density of individual dust grains [adimensional]
        :param a_min: minimum dust grain size
        :param internal_photoevap_flag: flag to activate internal photoevaporation [bool]
        :param external_photoevap_flag: flag to activate external photoevaporation [bool]
        """

        self.time = 0. | units.Myr

        self.key = star_id
        self.dispersed = False   # Disk is dispersed if the mass is lower than dispersed_mass_threshold
        self.disk_convergence_failure = False   # Viscous code can fail to converge; catch and do not involve further
        self.disk_active = True    # Disk is only evolved if it is not dispersed or failed to evolve

        self.internal_photoevap_flag = internal_photoevap_flag
        self.external_photoevap_flag = external_photoevap_flag

        #self.Tm = Tm
        self.Tm = 100. * (central_mass.value_in(units.MSun))**(1./4.) | units.K
        self.mu = mu
        T = self.Tm / numpy.sqrt(grid.r.value_in(units.AU))  # Disk midplane temperature at 1 AU is Tm, T(r)~r^-1/2

        self.grid = grid.copy()
        self.grid.column_density = self.column_density(disk_radius, disk_gas_mass)
        self.grid.pressure = self.grid.column_density * constants.kB*T / (mu*1.008*constants.u)  # Ideal gas law

        self.central_mass = central_mass
        self.accreted_mass = 0. | units.MSun  # Mass accreted from disk (still important for gravity)

        self.dispersed_mass_threshold = dispersed_mass_threshold
        self.dispersed_density_threshold = dispersed_density_threshold

        self.delta = delta
        self.rho_g = rho_g
        self.a_min = a_min

        self.disk_dust_mass = self.delta * self.disk_gas_mass

    def evolve_disk_for(self, dt):
        """
        Evolve a protoplanetary disk for a time step. Gas evolution is done through VADER, dust evaporation through
        the prescription of Haworth et al. 2018 (MNRAS 475).
        Note that before calling this function, a VADER code must be assigned to the self.viscous property.

        :param dt: time step to evolve the disk for [scalar, units of time]
        """

        # Adjust rotation curves to current central mass
        self.viscous.update_keplerian_grid(self.central_mass)

        # Specified mass flux, using VADER function
        self.viscous.parameters.inner_pressure_boundary_type = 1
        self.viscous.parameters.inner_boundary_function = True

        # User-defined VADER parameters
        # Internal photoevaporation rate
        self.viscous.set_parameter(0, 
                                   self.internal_photoevap_flag * self.inner_photoevap_rate.value_in(units.g/units.s))
        # External photoevaporation rate
        self.viscous.set_parameter(1,
                                   self.external_photoevap_flag * self.outer_photoevap_rate.value_in(units.g/units.s))
        #print "PE rate from disk_class: ", (self.external_photoevap_flag * self.outer_photoevap_rate).in_(units.MSun / units.yr)

        # Nominal accretion rate
        self.viscous.set_parameter(5, self.accretion_rate.value_in(units.g/units.s))

        # As codes are re-used, need to remember initial state
        initial_accreted_mass = -self.viscous.inner_boundary_mass_out

        # Channels to efficiently transfer data to and from code
        ch_fram_to_visc = self.grid.new_channel_to(self.viscous.grid)  # class to code
        ch_visc_to_fram = self.viscous.grid.new_channel_to(self.grid)  # code to class

        # Copy disk data to code
        ch_fram_to_visc.copy()

        # Gas and dust evaporation are coupled in a leapfrog-like method
        # (half step gas, full step dust, half step gas)
        try:
            self.viscous.evolve_model(self.viscous.model_time + dt/2.)

        except:
            print("Partial convergence failure at {a} Myr".format(a=self.time))
            # Failure is often due to excessive accretion, so switch to zero-torque and restart
            self.viscous.parameters.inner_pressure_boundary_type = 3
            self.viscous.parameters.inner_boundary_function = False

            initial_accreted_mass = -self.viscous.inner_boundary_mass_out

            ch_fram_to_visc.copy()

            try:
                self.viscous.evolve_model(self.viscous.model_time + dt/2.)

            except:
                # If still fails, give up hope
                print("Absolute convergence failure at {a} Myr".format(a=self.time))
                self.disk_convergence_failure = True

        self.time += dt/2.

        # Copy disk data to class 
        ch_visc_to_fram.copy()

        # If the disk is below dispersion threshold
        if self.disk_gas_mass < self.dispersed_mass_threshold or self.disk_density < self.dispersed_density_threshold:
            print "Disk {0} dispersed".format(self.key)
            self.dispersed = True
            self.disk_radius = 0.0 | units.au
            self.disk_mass = 0.0 | units.MSun

        # Keep track of mass accreted from the disk,
        # as in the code this is the sum of all past mass accretions (including from other disks)
        if not self.disk_convergence_failure:
            self.accreted_mass += -self.viscous.inner_boundary_mass_out - initial_accreted_mass
            initial_accreted_mass = -self.viscous.inner_boundary_mass_out

        # Flag to decide whether or not to evolve the disk
        self.disk_active = (not self.dispersed) * (not self.disk_convergence_failure)

        # Remove dust in a midpoint-like integration
        # Follows the prescription of Haworth et al. 2018 (MNRAS 475)

        # Thermal speed of particles
        v_th = (8. * constants.kB * self.Tm / numpy.sqrt(self.disk_radius.value_in(units.AU))
                / (numpy.pi * self.mu * 1.008 * constants.u))**(1./2.)

        # Disk scale height at disk edge
        Hd = (constants.kB * self.Tm * (1. | units.AU)**(1./2.) * self.disk_radius**(5./2.)
              / (self.mu * 1.008 * constants.u * self.central_mass*constants.G))**(1./2.)

        # Disk filling factor of sphere at disk edge
        F = Hd/(Hd**2 + self.disk_radius**2)**(1./2.)

        Mdot_dust = self.delta * self.outer_photoevap_rate**(3./2.) * \
                    (v_th / (4. * numpy.pi * F * constants.G * self.central_mass * self.rho_g*self.a_min))**(1./2.) * \
                    numpy.exp(-self.delta * (constants.G * self.central_mass)**(1./2.) *
                           self.time / (2. * self.disk_radius**(3./2.)))

        # Can't entrain more dust than is available
        if self.delta * self.outer_photoevap_rate < Mdot_dust:
            Mdot_dust = self.delta * self.outer_photoevap_rate

        # Eulerian integration
        dM_dust = Mdot_dust * dt
        self.disk_dust_mass -= dM_dust

        # Can't have negative mass
        if self.disk_dust_mass < 0. | units.MSun:
            self.disk_dust_mass = 0. | units.MSun

        # Back to fixed accretion rate, has potentially switched above
        self.viscous.parameters.inner_pressure_boundary_type = 1
        self.viscous.parameters.inner_boundary_function = True

        if self.disk_active:
            try:
                self.viscous.evolve_model(self.viscous.model_time + dt/2.)

            except:
                print("Partial convergence failure at {a} Myr".format(a=self.time))
                self.viscous.parameters.inner_pressure_boundary_type = 3
                self.viscous.parameters.inner_boundary_function = False

                initial_accreted_mass = -self.viscous.inner_boundary_mass_out

                ch_fram_to_visc.copy()

                try: 
                    self.viscous.evolve_model(self.viscous.model_time + dt/2.)

                except:
                    print ("Absolute convergence failure at {a} Myr".format(a=self.time))
                    self.disk_convergence_failure = True

        self.time += dt/2.

        ch_visc_to_fram.copy()

        if self.disk_gas_mass < self.dispersed_mass_threshold or self.disk_density < self.dispersed_density_threshold:
            print "Disk {0} dispersed".format(self.key)
            self.dispersed = True
            self.disk_radius = 0.0 | units.au
            self.disk_mass = 0.0 | units.MSun

        if not self.disk_convergence_failure:
            self.accreted_mass += -self.viscous.inner_boundary_mass_out - initial_accreted_mass

        self.disk_active = (not self.dispersed) * (not self.disk_convergence_failure)

    def column_density(self,
                       rc,
                       disk_gas_mass,
                       lower_density=1E-12 | units.g / units.cm**2,
                       rd=None):
        """ Set up a Lynden-Bell & Pringle 1974 disk profile.

        :param rc: scale length of disk profile [scalar, units of length]
        :param disk_gas_mass: target disk gas mass [scalar, units of mass]
        :param lower_density: minimum surface density of disk [scalar, units of mass per surface]
        :param rd: disk cutoff length [scalar, units of length]

        :return: surface density at positions defined on the grid [vector, units of mass per surface]
        """

        # If no cutoff is specified, the scale length is used
        # Following Anderson et al. 2013
        if rd is None:
            rd = rc

        r = self.grid.r.copy()

        Sigma_0 = disk_gas_mass/(2.*numpy.pi * rc**2 * (1. - numpy.exp(-rd/rc)))
        Sigma = Sigma_0 * (rc/r) * numpy.exp(-r/rc) * (r <= rd) + lower_density

        return Sigma

    def truncate(self,
                 new_radius,
                 lower_density=1E-12 | units.g / units.cm ** 2):
        """ Truncate a disk.

        :param new_radius: new radius of disk
        :param lower_density: lowerdensity limit for disk boundary definition
        """
        self.grid[self.grid.r > new_radius].column_density = lower_density
        return self

    @property
    def accretion_rate(self):
        """ Mass-dependent accretion rate of T-Tauri stars according to Alcala et al. 2014
        """
        return 10.**(1.81*numpy.log10(self.central_mass.value_in(units.MSun)) - 8.25) | units.MSun / units.yr

    @property
    def inner_photoevap_rate(self):
        """ Internal photoevaporation rate of protoplanetary disks from Picogna et al. 2019,
            with mass scaling following Owen et al. 2012
        """
        Lx = self.xray_luminosity.value_in(units.erg / units.s)
        return 10.**(-2.7326*numpy.exp(-(numpy.log(numpy.log10(Lx)) - 3.3307)**2/2.9868e-3) - 7.2580) \
                  * (self.central_mass/(0.7 | units.MSun))**-0.068 | units.MSun / units.yr

    @property
    def xray_luminosity(self):
        """ Mass-dependent X-ray luminosity of classical T-Tauri stars
            according to Flaccomio et al. 2012 (typical luminosities)
        """
        return 10.**(1.40*numpy.log10(self.central_mass.value_in(units.MSun)) + 30.) | units.erg / units.s

    @property
    def disk_radius(self, f=0.999):
        """ Gas radius of the disk, defined as the radius within which a fraction f of the total mass is contained

        :param f: fraction of mass within disk radius [float]

        :return: disk radius [scalar, units of length]
        """

        Mtot = (self.grid.area * self.grid.column_density).sum()
        Mcum = 0. | units.MSun

        edge = -1

        for i in range(len(self.grid.r)):

            Mcum += self.grid.area[i] * self.grid.column_density[i]
            
            if Mcum >= Mtot * f:
                edge = i
                break

        return self.grid.r[edge]

    @property
    def disk_gas_mass(self):
        """ Gas mass of disk (defined as total mass on VADER grid) """
        return (self.grid.area * self.grid.column_density).sum()

    @property
    def disk_mass(self):
        """ Total disk mass """
        return self.disk_dust_mass + self.disk_gas_mass

    @property
    def disk_density(self):
        return self.disk_mass / (numpy.pi * self.disk_radius**2)
