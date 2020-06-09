import numpy
from matplotlib import pyplot

from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *


def radii_vs_time(open_path, save_path, t_end, N, nruns, save):
    #fig1, ax1 = pyplot.subplots(1)
    #fig2, ax2 = pyplot.subplots(1)
    fig, axs = pyplot.subplots(1, 2, figsize=(16, 8))
    ax1 = axs[0]
    ax2 = axs[1]
    dt = 0.05
    times = numpy.arange(0.0, t_end + dt, dt)

    #rc_mean = {'N1E3_R01': }

    #folders = ['N1E3_R01', 'N1E3_R05', 'N1E3_R03']

    for folder in folders:
        i_folder = folders.index(folder)
        path = '{0}/{1}/'.format(open_path, folder)
        try:
            label = path.split('/')[-2].split('_')[1]
        except:
            label = path.split('/')[-3].split('_')[1]

        radius_label = label.split('R')[-1]
        if len(radius_label) > 1:
            radius_label = radius_label[0] + '.' + radius_label[1]
        else:
            radius_label = radius_label + '.0'

        all_rc, all_rhm = [], []
        t_relax = []
        for n in range(nruns):
            rc, rhm = [], []
            run_path = '{0}/{1}'.format(path, n)
            for t in times:
                f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                Rvir = float(radius_label) | units.parsec

                converter = nbody_system.nbody_to_si(stars.stellar_mass.sum(), Rvir)

                # core radius
                pos, r_c, coredens = stars.densitycentre_coreradius_coredens(number_of_neighbours=12,
                                                                              unit_converter=converter)
                Rc = r_c.value_in(units.parsec)
                #Rc = stars.virial_radius().value_in(units.parsec)#r_c.value_in(units.parsec)

                # half-mass radius
                r_hm, mf = stars.LagrangianRadii(mf=[0.5], unit_converter=converter)
                Rhm = r_hm.value_in(units.parsec)[0]

                rc.append(Rc)
                rhm.append(Rhm)

                if t == 0.0:
                    bound = stars.bound_subset(tidal_radius=r_hm, unit_converter=converter)
                    tdyn = numpy.sqrt(Rvir ** 3 / (constants.G * bound.stellar_mass.sum()))
                    print tdyn.value_in(units.Myr)
                    Nn = len(bound)
                    g = 0.4
                    trh = 0.138 * (Nn / numpy.log(g * Nn)) * tdyn
                    t_relax.append(1E-6 * trh.value_in(units.yr))

            #for thisrc in all_rc:
            #print folder
            #print rc
            #print rhm
            #print "t relax"
            #print t_relax
            ax1.plot(times, rc, lw=3, color=colors[label], ls=lines[label], alpha=0.5)

            #for thisrhm in all_rhm:
            ax2.plot(times, rhm, lw=3, color=colors[label], ls=lines[label], alpha=0.5)

        """    all_rc.append(rc)
            all_rhm.append(rhm)

        mean_rc = numpy.mean(all_rc, axis=0)
        mean_rhm = numpy.mean(all_rhm, axis=0)

        rc_stdev = numpy.std(all_rc, axis=0)
        rhm_stdev = numpy.std(all_rhm, axis=0)

        rc_high = mean_rc + rc_stdev
        rc_low = mean_rc - rc_stdev
        rhm_high = mean_rhm + rhm_stdev
        rhm_low = mean_rhm - rhm_stdev

        ax1.fill_between(times,
                          rc_high,
                          rc_low,
                          facecolor=colors[label],
                          alpha=0.2)
        ax1.plot(times, mean_rc, lw=3, label=labels[label], color=colors[label], ls=lines[label])

        ax2.fill_between(times,
                          rhm_high,
                          rhm_low,
                          facecolor=colors[label],
                          alpha=0.2)
        ax2.plot(times, mean_rhm, lw=3, label=labels[label], color=colors[label], ls=lines[label])

        print "{0}, mean_rc".format(folder)
        print list(mean_rc)
        print "{0}, rc_low".format(folder)
        print list(rc_low)
        print "{0}, rc_high".format(folder)
        print list(rc_high)

        print "{0}, mean_rhm".format(folder)
        print list(mean_rhm)
        print "{0}, rhm_low".format(folder)
        print list(rhm_low)
        print "{0}, rhm_high".format(folder)
        print list(rhm_high)"""

    ax1.set_xlim([0.0, 2.0])
    ax2.set_xlim([0.0, 2.0])

    ax1.set_xlabel('Time [Myr]', fontsize=28)
    ax1.set_ylabel(r'Core radius [pc]', fontsize=28)

    ax2.set_xlabel('Time [Myr]', fontsize=28)
    ax2.set_ylabel(r'Half mass radius [pc]', fontsize=28)

    fig.subplots_adjust(top=0.98, bottom=0.3, wspace=0.33)

    pyplot.legend([R01ShadedObject(), R03ShadedObject(), R05ShadedObject(),
                   R1ShadedObject(), R25ShadedObject(), R5ShadedObject()],
                  [labels['R01'], labels['R03'], labels['R05'],
                   # "", "",
                   labels['R1'], labels['R25'], labels['R5'],
                   ],
                  handler_map={R01ShadedObject: R01ShadedObjectHandler(),
                               R03ShadedObject: R03ShadedObjectHandler(),
                               R05ShadedObject: R05ShadedObjectHandler(),
                               R1ShadedObject: R1ShadedObjectHandler(),
                               R25ShadedObject: R25ShadedObjectHandler(),
                               R5ShadedObject: R5ShadedObjectHandler()},
                  loc='upper right',
                  bbox_to_anchor=(0.55, -0.18),
                  ncol=3,
                  fontsize=22, framealpha=0.4)

    #pyplot.suptitle(r'N = {0}'.format(N))
    #pyplot.xlim([0.0, t_end])
    #pyplot.ylim([0.0, 1.0])

    if save:
        pyplot.savefig('{0}/core_radii_vs_time.png'.format(save_path, N))

    #pyplot.show()


def main(open_path, N, save_path, t_end, save, nruns):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    #separated_masses(open_path, save_path, t_end, N, nruns, save)
    radii_vs_time(open_path, save_path, t_end, N, nruns, save)

    if not save:
        pyplot.show()


def new_option_parser():
    from amuse.units.optparse import OptionParser
    result = OptionParser()

    result.add_option("-p", dest="open_path", type="string", default='/media/fran/data1/photoevap/results',
                      help="path to results to plot [%default]")
    result.add_option("-N", dest="N", type="int", default=1000,
                      help="number of stars [%default]")
    result.add_option("-S", dest="save", type="int", default=0,
                      help="save plot? [%default]")
    result.add_option("-s", dest="save_path", type="string", default='/media/fran/data1/photoevap-paper/figures',
                      help="path to save the results [%default]")
    result.add_option("-e", dest="t_end", type="float", default='5.0',
                      help="end time to use for plots [%default]")
    result.add_option("-n", dest="nruns", type="int", default=1,
                      help="number of runs to plot for averages [%default]")
    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
