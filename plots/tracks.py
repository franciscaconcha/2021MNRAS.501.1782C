import numpy
from matplotlib import pyplot

from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *


def tracks(open_path, N, save_path, t_end, save, nrun):
    """ Function to create Figure 1 on the paper.
    
    :param open_path: path to results file
    :param N: number of stars in results
    :param save_path: path to save figure
    :param t_end: final time to use when plotting
    :param save: if True, figure will be saved
    :param nrun: run number to use for the plot
    """
    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.gca()

    dt = 0.01
    times = numpy.arange(0.000, t_end + dt, dt)

    # Indexes of stars to plot
    chosen_stars = [166, 811, 963, 33, 58, 69, 171]
    massive_stars = [325, 37, 189, 339]

    label = open_path.split('/')[-2].split('_')[1].split('R')[1]

    if len(label) > 1:
        radius_label = label[0] + '.' + label[1]
    else:
        radius_label = label + '.0'

    Rvir = float(radius_label) | units.parsec

    f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(open_path, nrun, N, 0.000)
    stars0 = io.read_set_from_file(f, 'hdf5', close_file=True)
    converter = nbody_system.nbody_to_si(stars0.stellar_mass.sum(), Rvir)

    z = zip(range(len(stars0)), stars0.stellar_mass.value_in(units.MSun))

    min_mass = min(stars0.disk_mass.value_in(units.MJupiter))
    max_mass = max(stars0.disk_mass.value_in(units.MJupiter))

    # core radius
    pos, r_c, coredens = stars0.densitycentre_coreradius_coredens(number_of_neighbours=12,
                                                                  unit_converter=converter)
    Rc = r_c.value_in(units.parsec)

    # half-mass radius
    r_hm, mf = stars0.LagrangianRadii(mf=[0.5], unit_converter=converter)
    Rhm = r_hm.value_in(units.parsec)[0]

    # tidal radius
    r_t = (30. | units.parsec) * (stars0.stellar_mass.sum().value_in(units.MSun)
                                  / (3 * constants.G * 1E12 | units.MSun))**(1/3)
    Rt = r_t.value_in(units.parsec)

    Rc_circle = pyplot.Circle((0, 0), Rc, color='k', fill=False, ls='-', alpha=0.5)
    Rhm_circle = pyplot.Circle((0, 0), Rhm, color='k', fill=False, ls='--', alpha=0.5)
    Rt_circle = pyplot.Circle((0, 0), Rt, color='k', fill=False, ls='--', alpha=0.5)

    ax.add_artist(Rc_circle)
    ax.add_artist(Rhm_circle)
    ax.add_artist(Rt_circle)

    pyplot.set_cmap('viridis_r')

    disked = {}
    for s in chosen_stars:  # To keep track of which stars still have disks to plot
        disked[s] = True

    radii = []

    # Locations for text labels
    locx = {325: stars0[325].x.value_in(units.parsec) - 0.15,
            37: stars0[37].x.value_in(units.parsec) + stars0[37].x.value_in(units.parsec) / 10,
            189: stars0[189].x.value_in(units.parsec) + stars0[189].x.value_in(units.parsec) / 10,
            339: stars0[339].x.value_in(units.parsec) + 0.1,
            166: stars0[166].x.value_in(units.parsec),
            811: stars0[811].x.value_in(units.parsec) - 0.1,
            963: stars0[963].x.value_in(units.parsec),
            33: stars0[33].x.value_in(units.parsec) - 0.1,
            58: stars0[58].x.value_in(units.parsec) + 0.1,
            69: stars0[69].x.value_in(units.parsec),
            171: stars0[171].x.value_in(units.parsec) - 0.1}
    locy = {325: stars0[325].y.value_in(units.parsec) + 0.1,
            37: stars0[37].y.value_in(units.parsec) - 0.2,
            189: stars0[189].y.value_in(units.parsec) + 0.05,
            339: stars0[339].y.value_in(units.parsec) - 0.05,
            166: stars0[166].y.value_in(units.parsec) + 0.1,
            811: stars0[811].y.value_in(units.parsec) - 0.2,
            963: stars0[963].y.value_in(units.parsec) + 0.05,
            33: stars0[33].y.value_in(units.parsec) + 0.1,
            58: stars0[58].y.value_in(units.parsec),
            69: stars0[69].y.value_in(units.parsec) + 0.1,
            171: stars0[171].y.value_in(units.parsec) + 0.1
            }

    for t in times:
        f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(open_path, nrun, N, t)
        stars = io.read_set_from_file(f, 'hdf5', close_file=True)

        z = stars.disk_mass.value_in(units.MJupiter)

        for s in massive_stars:
            star = stars[s]
            pyplot.plot(star.x.value_in(units.parsec),
                        star.y.value_in(units.parsec),
                        marker='.',
                        markersize=0.5,
                        c='b',
                        alpha=1)
            if t == 0.0:
                pyplot.scatter(star.x.value_in(units.parsec),
                               star.y.value_in(units.parsec),
                               marker='x',
                               s=150,
                               c='k',
                               lw=1)
                txt = pyplot.text(locx[s], # x
                                  locy[s], # y
                                  r'{0:.2f} $M_{{\odot}}$'.format(star.stellar_mass.value_in(units.MSun)),
                                  fontsize=10)

        for s in chosen_stars:
            star = stars[s]
            radii.append(star.disk_radius.value_in(units.au))
            if star.disked:
                p = pyplot.scatter(star.x.value_in(units.parsec),
                                   star.y.value_in(units.parsec),
                                   s=5 * star.disk_radius.value_in(units.au),
                                   c=z[s],
                                   alpha=0.5,
                                   norm=matplotlib.colors.SymLogNorm(linthresh=0.001,
                                                                     vmin=min_mass,
                                                                     vmax=max_mass))

                if t == 0.0:
                    pyplot.scatter(star.x.value_in(units.parsec),
                                   star.y.value_in(units.parsec),
                                   marker='x',
                                   s=150,
                                   c='k',
                                   lw=1)
                    txt = pyplot.text(locx[s],  # x
                                      locy[s],  # y
                                      r'{0:.2f} $M_{{\odot}}$'.format(star.stellar_mass.value_in(units.MSun)),
                                      fontsize=10)
            else:
                if disked[s]:
                    pyplot.scatter(star.x.value_in(units.parsec),
                                   star.y.value_in(units.parsec),
                                   marker='x',
                                   s=80,
                                   c='r',
                                   alpha=1,
                                   lw=2)
                    disked[s] = False
                else:
                    pyplot.plot(star.x.value_in(units.parsec),
                                   star.y.value_in(units.parsec),
                                   marker='.',
                                   markersize=0.5,
                                   c='k',
                                   alpha=1)

    legend_elements = [pyplot.scatter([],
                                      [],
                                      s=15**2,
                                      facecolors='w',
                                      edgecolors='k',
                                      alpha=0.5,
                                      linestyle='-',
                                      label='Core radius ({0:.1f} pc)'.format(Rc)),
                       pyplot.scatter([],
                                      [],
                                      s=14 ** 2,
                                      facecolors='w',
                                      edgecolors='k',
                                      alpha=0.5,
                                      linestyle='--',
                                      label='Half-mass radius ({0:.1f} pc)'.format(Rhm))
                       ]

    pyplot.legend(handles=legend_elements,
                  loc='lower right',
                  fontsize=16,
                  handletextpad=0.1)

    cbar = pyplot.colorbar(p)
    cbar.set_clim(0, 100)
    cbar.set_label(r'Total disc mass [$\mathrm{M}_{Jup}$]')

    pyplot.xlim([-2.5, 2.5])
    pyplot.ylim([-2.5, 2.5])

    pyplot.xlabel("x [pc]")
    pyplot.ylabel("y [pc]")

    pyplot.axes().set_aspect('equal')

    if save:
        pyplot.savefig('{0}/tracks.png'.format(save_path))


def main(open_path, N, save_path, t_end, save, Rvir, distance, nruns, movie):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    tracks(open_path, N, save_path, t_end, save, nruns)

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
    result.add_option("-R", dest="Rvir", type="float",
                      unit=units.parsec, default=0.5,
                      help="cluster virial radius [%default]")
    result.add_option("-d", dest="distance", type="float", default=0.0,
                      help="When using galactic potential, ('projected') distance to galactic center [%default]")
    result.add_option("-n", dest="nruns", type="int", default=1,
                      help="number of runs to plot for averages [%default]")
    result.add_option("-m", dest="movie", type="int", default=0,
                      help="make movie? [%default]")
    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
