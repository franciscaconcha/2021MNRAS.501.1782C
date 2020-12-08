import numpy
from matplotlib import pyplot
from scipy import stats
from sklearn.neighbors import KDTree

from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *  # My own custom legend definitions


# Changing some legend labels for these particular plots
class R01ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R01'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R01'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R01'],
            facecolor=colors['R01'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


class R03ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R03'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R03'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R03'],
            facecolor=colors['R03'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


class R05ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R05'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R05'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R05'],
            facecolor=colors['R05'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


class R1ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R1'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R1'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R1'],
            facecolor=colors['R1'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


class R25ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R25'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R25'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R25'],
            facecolor=colors['R25'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


class R5ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+1, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R5'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0+2, y0 + width + 7],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="-",
                           color=colors['R5'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 40),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R5'],
            facecolor=colors['R5'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        return [l1, l2, l3]


def model_mass_vs_local_density(open_path, save_path, t_end, N, nruns, save, all_in_one=True):
    """ Figure 3: Binned mean disc mass versus local stellar number density.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    :param all_in_one: plot all curves in one single plot
    """
    if all_in_one:
        fig, axes = pyplot.subplots(2, 3,
                                    figsize=(15, 10),
                                    # sharex=True,
                                    # sharey=True,
                                    gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
        axs = {folders[0]: axes[0, 0],  # R0.1
               folders[1]: axes[0, 1],  # R0.3
               folders[2]: axes[0, 2],  # R0.5
               folders[3]: axes[1, 0],  # R1.0
               folders[4]: axes[1, 1],  # R2.5
               folders[5]: axes[1, 2]}  # R5.0
    else:
        fig0, axes0 = pyplot.subplots(1)
        fig1, axes1 = pyplot.subplots(1)
        fig2, axes2 = pyplot.subplots(1)
        fig3, axes3 = pyplot.subplots(1)
        fig4, axes4 = pyplot.subplots(1)
        fig5, axes5 = pyplot.subplots(1)

        axs = {folders[0]: axes0,  # R0.1
               folders[1]: axes1,  # R0.3
               folders[2]: axes2,  # R0.5
               folders[3]: axes3,  # R1.0
               folders[4]: axes4,  # R2.5
               folders[5]: axes5}  # R5.0

    dt = 0.2
    times = numpy.arange(0.000, t_end + dt, dt)

    xmin = 1E10
    xmax = 0.0

    for folder in folders:
        i_folder = folders.index(folder)
        path = '{0}/{1}/'.format(open_path, folder)
        for t in times:
            label = path.split('/')[-2].split('_')[1]
            all_binned_means = []
            all_binned_stds = []
            all_binned_means_locdens = []
            all_binned_stds_locdens = []

            i = 0

            for n in range(nruns):
                f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(path, n, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                center = stars.center_of_mass()

                # Calculate local densities
                positions = []
                for s in stars:
                    positions.append(numpy.array([s.x.value_in(units.parsec),
                                                  s.y.value_in(units.parsec),
                                                  s.z.value_in(units.parsec)]))
                positions = numpy.array(positions)

                tree = KDTree(positions)

                nearest_dist, nearest_ind = tree.query(positions, k=5)

                for i in range(len(stars)):
                    s = stars[i]
                    distances_to_neighbours = nearest_dist[i]
                    max_distance = max(distances_to_neighbours)
                    s.local_density = 5. / ((4. / 3.) * numpy.pi * max_distance ** 3)

                disk_masses = stars.disk_mass.value_in(units.MJupiter)
                masses_sorted_by_local_dens = [float(x) for _, x in sorted(zip(stars.local_density, disk_masses))]
                sorted_local_dens = sorted(stars.local_density)

                binned_means_locdens = []
                binned_stds_locdens = []
                locdens_means = []
                d = 100

                for i in range(len(masses_sorted_by_local_dens)):
                    if len(masses_sorted_by_local_dens) - (i + d) > 0:
                        binned_means_locdens.append(numpy.mean(masses_sorted_by_local_dens[i:i + d]))
                        binned_stds_locdens.append(stats.sem(masses_sorted_by_local_dens[i:i + d]))
                        locdens_means.append(numpy.mean(sorted_local_dens[i:i + d]))
                    else:
                        # print "end"
                        binned_means_locdens.append(numpy.mean(masses_sorted_by_local_dens[i:]))
                        binned_stds_locdens.append(stats.sem(masses_sorted_by_local_dens[i:]))
                        locdens_means.append(numpy.mean(sorted_local_dens[i:i + d]))
                        break

                all_binned_means_locdens.append(numpy.array(binned_means_locdens))
                all_binned_stds_locdens.append(binned_stds_locdens)

            try:
                all_means = numpy.mean(all_binned_means_locdens, axis=0)
                devs = numpy.mean(all_binned_stds_locdens, axis=0)
            except ValueError:
                max_len = 0
                for a in all_binned_means_locdens:
                    if len(a) > max_len:
                        max_len = len(a)

                new_sorted = []
                for a in all_binned_means_locdens:
                    b = numpy.pad(a, (max_len - len(a), 0), 'constant')
                    new_sorted.append(b)
                all_means = numpy.mean(all_binned_means_locdens, axis=0)
                devs = numpy.mean(all_binned_stds_locdens, axis=0)

            all_means_high = all_means + devs
            all_means_low = all_means - devs

            if min(locdens_means) < xmin:
                xmin = min(locdens_means)
            if max(locdens_means) > xmax:
                xmax = max(locdens_means)

            if t == 0.0:
                axs[folders[i_folder]].plot(locdens_means,
                                            all_means,
                                            color=colors[label],
                                            lw=2, ls=":")

                axs[folders[i_folder]].fill_between(locdens_means,
                                                    all_means_high,
                                                    all_means_low,
                                                    facecolor=colors[label],
                                                    alpha=0.2)
            elif t == 2.0:
                axs[folders[i_folder]].plot(locdens_means,
                                            all_means,
                                            color=colors[label],
                                            lw=2, ls="-")

                axs[folders[i_folder]].fill_between(locdens_means,
                                                    all_means_high,
                                                    all_means_low,
                                                    facecolor=colors[label],
                                                    alpha=0.2)
            else:
                axs[folders[i_folder]].plot(locdens_means,
                                            all_means,
                                            color=colors[label],
                                            lw=1, ls="-", alpha=0.2)

            axs[folders[i_folder]].set_xscale('log')
            axs[folders[i_folder]].set_yscale('log')

    # To make sure all horizontal plots have the same yticks
    axs[folders[1]].set_ylim(axs[folders[0]].get_ylim())
    axs[folders[2]].set_ylim(axs[folders[0]].get_ylim())

    axs[folders[4]].set_ylim(axs[folders[3]].get_ylim())
    axs[folders[5]].set_ylim(axs[folders[3]].get_ylim())

    # Radii labels
    axs[folders[0]].text(0.5, 0.1, labels[folders[0].split('_')[1]])
    axs[folders[1]].text(0.5, 0.1, labels[folders[1].split('_')[1]])
    axs[folders[2]].text(3.5, 0.1, labels[folders[2].split('_')[1]])
    axs[folders[3]].text(0.6, 0.9, labels[folders[3].split('_')[1]])
    axs[folders[4]].text(0.05, 0.9, labels[folders[4].split('_')[1]])
    axs[folders[5]].text(0.005, 0.9, labels[folders[5].split('_')[1]])

    # x-labels
    if all_in_one:
        fig.text(0.5, 0.58,
                 r'Local stellar number density [pc$^{-3}$]',
                 ha='center', va='center', fontsize=26)
        fig.text(0.5, 0.12,
                 r'Local stellar number density [pc$^{-3}$]',
                 ha='center', va='center', fontsize=26)

        # y-label
        fig.text(0.06, 0.6,
                 r'Binned mean disc mass [$\mathrm{M}_{Jup}$]',
                 ha='center', va='center', rotation='vertical', fontsize=26)

    fig.subplots_adjust(top=0.98, bottom=0.2, wspace=0.33)

    pyplot.legend([DottedShadedObject(), SolidShadedObject()],
                  [r'$t = 0.0$ Myr',
                   r'$t = 2.0$ Myr'],
                  handler_map={DottedShadedObject: DottedShadedObjectHandler(),
                               SolidShadedObject: SolidShadedObjectHandler()},
                  loc='upper center',
                  bbox_to_anchor=(-0.75, -0.3),
                  ncol=2,
                  fontsize=24, framealpha=0.5)

    if save:
        if all_in_one:
            fig.savefig('{0}/binned_mass_vs_local_density.png'.format(save_path))
        else:
            fig0.savefig('{0}/R01_binned_mass_vs_local_density.png'.format(save_path))
            fig1.savefig('{0}/R03_binned_mass_vs_local_density.png'.format(save_path))
            fig2.savefig('{0}/R05_binned_mass_vs_local_density.png'.format(save_path))
            fig3.savefig('{0}/R1_binned_mass_vs_local_density.png'.format(save_path))
            fig4.savefig('{0}/R25_binned_mass_vs_local_density.png'.format(save_path))
            fig5.savefig('{0}/R5_binned_mass_vs_local_density.png'.format(save_path))


def projected_mass_vs_local_density(open_path, save_path, t_end, N, nruns, save):
    """ Figure 4: Binned mean disc mass versus local stellar number density, projected in two dimensions.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """
    axs = [None, None]
    fig1, axs[0] = pyplot.subplots(1)

    max_mass = 0.0

    dt = 0.2
    times = numpy.arange(0.0, t_end + dt, dt)
    times = numpy.array([0.0, 2.0])

    for folder in folders:
        path = '{0}/{1}'.format(open_path, folder)
        last_t = 0.0
        for t in times:
            if nruns > 0:
                label = folder.split('_')[1]
                all_binned_means = []
                all_binned_stds = []
                all_binned_means_locdens = []
                all_binned_stds_locdens = []

                for n in range(nruns):
                    try:
                        f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(path, n, N, t)
                        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                        last_t = t
                    except:
                        f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(path, n, N, last_t)
                        stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                    center = stars.center_of_mass()
                    stars = stars[stars.disked]

                    positions = []
                    for s in stars:
                        positions.append(numpy.array([s.x.value_in(units.parsec),
                                                      s.y.value_in(units.parsec)]))#,
                                                      #s.z.value_in(units.parsec)]))
                    positions = numpy.array(positions)

                    tree = KDTree(positions)

                    nearest_dist, nearest_ind = tree.query(positions, k=5)
                    # print(nearest_dist)  # drop id; assumes sorted -> see args!
                    # print(nearest_ind)

                    for i in range(len(stars)):
                        s = stars[i]
                        distances_to_neighbours = nearest_dist[i]
                        max_distance = max(distances_to_neighbours)
                        s.local_density = 5. / (numpy.pi * max_distance ** 2)

                    disk_masses = stars.disk_mass.value_in(units.MEarth) / 100
                    masses_sorted_by_local_dens1 = [float(x) for _, x in sorted(zip(stars.local_density, disk_masses))]
                    sorted_local_dens = sorted(stars.local_density)

                    masses_sorted_by_local_dens2 = []

                    for m in masses_sorted_by_local_dens1:
                        #if m > 0.0:
                        masses_sorted_by_local_dens2.append(numpy.log10(m))
                        #else:
                        #masses_sorted_by_local_dens2.append(0.0)

                    masses_sorted_by_local_dens = []
                    for m in masses_sorted_by_local_dens2:
                        masses_sorted_by_local_dens.append(numpy.power(10, m))

                    # LOCAL DENSITY
                    binned_means_locdens = []
                    binned_stds_locdens = []
                    locdens_means = []
                    d = 100
                    for i in range(len(masses_sorted_by_local_dens)):
                        if len(masses_sorted_by_local_dens) - (i + d) > 0:
                            binned_means_locdens.append(numpy.mean(masses_sorted_by_local_dens[i:i+d]))
                            binned_stds_locdens.append(stats.sem(masses_sorted_by_local_dens[i:i+d]))
                            locdens_means.append(numpy.mean(sorted_local_dens[i:i+d]))
                        else:
                            #print "end"
                            binned_means_locdens.append(numpy.mean(masses_sorted_by_local_dens[i:]))
                            binned_stds_locdens.append(stats.sem(masses_sorted_by_local_dens[i:]))
                            locdens_means.append(numpy.mean(sorted_local_dens[i:i+d]))
                            break
                    all_binned_means_locdens.append(numpy.array(binned_means_locdens))
                    all_binned_stds_locdens.append(binned_stds_locdens)

                try:
                    all_means = numpy.mean(all_binned_means_locdens, axis=0)
                    devs = numpy.mean(all_binned_stds_locdens, axis=0)
                except:
                    max_len = 0
                    for a in all_binned_means_locdens:
                        if len(a) > max_len:
                            max_len = len(a)

                    new_sorted = []
                    new_stds = []
                    for a in all_binned_means_locdens:
                        b = numpy.pad(a, (max_len - len(a), 0), 'constant', constant_values=(min(a), max(a)))
                        # constant_values=(min([min(r) for r in all_initial])))
                        #print len(a), len(b), len(b)-len(a), max_len - len(a)
                        new_sorted.append(b)
                    for a in all_binned_stds_locdens:
                        b = numpy.pad(a, (max_len - len(a), 0), 'constant', constant_values=(min(a), max(a)))
                        # constant_values=(min([min(r) for r in all_initial])))
                        new_stds.append(b)
                    all_means = numpy.mean(new_sorted, axis=0)
                    devs = numpy.mean(new_stds, axis=0)

                locdens_means = numpy.pad(locdens_means, (len(all_means) - len(locdens_means), 0), 'constant', constant_values=(min(locdens_means), max(locdens_means)))

                all_means_high = all_means + devs
                all_means_low = all_means - devs

                #print all_means

                if t == 0.0:
                    axs[0].plot(locdens_means,
                                all_means,
                                color=colors[label],
                                lw=3,
                                ls=":")

                    axs[0].fill_between(locdens_means,
                                        all_means_high,
                                        all_means_low,
                                        facecolor=colors[label],
                                        alpha=0.2)

                elif t == 2.0:

                    axs[0].plot(locdens_means,
                                all_means,
                                color=colors[label],
                                lw=3,
                                label=labels[label])

                    axs[0].fill_between(locdens_means,
                                        all_means_high,
                                        all_means_low,
                                        facecolor=colors[label],
                                        alpha=0.2)

    # Observational points
    from astropy.table import Table
    data = Table.read('data/surfacedensities_var.dat', format='ascii.ecsv')
    point_ONC = [float(data[0]['surfdens_YSO']),
                 float(data[0]['Mavg']),
                 numpy.log(float(data[0]['sigma_ln_Mavg/Mearth'])),
                 data[0]['Region']]
    point_Lupus = [float(data[1]['surfdens_YSO']),
                   float(data[1]['Mavg']),
                   numpy.log(float(data[1]['sigma_ln_Mavg/Mearth'])),
                   data[1]['Region']]
    point_OMC2 = [float(data[2]['surfdens_YSO']),
                  float(data[2]['Mavg']),
                  numpy.log(float(data[2]['sigma_ln_Mavg/Mearth'])),
                  data[2]['Region']]
    point_NGCE = [float(data[3]['surfdens_YSO']),
                  float(data[3]['Mavg']),
                  numpy.log(float(data[3]['sigma_ln_Mavg/Mearth'])),
                  data[3]['Region']]
    point_NGCW = [float(data[4]['surfdens_YSO']),
                  float(data[4]['Mavg']),
                  numpy.log(float(data[4]['sigma_ln_Mavg/Mearth'])),
                  data[4]['Region']]
    point_Taurus = [float(data[5]['surfdens_YSO']),
                    float(data[5]['Mavg']),
                    numpy.log(float(data[5]['sigma_ln_Mavg/Mearth'])),
                    data[5]['Region']]

    surfdens = [point_ONC[0], point_Lupus[0], point_OMC2[0], point_NGCE[0], point_NGCW[0], point_Taurus[0]]
    obsmasses = [point_ONC[1], point_Lupus[1], point_OMC2[1], point_NGCE[1], point_NGCW[1], point_Taurus[1]]
    sigmas = [point_ONC[2], point_Lupus[2], point_OMC2[2], point_NGCE[2], point_NGCW[2], point_Taurus[2]]
    names = [point_ONC[3], point_Lupus[3], point_OMC2[3], point_NGCE[3], point_NGCW[3], point_Taurus[3]]

    markers, caps, bars = pyplot.errorbar(surfdens[:3], obsmasses[:3], yerr=sigmas[:3],
                                          color='navy',
                                          marker='D',
                                          markersize=10,
                                          #markeredgewidth=6,
                                          elinewidth=2,
                                          capsize=2,
                                          ls="None",
                                          #alpha=0.8,
                                          zorder=10)
    [bar.set_alpha(0.5) for bar in bars]

    markers, caps, bars = pyplot.errorbar(surfdens[3], obsmasses[3], yerr=sigmas[3],
                                          color='royalblue',
                                          marker='D',
                                          markersize=10,
                                          #markeredgewidth=6,
                                          elinewidth=2,
                                          capsize=2,
                                          ls="None",
                                          #alpha=0.5,
                                          zorder=12)
    [bar.set_alpha(0.5) for bar in bars]

    markers, caps, bars = pyplot.errorbar(surfdens[4:], obsmasses[4:], yerr=sigmas[4:],
                                          color='navy',
                                          marker='D',
                                          markersize=10,
                                          #markeredgewidth=6,
                                          elinewidth=2,
                                          capsize=2,
                                          ls="None",
                                          #alpha=0.5,
                                          zorder=15)
    [bar.set_alpha(0.5) for bar in bars]

    pyplot.scatter(21.97220445068663, 2.24,
                   color='red',
                   marker='D',
                   s=100,
    )

    # Positions for text labels
    xlocs = {data[0]['Region']: surfdens[0] * 1.25,  # ONC
             data[1]['Region']: surfdens[1] * 0.26,  # Lupus
             data[2]['Region']: surfdens[2] * 1.25,  # OMC-2
             data[3]['Region']: surfdens[3] * 1.2,  # NGC2024 East
             data[4]['Region']: surfdens[4] * 1.2,  # NGC2024 West
             data[5]['Region']: surfdens[5] * 0.22}  # Taurus

    ylocs = {data[0]['Region']: obsmasses[0] * 0.9,  # ONC
             data[1]['Region']: obsmasses[1] * 0.9,  # Lupus
             data[2]['Region']: obsmasses[2] * 0.9,  # OMC-2
             data[3]['Region']: obsmasses[3] * 0.9,  # NGC2024 East
             data[4]['Region']: obsmasses[4] * 0.9,  # NGC2024 West
             data[5]['Region']: obsmasses[5] * 0.9}  # Taurus

    for i, txt in enumerate(names):
        if txt == data[3]['Region']:
            axs[0].annotate(txt,
                            (xlocs[data[i]['Region']], ylocs[data[i]['Region']]),
                            fontsize=20,
                            color='royalblue')
        else:
            axs[0].annotate(txt,
                            (xlocs[data[i]['Region']], ylocs[data[i]['Region']]),
                            fontsize=20,
                            color='navy')

    axs[0].set_xlabel(r'Projected local stellar number density [pc$^{-2}$]')
    axs[0].set_ylabel(r'Binned mean disc dust mass [$\mathrm{M}_{\oplus}$]')

    first_legend = pyplot.legend([R01ShadedObject(), R03ShadedObject(), R05ShadedObject(),
                                  R1ShadedObject(), R25ShadedObject(), R5ShadedObject()],
                                  [labels['R01'], labels['R03'], labels['R05'],
                                   labels['R1'], labels['R25'], labels['R5'],
                                   ],
                                  handler_map={R01ShadedObject: R01ShadedObjectHandler(),
                                               R03ShadedObject: R03ShadedObjectHandler(),
                                               R05ShadedObject: R05ShadedObjectHandler(),
                                               R1ShadedObject: R1ShadedObjectHandler(),
                                               R25ShadedObject: R25ShadedObjectHandler(),
                                               R5ShadedObject: R5ShadedObjectHandler()},
                                  loc='lower left',
                                  ncol=2,
                                  fontsize=18, framealpha=0.4)

    pyplot.gca().add_artist(first_legend)

    pyplot.legend([DottedShadedObject(), SolidShadedObject()],
                  [r"t = 0.0 Myr",
                   r"t = 2.0 Myr",
                   ],
                  handler_map={DottedShadedObject: DottedShadedObjectHandler(),
                               SolidShadedObject: SolidShadedObjectHandler()},
                  loc='lower left',
                  bbox_to_anchor=(0.0, 0.2),
                  fontsize=18, framealpha=0.4)

    pyplot.xscale('log')
    pyplot.yscale('log')

    if save:
        pyplot.savefig('{0}/2D_dustmass_localdensity.png'.format(save_path))


def main(open_path, N, save_path, t_end, save, nruns, all_in_one):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    projected_mass_vs_local_density(open_path, save_path, t_end, N, nruns, save)

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
    result.add_option("-a", dest="all_in_one", type="int", default=1,
                      help="plot all in one single figure [%default]")
    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
