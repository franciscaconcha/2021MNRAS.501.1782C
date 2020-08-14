import numpy
from matplotlib import pyplot
from scipy import stats
import pandas

from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *

# Re-defining some legend stuff for these specific plots
class R01ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0 + 2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R01'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R01'])
        l3 = patches.Rectangle( (x0, y0 + width - 43),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R01'],
            facecolor=colors['R01'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


class R03ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 10],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R03'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R03'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 36),  # (x,y)
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
        #handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l2]


class R05ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 10],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R05'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R05'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 36),  # (x,y)
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
        #handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l2]


class R1ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 10],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R1'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R1'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 36),  # (x,y)
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
        #handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l2]


class R25ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 10],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
                           color=colors['R25'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R25'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 36),  # (x,y)
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
        #handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l2]


class R5ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0 + 2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R5'])  # Have to change color by hand for different plots
        l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])
        l3 = patches.Rectangle(
            (x0, y0 + width - 42),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R5'],
            facecolor=colors['R5'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


def separated_masses(open_path, save_path, t_end, N, nruns, save):
    """ Figure 7: Cumulative distributions of disc dust mass separated in stellar masses M* < 0.5MSun
        and 0.5MSun < M* < 1.9MSun.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """
    fig, axes = pyplot.subplots(1)

    all_final_low, all_final_high = [], []

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        try:
            label = path.split('/')[-2].split('_')[1]
        except:
            label = path.split('/')[-3].split('_')[1]

        all_cumulative_low = []
        all_cumulative_high = []

        for n in range(nruns):
            p1 = '{0}/{1}'.format(path, n)
            f1 = '{0}/N{1}_t{2:.3f}.hdf5'.format(p1, N, t_end)
            final_stars = io.read_set_from_file(f1, 'hdf5', close_file=True)
            final_stars = final_stars[final_stars.disked == True]

            final_low = final_stars[final_stars.stellar_mass <= 0.5 | units.MSun]
            final_high = final_stars[final_stars.stellar_mass > 0.5 | units.MSun]

            final_disk_masses_low = numpy.sort(final_low.disk_mass.value_in(units.MEarth) / 100.)
            final_disk_masses_high = numpy.sort(final_high.disk_mass.value_in(units.MEarth) / 100.)

            all_final_low.append(final_disk_masses_low)
            all_final_high.append(final_disk_masses_high)

            all_cumulative_low.append(1. * numpy.arange(len(final_disk_masses_low)) / (len(final_disk_masses_low) - 1))
            all_cumulative_high.append(1. * numpy.arange(len(final_disk_masses_high)) / (len(final_disk_masses_high) - 1))

        try:
            final_low_mean = numpy.mean(all_final_low, axis=0)
            final_low_std = numpy.std(all_cumulative_low, axis=0)
        except ValueError:
            max_len = 0
            for a in all_final_low:
                if len(a) > max_len:
                    max_len = len(a)

            new_sorted = []
            new_cumulative = []
            for a in all_final_low:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_final_low])))
                new_sorted.append(b)
            for a in all_cumulative_low:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_cumulative_low])))
                new_cumulative.append(b)

            final_low_mean = numpy.mean(new_sorted, axis=0)
            final_low_std = numpy.std(new_cumulative, axis=0)

        final_low_cumulative = 1. * numpy.arange(len(final_low_mean)) / (len(final_low_mean) - 1)

        final_low_high = final_low_cumulative + final_low_std
        final_low_low = final_low_cumulative - final_low_std

        if label == 'R5' or label == 'R01':
            axes.fill_between(final_low_mean,
                              1.0 - final_low_high,
                              1.0 - final_low_low,
                              facecolor=colors[label],
                              alpha=0.2)

        axes.plot(final_low_mean,
                  1.0 - final_low_cumulative,
                  c=colors[label],
                  ls=':',
                  lw=3, label=r"$M \leq 0.5 M_{{\odot}}$, {0} Myr".format(2.0))

        try:
            final_high_mean = numpy.mean(all_final_high, axis=0)
            final_high_std = numpy.std(all_cumulative_high, axis=0)
        except ValueError:
            max_len = 0
            for a in all_final_high:
                if len(a) > max_len:
                    max_len = len(a)

            new_sorted = []
            new_cumulative = []
            for a in all_final_high:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_final_high])))
                new_sorted.append(b)
            for a in all_cumulative_high:
                b = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_cumulative_high])))
                new_cumulative.append(b)

            final_high_mean = numpy.mean(new_sorted, axis=0)
            final_high_std = numpy.std(new_cumulative, axis=0)

        final_high_cumulative = 1. * numpy.arange(len(final_high_mean)) / (len(final_high_mean) - 1)

        final_high_high = final_high_cumulative + final_high_std
        final_high_low = final_high_cumulative - final_high_std

        if label == 'R5' or label == 'R01':
            axes.fill_between(final_high_mean,
                              1.0 - final_high_high,
                              1.0 - final_high_low,
                              facecolor=colors[label],
                              alpha=0.2)

        axes.plot(final_high_mean,
                  1.0 - final_high_cumulative,
                  c=colors[label],
                  ls='--',
                  lw=3,
                  label=r"$M > 0.5 M_{{\odot}}$, {0} Myr".format(2.0))

    axes.set_xscale('log')
    axes.set_ylim([0.0, 1.0])

    axes.set_xlabel(r'$\mathrm{M}_{disc, dust}$ [$\mathrm{M}_{\oplus}$]')
    axes.set_ylabel(r'$f_{\geq \mathrm{M}_{disc, dust}}$')

    first_legend = pyplot.legend([R01ShadedObject(), R03ShadedObject(), R05ShadedObject(),
                                  R1ShadedObject(), R25ShadedObject(), R5ShadedObject()],
                                  [labels['R01'], labels['R03'], labels['R05'],
                                   labels['R1'], labels['R25'], labels['R5']],
                                  handler_map={R01ShadedObject: R01ShadedObjectHandler(),
                                               R03ShadedObject: R03ShadedObjectHandler(),
                                               R05ShadedObject: R05ShadedObjectHandler(),
                                               R1ShadedObject: R1ShadedObjectHandler(),
                                               R25ShadedObject: R25ShadedObjectHandler(),
                                               R5ShadedObject: R5ShadedObjectHandler()},
                                  loc='upper right',
                                  ncol=2,
                                  fontsize=16, framealpha=0.4)

    pyplot.gca().add_artist(first_legend)

    pyplot.legend([DottedShadedObject(), DashedShadedObject()],
                  [r"$\mathrm{M}_* \leq 0.5 \mathrm{M}_{{\odot}}$",
                   r'$0. 5 \mathrm{M}_{\odot} < \mathrm{M}_* \leq 1.9 \mathrm{M}_{\odot}$',
                   ],
                  handler_map={DottedShadedObject: DottedShadedObjectHandler(),
                               DashedShadedObject: DashedShadedObjectHandler(),},
                  loc='upper left',
                  bbox_to_anchor=(0.665, 0.8),
                  fontsize=16, framealpha=0.4)

    if save:
        fig.savefig('{0}/separated_cumulative_masses.png'.format(save_path))


def cumulative_masses(open_path, save_path, t_end, N, nruns, save):
    """ Figure 7: Cumulative distributions of disc dust mass.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """
    fig, axes = pyplot.subplots(1)

    for folder in folders:
        B = []
        O = []

        i_folder = folders.index(folder)
        path = '{0}/{1}/'.format(open_path, folder)
        try:
            label = path.split('/')[-2].split('_')[1]
        except:
            label = path.split('/')[-3].split('_')[1]

        all_cumulative = []
        all_masses = []

        for n in range(nruns):
            # Final distributions
            p1 = '{0}/{1}'.format(path, n)
            f1 = '{0}/N{1}_t{2:.3f}.hdf5'.format(p1, N, t_end)
            final_stars = io.read_set_from_file(f1, 'hdf5', close_file=True)
            final_stars = final_stars[final_stars.disked == True]

            sorted_masses = numpy.sort(final_stars.disk_mass.value_in(units.MEarth) / 100.)
            cumulative = 1. * numpy.arange(len(sorted_masses)) / (len(sorted_masses) - 1)

            all_cumulative.append(cumulative)
            all_masses.append(sorted_masses)

        try:
            masses_mean = numpy.mean(all_masses, axis=0)
        except ValueError:
            max_len = 0
            for a in all_masses:
                if len(a) > max_len:
                    max_len = len(a)

            #new_sorted = []
            new_masses = []
            for a in all_masses:
                c = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_masses])))
                new_masses.append(c)

            masses_mean = numpy.mean(new_masses, axis=0)

        mean_cumulative = 1. * numpy.arange(len(masses_mean)) / (len(masses_mean) - 1)

        try:
            masses_std = numpy.std(all_cumulative, axis=0)
        except ValueError:
            max_len = 0
            for a in all_cumulative:
                if len(a) > max_len:
                    max_len = len(a)

            new_cumulative = []
            for a in all_cumulative:
                c = numpy.pad(a, (max_len - len(a), 0), 'constant',
                              constant_values=(min([min(r) for r in all_cumulative])))
                new_cumulative.append(c)

            masses_std = numpy.std(new_cumulative, axis=0)

        all_high = mean_cumulative + masses_std
        all_low = mean_cumulative - masses_std

        if label == 'R01' or label == 'R5':
            axes.fill_between(masses_mean,
                              1.0 - all_high,
                              1.0 - all_low,
                              facecolor=colors[label],
                              alpha=0.2)

        axes.plot(masses_mean,
                  1.0 - mean_cumulative,
                  c=colors[label],
                  #alpha=0.4,
                  ls='-',
                  lw=3, label=r"$M \leq 0.5 M_{{\odot}}$, {0} Myr".format(2.0))

    # Plotting observational data
    # OMC-2
    # Masses are sorted in the table
    OMC2 = pandas.read_csv('data/OMC-2_vanTerwisga2019.txt',
                           sep='\xc2\xb1',
                           names=['disk_mass', 'error'],
                           skiprows=3,
                           dtype=numpy.float64)

    OMC2_higher, OMC2_lower = [], []

    for m, e in zip(OMC2.disk_mass, OMC2.error):
        OMC2_higher.append(m + e)
        OMC2_lower.append(m - e)

    OMC2_cumulative = 1. * numpy.arange(len(OMC2.disk_mass)) / (len(OMC2.disk_mass) - 1)
    OMC2_error_cumulative = 1. * numpy.arange(len(OMC2.error)) / (len(OMC2.error) - 1)
    OMC2_higher_cumulative = 1. * numpy.arange(len(OMC2_higher)) / (len(OMC2_higher) - 1)
    OMC2_lower_cumulative = 1. * numpy.arange(len(OMC2_lower)) / (len(OMC2_lower) - 1)

    #print OMC2_error_cumulative
    #print OMC2_higher
    #print OMC2_lower

    axes.plot(OMC2.disk_mass, OMC2_cumulative, c='r', lw=3, label="OMC-2")

    axes.fill_between(OMC2.disk_mass,
                      OMC2_higher,
                      OMC2_lower,
                      facecolor='r',
                      alpha=0.2)

    # ONC
    ONC_Eisner2018 = pandas.read_csv('data/ONC_Eisner2018.txt',
                                     sep='&',
                                     names=['ID', 'alpha', 'delta', 'M_star', 'F_{\rm \lambda 850 \mu m}', 'F_{\rm dust}',
                                            'M_dust', 'R_disk'],
                                     skiprows=4)

    ONC_masses, ONC_error = [], []

    for me in ONC_Eisner2018.M_dust:
        m, e = me.split('$\pm$')
        ONC_masses.append(float(m))
        ONC_error.append(float(e))

    ONC_Mann2014 = pandas.read_csv('data/ONC_Mann2014.txt',
                                     sep='\t',
                                     names=['Field', 'Name', 'alpha', 'delta', 'M_star',
                                            'F_{\rm \lambda 850 \mu m}', 'F_{\rm dust}',
                                            'M_dust', 'd', 'Maj', 'Min', 'P.A.', 'Notes'],
                                     skiprows=7)

    for me in ONC_Mann2014.M_dust:
        try:
            m, e = me.split('+or-')
        except ValueError:  # For the *one* row that is different
            m = me.split('<or=')[1]
            e = 0.0

        ONC_masses.append(float(m))
        ONC_error.append(float(e))

    ONC_masses.sort()

    ONC_masses_cumulative = 1. * numpy.arange(len(ONC_masses)) / (len(ONC_masses) - 1)

    axes.plot(ONC_masses[::-1], ONC_masses_cumulative, c='b', lw=3, label="ONC")

    """axes.fill_between(OMC2['disk_mass'],
                      OMC2_higher,
                      OMC2_lower,
                      facecolor='r',
                      alpha=0.2)"""

    # Lupus
    Lupus_Ansdell2016 = pandas.read_csv('data/Lupus_Ansdell2016.txt',
                                     sep='&',
                                     names=['Name', 'RAh', 'DE-',
                                            'Fcont', 'e_Fcont',
                                            'rms', 'a', 'e_a', 'PosAng', 'e_PosAng', 'i', 'e_i',
                                            'M_dust', 'e_M_dust'],
                                     skiprows=31)

    Lupus_Ansdell2018 = pandas.read_csv('data/Lupus_Ansdell2018.txt',
                                     sep='&',
                                     names=['everything_else',
                                            'M_dust', 'e_M_dust'],
                                     skiprows=47)

    Lupus_masses = numpy.concatenate([Lupus_Ansdell2016['M_dust'].to_numpy(), Lupus_Ansdell2018['M_dust'].to_numpy()])
    Lupus_errors = numpy.concatenate([Lupus_Ansdell2016['e_M_dust'].to_numpy(), Lupus_Ansdell2018['e_M_dust'].to_numpy()])

    Lupus_masses.sort()
    # todo sort errors

    Lupus_masses_cumulative = 1. * numpy.arange(len(Lupus_masses)) / (len(Lupus_masses) - 1)
    axes.plot(Lupus_masses[::-1], Lupus_masses_cumulative, c='g', lw=3, label="Lupus")

    axes.set_xscale('log')
    axes.set_xlim([0.01, 500.0])
    axes.set_ylim([0.0, 1.0])

    axes.set_xlabel(r'$\mathrm{M}_{disc, dust}$ [$\mathrm{M}_{\oplus}$]')
    axes.set_ylabel(r'$f_{\geq \mathrm{M}_{disc, dust}}$', fontsize=24)

    pyplot.legend([R01ShadedObject(), R03Object(), R05Object(),
                                  R1Object(), R25Object(), R5ShadedObject()],
                                  [labels['R01'], labels['R03'], labels['R05'],
                                   #"", "",
                                   labels['R1'], labels['R25'], labels['R5'],
                                   ],
                                  handler_map={R01ShadedObject: R01ShadedObjectHandler(),
                                               R03Object: R03ObjectHandler(),
                                               R05Object: R05ObjectHandler(),
                                               R1Object: R1ObjectHandler(),
                                               R25Object: R25ObjectHandler(),
                                               R5ShadedObject: R5ShadedObjectHandler()},
                                  loc='lower left',
                                  #bbox_to_anchor=(0.52, -0.4),
                                  ncol=2,
                                  fontsize=20, framealpha=0.4)

    #pyplot.title(r'$\mathrm{N} = 10^3$')

    if save:
        fig.savefig('{0}/cumulative_masses.png'.format(save_path))


def main(open_path, N, save_path, t_end, save, nruns):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    #separated_masses(open_path, save_path, t_end, N, nruns, save)
    cumulative_masses(open_path, save_path, t_end, N, nruns, save)

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
    result.add_option("-n", dest="nruns", type="int", default=5,
                      help="number of runs to plot for averages [%default]")
    return result


if __name__ == '__main__':
    o, arguments = new_option_parser().parse_args()
    main(**o.__dict__)
