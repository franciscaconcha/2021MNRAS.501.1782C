import numpy
from matplotlib import pyplot
from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *  # My own custom legend definitions

G0 = 1.6e-3 * units.erg / units.s / units.cm**2

class R01ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R01'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
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
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R03'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R03'],
            facecolor=colors['R03'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


class R05ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R05'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R05'],
            facecolor=colors['R05'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


class R1ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R1'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R1'],
            facecolor=colors['R1'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


class R25ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R25'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 10,  # width
            1.4 * height,  # height
            fill=colors['R25'],
            facecolor=colors['R25'],
            # edgecolor="black",
            alpha=0.2,
            # hatch="/",
        )
        handlebox.add_artist(l1)
        #handlebox.add_artist(l2)
        handlebox.add_artist(l3)
        #return [l1, l2, l3]
        return [l1, l3]


class R5ShadedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="-",
                           color=colors['R5'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
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


def mean_rad_in_time(open_path, save_path, t_end, N, nruns, save):
    """ Figure 4.Mean FUV radiation field received by the circumstellar discsin time.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """
    axs = [None, None]
    fig, axs[0] = pyplot.subplots(1, figsize=(12, 11))

    dt = 0.02
    times = numpy.arange(dt, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}'.format(open_path, folder)
        all_radiation = []
        all_stds = []
        for t in times:
            label = folder.split('_')[1]
            n_radiation = []
            n_std = []
            for n in range(nruns):
                f = '{0}/{1}/N{2}_t{3:.3f}.hdf5'.format(path, n, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                stars = stars[stars.disked == True]
                n_radiation.append(numpy.mean(stars.total_radiation))
                n_std.append(numpy.std(stars.total_radiation))
            all_radiation.append(numpy.mean(n_radiation))
            all_stds.append(numpy.mean(n_std))
        print all_radiation

        all_radiation = numpy.array(all_radiation)
        all_stds = numpy.array(all_stds)

        if label == 'R1' or label == 'R05':
            pyplot.fill_between(times,
                                all_radiation + all_stds,
                                all_radiation - all_stds,
                                facecolor=colors[label],
                                alpha=0.2)
        pyplot.plot(times, all_radiation,
                    c=colors[label],
                    lw=3,
                    label=label)

    pyplot.yscale('log')

    pyplot.xlim([0.0, t_end])

    pyplot.xlabel('Time [Myr]')
    pyplot.ylabel(r'Mean FUV field [$\mathrm{G}_0$]')

    fig.subplots_adjust(top=0.98, bottom=0.2, wspace=0.33)

    pyplot.legend([R01Object(), R03Object(), R05ShadedObject(),
                  R1ShadedObject(), R25Object(), R5Object()],
                  [labels['R01'], labels['R03'], labels['R05'],
                   #"", "",
                   labels['R1'], labels['R25'], labels['R5'],
                   ],
                  handler_map={R01Object: R01ObjectHandler(),
                               R03Object: R03ObjectHandler(),
                               R05ShadedObject: R05ShadedObjectHandler(),
                               R1ShadedObject: R1ShadedObjectHandler(),
                               R25Object: R25ObjectHandler(),
                               R5Object: R5ObjectHandler()},
                  loc='lower left',
                  bbox_to_anchor=(0.1, -0.25),
                  ncol=3,
                  fontsize=20, framealpha=0.4)

    if save:
        pyplot.savefig('{0}/radiation_vs_time.png'.format(save_path))


def main(open_path, N, save_path, t_end, save, nruns):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    mean_rad_in_time(open_path, save_path, t_end, N, nruns, save)

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
