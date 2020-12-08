import numpy
from matplotlib import pyplot

from amuse.lab import *
from amuse import io

#movie command
#"ffmpeg -framerate 5 -i {0}/%01d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {0}/movie.mp4

from legends import *


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
                           [0.6 * height, 0.6 * height],
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
                           [0.6 * height, 0.6 * height],
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
                           [0.6 * height, 0.6 * height],
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


class R01ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R01'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R03ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R03'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R05ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R05'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R1ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R1'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R25ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R25'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R5ShadedDashedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width + 15],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls="--",
                           color=colors['R5'])  # Have to change color by hand for different plots
        """l2 = mlines.Line2D([x0, y0 + width + 10],
                           [0.2 * height, 0.2 * height],
                           lw=3, ls="--",
                           color=colors['R5'])"""
        l3 = patches.Rectangle(
            (x0, y0 + width - 46),  # (x,y)
            y0 + width + 15,  # width
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


class R01ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


class R03ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


class R05ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


class R1ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


class R25ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


class R5ShadedDottedObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0+2, y0 + width + 8],
                           [0.5 * height, 0.5 * height],
                           lw=3, ls=":",
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


def disk_fractions(open_path, save_path, t_end, N, nruns, save):
    """ Figure 2: Disk fractions in time.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """

    fig = pyplot.figure()
    dt = 0.005
    times = numpy.arange(0.0, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        try:
            label = path.split('/')[-2].split('_')[1]
        except:
            label = path.split('/')[-3].split('_')[1]

        all_fractions = []
        all_m = []
        for n in range(nruns):
            fractions = []
            mfractions = []
            run_path = '{0}/{1}'.format(path, n)
            last_t = 0.0
            init_disks = 0
            for t in times:
                try:
                    f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                    stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                    last_t = t
                except:
                    f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, last_t)
                    stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                if t == 0.0:
                    init_disks = float(len(stars[stars.disked]))
                disked_stars = stars[stars.disked]
                fraction = float(len(disked_stars)) / init_disks
                mfraction = numpy.mean(disked_stars.disk_mass.value_in(units.MJupiter))
                fractions.append(fraction)
                mfractions.append(mfraction)
            all_fractions.append(fractions)
            all_m.append(mfractions)

        mean_fractions = numpy.mean(all_fractions, axis=0)
        fractions_stdev = numpy.std(all_fractions, axis=0)

        fractions_high = mean_fractions + fractions_stdev
        fractions_low = mean_fractions - fractions_stdev

        pyplot.fill_between(times,
                            fractions_high,
                            fractions_low,
                            facecolor=colors[label],
                            alpha=0.2)
        pyplot.plot(times, mean_fractions, lw=3, label=labels[label], color=colors[label], ls="-")

    pyplot.xlabel('Time [Myr]', fontsize=28)
    pyplot.ylabel(r'Disc fraction', fontsize=28)

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
                  ncol=2,
                  fontsize=22, framealpha=0.4)

    pyplot.xlim([0.0, t_end])
    pyplot.ylim([0.0, 1.0])

    if save:
        pyplot.savefig('{0}/disc_fractions.png'.format(save_path, N))


def separated_masses(open_path, save_path, t_end, N, nruns, save):
    """ Figure 3 in paper: disk fractions separated in stellar masses M* < 0.5MSun
        and 0.5MSun < M* < 1.9MSun.

    :param open_path: path to open results
    :param save_path: path to save figure
    :param t_end: final time to plot
    :param N: number of stars in results
    :param nruns: number of runs to plot
    :param save: if True, save figure in save_path
    """
    fig, axs = pyplot.subplots(2, 1,
                               figsize=(10, 12),
                               sharex=True)
    dt = 0.005
    times = numpy.arange(0.0, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        try:
            label = path.split('/')[-2].split('_')[1]
        except:
            label = path.split('/')[-3].split('_')[1]

        all_fractions_low_mass = []
        all_fractions_high_mass = []
        for n in range(nruns):
            fractions_low = []
            fractions_high = []
            run_path = '{0}/{1}'.format(path, n)
            init_disks = 0
            for t in times:
                f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                disked_stars = stars[stars.disked]

                if t == 0.0:
                    init_disks = float(len(disked_stars))

                low_mass = disked_stars[disked_stars.stellar_mass <= 0.5 | units.MSun ]
                high_mass = disked_stars[disked_stars.stellar_mass > 0.5 | units.MSun ]

                fraction_low = float(len(low_mass)) / init_disks
                fractions_low.append(fraction_low)

                fraction_high = float(len(high_mass)) / init_disks
                fractions_high.append(fraction_high)

            all_fractions_low_mass.append(fractions_low)
            all_fractions_high_mass.append(fractions_high)

        mean_fractions_low = numpy.mean(all_fractions_low_mass, axis=0)
        fractions_stdev_low = numpy.std(all_fractions_low_mass, axis=0)
        fractions_high_low = mean_fractions_low + fractions_stdev_low
        fractions_low_low = mean_fractions_low - fractions_stdev_low

        axs[0].fill_between(times,
                            fractions_high_low,
                            fractions_low_low,
                            facecolor=colors[label],
                            alpha=0.2)
        axs[0].plot(times,
                    mean_fractions_low,
                    lw=3,
                    label=labels[label],
                    color=colors[label],
                    ls=":")

        mean_fractions_high = numpy.mean(all_fractions_high_mass, axis=0)
        fractions_stdev_high = numpy.std(all_fractions_high_mass, axis=0)
        fractions_high_high = mean_fractions_high + fractions_stdev_high
        fractions_low_high = mean_fractions_high - fractions_stdev_high

        if label == 'R05' or label == 'R5':
            axs[1].fill_between(times,
                                fractions_high_high,
                                fractions_low_high,
                                facecolor=colors[label],
                                alpha=0.2)
        axs[1].plot(times,
                    mean_fractions_high,
                    lw=3,
                    label=labels[label],
                    color=colors[label],
                    ls='--')

    axs[0].set_xlabel('Time [Myr]', fontsize=26)
    axs[0].xaxis.set_tick_params(labelbottom=True)
    axs[1].set_xlabel('Time [Myr]', fontsize=26, labelpad=15)
    axs[0].set_ylabel(r'Disc fraction', fontsize=26)
    axs[1].set_ylabel(r'Disc fraction', fontsize=26)

    axs[0].legend([R01ShadedObject(), R03ShadedObject(), R05ShadedObject(),
                  R1ShadedObject(), R25ShadedObject(), R5ShadedObject()],
                  [labels['R01'], labels['R03'], labels['R05'],
                   #"", "",
                   labels['R1'], labels['R25'], labels['R5'],
                   ],
                  handler_map={R01ShadedObject: R01ShadedDottedObjectHandler(),
                               R03ShadedObject: R03ShadedDottedObjectHandler(),
                               R05ShadedObject: R05ShadedDottedObjectHandler(),
                               R1ShadedObject: R1ShadedDottedObjectHandler(),
                               R25ShadedObject: R25ShadedDottedObjectHandler(),
                               R5ShadedObject: R5ShadedDottedObjectHandler()},
                  loc='upper right',
                  ncol=2,
                  fontsize=22, framealpha=0.4)

    axs[1].legend([R01ShadedObject(), R03ShadedObject(), R05ShadedObject(),
                  R1ShadedObject(), R25ShadedObject(), R5ShadedObject()],
                  [labels['R01'], labels['R03'], labels['R05'],
                   #"", "",
                   labels['R1'], labels['R25'], labels['R5'],
                   ],
                  handler_map={R01ShadedObject: R01ShadedDashedObjectHandler(),
                               R03ShadedObject: R03ShadedDashedObjectHandler(),
                               R05ShadedObject: R05ShadedDashedObjectHandler(),
                               R1ShadedObject: R1ShadedDashedObjectHandler(),
                               R25ShadedObject: R25ShadedDashedObjectHandler(),
                               R5ShadedObject: R5ShadedDashedObjectHandler()},
                  loc='lower left',
                  ncol=2,
                  fontsize=22, framealpha=0.4)

    axs[0].text(1.1, 0.55, r'$\mathrm{M}_* \leq 0.5 \mathrm{M}_{\odot}$', fontsize=28)
    axs[1].text(0.25, 0.105, r'$0.5 \mathrm{M}_{\odot} < \mathrm{M}_* \leq 1.9 \mathrm{M}_{\odot}$', fontsize=28)

    fig.subplots_adjust(top=0.96, hspace=0.2)

    axs[0].set_xlim([0.0, t_end])
    axs[1].set_xlim([0.0, t_end])
    axs[0].set_ylim([0.0, 1.0])

    if save:
        pyplot.savefig('{0}/separate_disc_fractions.png'.format(save_path, N))


def count_stars(open_path, N, nruns):
    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        disked_masses, disked_stds = [], []
        bright_masses, bright_stds = [], []
        disked_fractions = []
        bright_fractions = []
        disked = []
        bright = []
        for n in range(nruns):
            run_path = '{0}/{1}'.format(path, n)
            f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, 0.0)
            stars = io.read_set_from_file(f, 'hdf5', close_file=True)
            disked_stars = stars[stars.disked]
            bright_stars = stars[stars.bright]
            print "{0} || disked: {1}, {2} || bright: {3}, {4}".format(folder,
                                                                       len(disked_stars),
                                                                       float(len(disked_stars))/len(stars),
                                                                       len(bright_stars),
                                                                       float(len(bright_stars))/len(stars))
            print "TOTAL MASS: {0}".format(stars.stellar_mass.sum().value_in(units.MSun))
            disked.append(len(disked_stars))
            bright.append(len(bright_stars))
            disked_fractions.append(float(len(disked_stars))/len(stars))
            bright_fractions.append(float(len(bright_stars))/len(stars))

            disked_masses.append(disked_stars.stellar_mass.value_in(units.MSun))
            bright_masses.append(bright_stars.stellar_mass.value_in(units.MSun))

            disked_stds.append(numpy.std(disked_stars.stellar_mass.value_in(units.MSun)))
            bright_stds.append(numpy.std(bright_stars.stellar_mass.value_in(units.MSun)))

        print "MEANS"
        print "Ndisked: {0} +- {1} || Nbright: {2} +- {3}".format(numpy.mean(disked),
                                                                  numpy.std(disked),
                                                                  numpy.mean(bright),
                                                                  numpy.std(bright))
        print "Ndisked: {0} +- {1} || Nbright: {2} +- {3}".format(numpy.mean(disked_fractions),
                                                                  numpy.std(disked_fractions),
                                                                  numpy.mean(bright_fractions),
                                                                  numpy.std(bright_fractions))
        flat_disked_masses = [item for sublist in disked_masses for item in sublist]
        flat_bright_masses = [item for sublist in bright_masses for item in sublist]

        print "M*disked: {0} -{1}+{2} MSun || M*bright: {3}-{4}+{5}".format(numpy.mean(flat_disked_masses),
                                                                            numpy.mean(flat_disked_masses) - min(flat_disked_masses),
                                                                            max(flat_disked_masses) - numpy.mean(flat_disked_masses),
                                                                            numpy.mean(flat_bright_masses),
                                                                            numpy.mean(bright_stds) - min(flat_bright_masses),
                                                                            max(flat_bright_masses) - numpy.mean(flat_bright_masses))
        print max(flat_bright_masses)
        print "*"


def disk_survival(open_path, N, nruns, t_end):
    from lifelines import KaplanMeierFitter
    from collections import defaultdict

    dt = 0.005
    times = numpy.arange(0.0, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        all_times = []
        all_stds = []
        print folder

        durations = []
        event_observed = []

        for n in range(nruns):
            run_path = '{0}/{1}'.format(path, n)
            dispersed_times = []
            prev_t = 0.0
            for t in times:
                prev_f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, prev_t)
                prev_stars = io.read_set_from_file(prev_f, 'hdf5', close_file=True)
                f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                for s in range(len(stars)):
                    if not stars[s].disked and prev_stars[s].disked:
                        dispersed_times.append(t)
                        durations.append(t)
                        event_observed.append(1)
                    elif not stars[s].disked and not prev_stars[s].disked:
                        durations.append(t)
                        event_observed.append(1)
                    else:
                        durations.append(t)
                        event_observed.append(0)

                prev_t = t
            all_times.append(numpy.mean(dispersed_times))
            all_stds.append(numpy.std(dispersed_times))

        #durations = durations_dict.keys()
        #event_observed = durations_dict.values()

        #print durations
        #print event_observed

        print numpy.mean(all_times)
        print numpy.mean(all_stds)

        ## create a kmf object
        kmf = KaplanMeierFitter()

        ## Fit the data into the model
        kmf.fit(durations, event_observed, label=folder)
        print kmf.median_

        ## Create an estimate
        kmf.plot()


def disk_lifetimes(open_path, N, nruns, t_end):
    dt = 0.005
    times = numpy.arange(0.0, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        all_times = []
        all_stds = []
        print folder
        for n in range(nruns):
            run_path = '{0}/{1}'.format(path, n)
            dispersed_times = []
            prev_t = 0.0
            for t in times:
                prev_f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, prev_t)
                prev_stars = io.read_set_from_file(prev_f, 'hdf5', close_file=True)
                f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)

                for s in range(len(stars)):
                    if not stars[s].disked and prev_stars[s].disked:
                        dispersed_times.append(t)

                prev_t = t
            all_times.append(numpy.mean(dispersed_times))
            all_stds.append(numpy.std(dispersed_times))

        print numpy.mean(all_times)
        print numpy.mean(all_stds)


def disks_halflife(open_path, N, nruns, t_end):
    dt = 0.005
    times = numpy.arange(0.0, t_end + dt, dt)

    for folder in folders:
        path = '{0}/{1}/'.format(open_path, folder)
        print folder
        half_times = []
        for n in range(nruns):
            run_path = '{0}/{1}'.format(path, n)
            init_f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, 0.0)
            init_stars = io.read_set_from_file(init_f, 'hdf5', close_file=True)
            init_disks = len(init_stars[init_stars.disked])
            for t in times:
                f = '{0}/N{1}_t{2:.3f}.hdf5'.format(run_path, N, t)
                stars = io.read_set_from_file(f, 'hdf5', close_file=True)
                t_disks = len(stars[stars.disked])

                if t_disks <= 0.5 * init_disks:
                    half_times.append(t)
                    break

        print numpy.mean(half_times)
        print numpy.std(half_times)


def main(open_path, N, save_path, t_end, save, nruns):

    # My own stylesheet, comment out if not needed
    pyplot.style.use('paper')

    separated_masses(open_path, save_path, t_end, N, nruns, save)
    #disk_fractions(open_path, save_path, t_end, N, nruns, save)

    #count_stars(open_path, N, nruns)
    #disk_lifetimes(open_path, N, nruns, t_end)
    #disks_halflife(open_path, N, nruns, t_end)
    #disk_survival(open_path, N, nruns, t_end)

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
