#!/usr/bin/python

import collections as col
import time
import psutil as psu
import curses as crc
import argparse as arg
# import pynvml as nvidia   # TO DO

# ########## PILE OF GRAPH ABSTRACTIONS ##########


class Graph:
    """Basic class for dynamic graph."""

    def __init__(self, discrete: int, track_size: int):
        super().__init__()
        self.discrete = discrete
        # -1 - under; {0, ..., self.discerete-1} - bins; self.discrete - over;
        self.observations = col.deque()
        self.size = track_size

        self.log_all = False
        self.log = None
        self.log_path = 'log.csv'

    def start_log(self) -> None:
        """Allows graph to start recording log."""
        self.log_all = True
        self.log = []

    def stop_log(self):
        """Stops log's recording."""
        self.log_all = False
        temp = self.log
        self.log = None
        return temp

    def flush_log(self) -> None:
        """Flushes log if it was recorded."""
        if self.log_all:
            self.log = []

    @property
    def track_size(self) -> int:
        """Size of track - graph timeline width."""
        return self.size

    @track_size.setter
    def track_size(self, size: int) -> int:
        if len(self.observations) > size:
            pops = len(self.observations) - size
            for i in range(pops):
                self.observations.popleft()
        self.size = size
        return size

    def all_observations(self) -> list[list[int]]:  # discrete
        """Returns discrete values of observation on current timeline."""
        return [obs for obs in self.observations]

    def add_observation(self) -> list[float]:
        """Informs graph to record new observation and update timeline/log."""
        pass

    def log_observation(self, obsrv) -> None:
        """Internal (called from 'add_observation()') call for log recoring."""
        if self.log_all:
            self.log.append((time.time(), obsrv))

    def row_num(self) -> int:
        """Returns count of graph rows (height of timeline)."""
        pass

    def push_observation(self, discr) -> None:
        """Internal call to update timeline."""
        self.observations.append(discr)
        if len(self.observations) > self.size:
            self.observations.popleft()

    row_name = 'ROW'
    log_name = 'log'

    def text_repr(self) -> str:
        """Textual graph repesentation for testing and debug."""
        # observ = self.observations
        observ = self.all_observations()
        rows = self.row_num()
        row_name = self.__class__.row_name
        return '\n'.join([f"{row_name}{i+1:02d}: " +
                          ''.join([' ' if len(observ[j]) <= i else '#'
                                   if observ[j][i] == -1 else str(observ[j][i])
                                   for j in range(len(observ))])
                          for i in range(rows)])


def compile_log(graphs: list, time_graph=None, row_names=None, delim=','
                ) -> str:
    """Transforms graphs logs into CSV text."""
    # time_graph - source of timestamps
    if time_graph is None:
        time_graph = graphs[0]
    if row_names is None:
        row_names = [[f"{graph.__class__.log_name}{i+1:02d}"
                      for i in range(graph.row_num())] for graph in graphs]
    colnames = 'time' + delim + delim.join([delim.join(graph_names)
                                            for graph_names in row_names])
    timestamps = [time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(observ[0]))
                  for observ in time_graph.log]
    data = '\n'.join([timestamps[i] + delim + delim.join([
        delim.join([str(obs) for obs in graph.log[i][1]])
        for graph in graphs]) for i in range(len(timestamps))])
    return colnames + '\n' + data


class CpuGraph(Graph):
    """Graphs for CPU cores statistics."""

    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)

    row_name = 'CPU'

    def row_num(self) -> int:
        return psu.cpu_count()


class FreqGraph(CpuGraph):
    """Graph of CPU frequences."""
    log_name = 'freq'

    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)
        self.mins = None
        self.maxs = None
        self.update_min_max = True

    def add_observation(self) -> list[float]:
        freqs = psu.cpu_freq(True)
        self.log_observation([frq.current for frq in freqs])
        discr = [(frq.current - frq.min) / (frq.max - frq.min)
                 for frq in freqs]
        discr = [-1 if frq < 0 else self.discrete if frq >= 1 else
                 int(frq * self.discrete) for frq in discr]
        self.push_observation(discr)
        if self.update_min_max:
            self.mins = [frq.min for frq in freqs]
            self.maxs = [frq.max for frq in freqs]
            self.update_min_max = False
        return [frq.current for frq in freqs]


class PercGraph(CpuGraph):
    """Graph of CPU occupancy percentage."""
    log_name = 'perc'

    def refresh(self) -> list[float]:
        return psu.cpu_percent(percpu=True)

    def __init__(self, discrete: int, track_size: int, refresh_already=False):
        super().__init__(discrete, track_size)
        if refresh_already:     # does it actaully work?
            self.refresh()

    def add_observation(self) -> list[float]:
        perc = self.refresh()
        self.log_observation(perc)
        discr = [prc / 100 for prc in perc]
        discr = [-1 if prc <= 0 else self.discrete if prc >= 1 else
                 int(prc * self.discrete) for prc in discr]
        self.push_observation(discr)
        return perc


class StaticGraph(Graph):
    """Static graph (without a timeline)."""

    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)
        self.observations = [[-1] * self.row_num()] * self.size

    @property
    def track_size(self) -> int:
        """Static graph width."""
        return self.size

    @track_size.setter
    def track_size(self, size: int) -> int:
        if len(self.observations) > size:
            self.observations = self.observations[:size]
        else:
            self.observations.extend(
                [[-1] * self.row_num()] * (size - self.size))
        self.size = size
        return size

    def push_observation(self, discr_list) -> None:
        self.observations = [[self.discrete if i < discr else 0
                              for discr in discr_list]
                             for i in range(self.size)]


class MemoryGraph(StaticGraph):  # Static by default - change StaticGraph
    """Static graph of used/available memory."""  # to Graph to make dynamic.
    log_name = 'mem'

    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)

    def row_num(self) -> int:
        return 1

    def add_observation(self) -> list[float]:
        mems = psu.virtual_memory()
        self.log_observation([mems.used])
        discr = [round(mems.percent / 100 * self.track_size)]
        self.push_observation(discr)
        return [mems.used / 1e+6]

# ########## DEVELOPMENT ##########


class MultiGraph(Graph):
    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)
        self.subs = []
        self.observations = None

    class SubGraph(Graph):
        def __init__(self, multi):
            super().__init__(multi.discrete, multi.track_size)
            multi.subs.append(self)
            self.multi = multi
            self.data = None    # field for multi to write in

        def row_num(self) -> int:
            return self.multi.row_num()

        # add_observation shouldn't call for new data!

    # LOGS AREN'T WORKING ON MULTI LEVEL

    sub_graph = SubGraph

    @property
    def track_size(self) -> int:
        return self.size

    @track_size.setter
    def track_size(self, size: int) -> int:
        for sub in self.subs:
            sub.track_size = size
        self.size = size
        return size

    def all_observations(self) -> list[list[int]]:
        raise Exception(
            "Method 'all_observations()' called from MultiGraph.")


class IOGraph(MultiGraph):
    class IOSubGraph(MultiGraph.SubGraph):
        def __init__(self, multi):
            super().__init__(multi)
            self.discrete_timeline = []  # observations now are float))

        def get_min_max(self) -> tuple[list[float]]:  # -> float:
            return ([min([dat[i] for dat in self.observations])
                     for i in range(len(self.data))],
                    [max([dat[i] for dat in self.observations])
                    for i in range(len(self.data))])

        def all_observations(self) -> list[list[int]]:
            return self.discrete_timeline

        def add_observation(self) -> list[float]:
            if len(self.observations) > 0:
                new_min, new_max = self.get_min_max()
                self.discrete_timeline = [[int(self.discrete * (
                    dat[i] - new_min[i]) / (
                        new_max[i] - new_min[i]
                    )) if new_max[i] != new_min[i] else -1
                                           for i in range(len(dat))]
                                          for dat in self.observations]
            else:
                self.discrete_timeline = []
            return self.data

        def push_observation(self, observ) -> None:  # now floats :)
            super().push_observation(observ)
            self.data = observ

    sub_graph = IOSubGraph

    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)
        self.i_graph = self.__class__.sub_graph(self)
        self.o_graph = self.__class__.sub_graph(self)
        self.names = []

    def row_num(self) -> int:
        return len(self.names)


class NetworkGraph(IOGraph):
    def __init__(self, discrete: int, track_size: int, time_delta: float,
                 show_devices=False):
        super().__init__(discrete, track_size)
        self.show_devices = show_devices
        self.time_delta = time_delta
        self.prev_recv = None
        self.prev_sent = None

    def add_observation_inner(self, nets):
        """Refactored out of add_observation for code reusage."""
        self.names = list(nets.keys())
        total_recv = [device.bytes_recv for device in nets.values()]
        total_sent = [device.bytes_sent for device in nets.values()]
        if self.prev_recv is None:
            self.prev_recv = total_recv
        if self.prev_sent is None:
            self.prev_sent = total_sent
        self.i_graph.push_observation([(x1 - x0) / 1e+3 / self.time_delta
                                       for x1, x0 in zip(
                                               total_recv, self.prev_recv)])
        self.o_graph.push_observation([(x1 - x0) / 1e+3 / self.time_delta
                                       for x1, x0 in zip(
                                               total_sent, self.prev_sent)])
        self.prev_recv = total_recv
        self.prev_sent = total_sent
        return None

    def add_observation(self) -> list[float]:
        nets = psu.net_io_counters(self.show_devices)
        if not self.show_devices:
            nets = {'NET': nets}
        return self.add_observation_inner(nets)


# quite unreliable
class DiskGraph(NetworkGraph):       # inherits NetworkGraph cuz it can))
    def __init__(self, discrete: int, track_size: int, time_delta: float,
                 show_devices=False):
        super().__init__(discrete, track_size, time_delta, show_devices)

    class DiskDataWrapper:      # ahahhahahahh
        def __init__(self, disk_io):
            super().__init__()
            self.bytes_recv = disk_io.read_bytes / 1000
            self.bytes_sent = disk_io.write_bytes / 1000

    def add_observation(self) -> list[float]:
        disks = psu.disk_io_counters(self.show_devices)
        if not self.show_devices:
            disks = {'DISK': disks}
        disks = {key: DiskGraph.DiskDataWrapper(value)
                 for key, value in disks.items()}
        return self.add_observation_inner(disks)


class MultiGraphStatic(MultiGraph):
    def __init__(self, discrete: int, track_size: int):
        super().__init__(discrete, track_size)

    class SubGraphStatic(StaticGraph):
        def __init__(self, multi):
            super().__init__(multi.discrete, multi.track_size)
            multi.subs.append(self)
            self.multi = multi
            self.data = None    # field for multi to write in

        def row_num(self) -> int:
            return self.multi.row_num()

    sub_graph_static = SubGraphStatic


# class NvidiaGraph(MultiGraphStatic):
#     class GpuPercGraph(MultiGraph.SubGraph):
#         row_name = 'GPU'

#         def __init__(self, multi):
#             super().__init__(multi)

#         def add_observation(self) -> list[float]:
#             perc = [nvidia.nvmlDeviceGetClockInfo]

#     def __init__(self, discrete: int, track_size: int):
#         super().__init__(discrete, track_size)
#         nvidia.nvmlInit()
#         # ...

#     def row_num(self) -> int:
#         return nvidia.nvmlDeviceGetCount()

#     def __del__(self):
#         nvidia.nvmlShutdown()


# if __name__ == '__main__':
#     freq_graph = DiskGraph(9, 96)
#     timespan = 60
#     timestep = 1
#     for i in range(int(timespan / timestep)):
#         freq_graph.add_observation()
#         freq_graph.i_graph.add_observation()
#         freq_graph.o_graph.add_observation()
#         # freqs = freq_graph.add_observation()
#         print(f"\nSTEP {i}:\n{freq_graph.i_graph.text_repr()}\n
#  \n{freq_graph.o_graph.text_repr()}")
#         # print(f"\nSTEP {i}:\n{freq_graph.text_repr()}")
#         time.sleep(timestep)

# class FormatList():
#     def __init__(self, str_list: list[str], out_format="@{0:02d}"):
#         super().__init__()
#         self.str_list = str_list
#         self.out_format = out_format

#     def format(self, d: int) -> str:
#         if d < len(self.str_list):
#             return self.str_list[d]
#         else:
#             return self.out_format.format(d)

class NamesIO():
    def __init__(self, name_provider, format_str: str):
        super().__init__()
        self.provider = name_provider
        self.format_str = format_str

    def format(self, d: int) -> str:
        return self.format_str.format(
            self.provider.names[d - 1] if d <= len(
                self.provider.names) else f"@{d}")

# ########## CURSES GRAPHICS ##########
# [to document...]


def init_linear_rgb(r: tuple[int], g: tuple[int], b: tuple[int], bins: int,
                    under: tuple[int]) -> list[tuple[int]]:
    steps = [(c[1] - c[0]) / bins for c in (r, g, b)]
    return [under, *[tuple([int(c[0] + st * i) for st, c in
                            zip(steps, (r, g, b))]) for i in range(bins + 1)]]


def register_rgb(rgb: tuple[int], index: int) -> int:
    crc.init_color(index, rgb[0], rgb[1], rgb[2])
    return index


def register_rgb_list(rgb_list: list[tuple[int]], start_index=0) -> list[int]:
    return [register_rgb(rgb, start_index + i) for i, rgb in
            enumerate(rgb_list)]


def register_pair(fg_indx: int, bg_indx: int, index: int) -> int:
    crc.init_pair(index, fg_indx, bg_indx)
    return index


def register_pairs(rgb_indx_list: list[int], def_bck_indx: int, start_index=0
                   ) -> tuple[list[int]]:
    solid = []
    highlight = []
    for i, rgb_indx in enumerate(rgb_indx_list):
        solid.append(register_pair(def_bck_indx, rgb_indx,
                                   start_index + 2 * i))
        highlight.append(register_pair(rgb_indx, def_bck_indx,
                                       start_index + 2 * i + 1))
    return (solid, highlight)


def writestr(scr, i: int, j: int, text: str, pairnum: int):
    return scr.addstr(i, j, text, crc.color_pair(pairnum) | crc.A_BOLD)


def writech(scr, i: int, j: int, char: str, pairnum: int):
    return scr.addstr(i, j, char, crc.color_pair(pairnum))


def draw_graph(screen, last_obs: list[float], data: list[list[int]],
               graph_name: str, i_start: int, j_start: int,
               format_begin: str, format_end: str, graph_begin: int,
               graph_end: int,
               graph_solid: list[int], graph_highlight: list[int],
               default_pair: int):
    try:
        writestr(screen, i_start, j_start, graph_name, default_pair)
        if len(data) == 0:
            writestr(screen, i_start + 1, j_start, '[COLLAPSED]', default_pair)
            return

        for i, discr_val in enumerate(zip(data[-1], last_obs)):
            discr, val = discr_val
            try:
                writestr(screen, i_start + 1 + i, j_start,
                         format_begin.format(i+1), graph_highlight[discr+1])
            except crc.error:
                pass
            try:
                writestr(screen, i_start + 1 + i, graph_end,
                         format_end.format(val), graph_highlight[discr+1])
                # default_pair)
            except crc.error:
                pass

        for j, obs in enumerate(data):
            for i, discr in enumerate(obs):
                try:
                    writech(screen, i_start + 1 + i, graph_begin + j, ' ',
                            graph_solid[discr+1])
                except crc.error:
                    pass

    except crc.error:
        pass
    except Exception:  # as e:
        writestr(screen, i_start, j_start, '[ERROR]', default_pair)
        # writestr(screen, i_start, j_start, f"[{e}]", default_pair)


graphs = []

# default_pallete = [1, 3, 5, 2, 6, 4]
default_pallete = [5, 1, 3, 2, 6, 4]
default_pallete.reverse()
default_black = 0
default_white = 7

def main(scr, need_log=False, vertical=False, single_pallete=False, default_colors=False):
    crc.curs_set(False)
    crc.start_color()
    # crc.use_default_colors()

    try:
        black = register_rgb((100, 100, 100), 0) if not default_colors else default_black
        white = register_rgb((255, 255, 255), 1) if not default_colors else default_white
        default = register_pair(white, black, 1)  # kinda waste (0 pair...)

        if single_pallete:
            raise Exception('single_pallete')

        freq_bins = 30
        perc_bins = 30
        memr_bins = 1
        netw_bins = 10
        disk_bins = 10

        if default_colors:
            solid, highlight = register_pairs(default_pallete, black, 2)

            freq_solid = perc_solid = memr_solid = netw_solid = solid
            disk_solid = solid

            freq_highlight = perc_highlight = highlight
            memr_highlight = netw_highlight = highlight
            disk_highlight = highlight

            freq_bins = perc_bins = memr_bins = netw_bins = len(default_pallete) - 2
            disk_bins = len(default_pallete) - 2
        else:
            freq_colors = register_rgb_list(init_linear_rgb(
                (255, 255), (225, 120), (225, 0), freq_bins, (255, 255, 255)), 2)
            perc_colors = register_rgb_list(init_linear_rgb(
                (225, 0), (255, 255), (255, 255), perc_bins, (255, 255, 255)),
                                            freq_colors[-1]+1)
            memr_colors = register_rgb_list(init_linear_rgb(
                (255, 125), (255, 140), (225, 180), memr_bins, (255, 255, 255)),
                                            perc_colors[-1]+1)  # (225, 185, 110)
            netw_colors = register_rgb_list(init_linear_rgb(
                (255, 93), (225, 187), (255, 151), netw_bins, (255, 255, 255)),
                                            memr_colors[-1]+1)
            disk_colors = register_rgb_list(init_linear_rgb(  # recolor
                (255, 187), (255, 151), (195, 63), disk_bins, (255, 255, 255)),
                                            netw_colors[-1]+1)
            # (255, 125), (255, 50), (225, 205)

            freq_solid, freq_highlight = register_pairs(freq_colors, black, 2)
            perc_solid, perc_highlight = register_pairs(perc_colors, black,
                                                        freq_highlight[-1]+1)
            memr_solid, memr_highlight = register_pairs(memr_colors, black,
                                                        perc_highlight[-1]+1)
            netw_solid, netw_highlight = register_pairs(netw_colors, black,
                                                        memr_highlight[-1]+1)
            disk_solid, disk_highlight = register_pairs(disk_colors, black,
                                                        netw_highlight[-1]+1)

    except Exception:           # addressing tty problem
        min_colors = 3
        max_colors = 100
        avail_colors = min(crc.COLORS - 3, max_colors)
        # avail_colors = 8 - 3
        if avail_colors < min_colors:
            raise Exception(f"Scipt requires at least {min_colors + 2}" +
                            + f" to work. Only {crc.COLORS} are available.")
        # if default_colors:
        #     avail_colors = min(avail_colors. len(pallete) - 1)
        #     pallete = default_pallete[:avail_colors + 1]
        # else:
        pallete = [white, *register_rgb_list(init_linear_rgb(
            # (225, 0), (255, 255), (255, 255), avail_colors, None)[1:], 2)]
            (255, 125), (255, 140), (225, 180), avail_colors, None)[1:], 2)]
        solid, highlight = register_pairs(pallete, black, 2)
        # raise Exception(f"plt = {pallete}\nsld = {solid}\nhlt = {highlight}")

        freq_solid = perc_solid = memr_solid = netw_solid = solid
        disk_solid = solid

        freq_highlight = perc_highlight = highlight
        memr_highlight = netw_highlight = highlight  # highlight bug in tty
        disk_highlight = highlight

        freq_bins = perc_bins = memr_bins = netw_bins = avail_colors
        disk_bins = avail_colors

    time_delay = 0.25

    # Frequency graph:
    freq_graph = FreqGraph(freq_bins, 0)
    freq_name = ' FREQUENCY:'   # 11 chars
    freq_fbegin = "CPU{0:02d}:"  # 6 chars
    freq_fend = "{0:4.0f} MHz"    # 8 chars

    # Percentage graph:
    perc_graph = PercGraph(perc_bins, 0, True)
    perc_name = ' PERCENTAGE:'  # 12 chars
    perc_fbegin = freq_fbegin
    # perc_fend = "  {0:2.0f}%"      # 3 chars
    perc_fend = "{0:2.0f}%"

    # Memory graph
    memr_graph = MemoryGraph(memr_bins, 0)
    memr_name = ' MEMORY:'   # 8 chars
    memr_fbegin = "  RAM:"  # 6 chars
    memr_fend = "{0:5.0f} MB"    # 8 chars

    # Network graph
    netw_graph = NetworkGraph(netw_bins, 0, time_delay)
    netw_name_i = ' NETWORK RECEIVING:'   # 19 chars
    netw_name_o = ' NETWORK SENDING:'   # 17 chars
    netw_fbegin = NamesIO(netw_graph, "  {0:.3s}:")  # 6 chars
    netw_fend = "{0:5.0f} KB/s"    # 8 chars

    # Disk graph
    disk_graph = DiskGraph(disk_bins, 0, time_delay)
    disk_name_i = ' DISK READ:'   # 11 chars
    disk_name_o = ' DISK WRITE:'   # 12 chars
    disk_fbegin = NamesIO(disk_graph, " {0:.4s}:")  # 6 chars
    disk_fend = "{0:5.0f} MB/s"    # 8 chars

    if need_log:
        graphs.extend([freq_graph, perc_graph, memr_graph])  # REPAIR
        for graph in graphs:
            graph.start_log()

    scr.bkgd(' ', crc.color_pair(default))
    scr.clear()

    while True:
        scr.erase()
        crc.update_lines_cols()
        if vertical:            # sizes
            available_size = max(0, crc.COLS - 19)
        else:
            available_size = max(0, crc.COLS - 38) // 2
        freq_graph.track_size = available_size
        perc_graph.track_size = available_size
        memr_graph.track_size = available_size if vertical else max(
            0, 22 + available_size * 2)  # crc.COLS - 17)
        # memr graph doesn't lock properly when freq and perc are collapsed[!]
        netw_graph.track_size = max(0, crc.COLS - 38) // 2  # problem with -v
        disk_graph.track_size = netw_graph.track_size

        # Observations:
        freq_obs = freq_graph.add_observation()
        perc_obs = perc_graph.add_observation()
        memr_obs = memr_graph.add_observation()
        netw_graph.add_observation()
        netw_i_obs = netw_graph.i_graph.add_observation()
        netw_o_obs = netw_graph.o_graph.add_observation()
        disk_graph.add_observation()
        disk_i_obs = disk_graph.i_graph.add_observation()
        disk_o_obs = disk_graph.o_graph.add_observation()

        draw_graph(scr, freq_obs, freq_graph.all_observations(), freq_name, 0,
                   0, freq_fbegin, freq_fend,
                   7, 11 + available_size, freq_solid, freq_highlight, default)
        if vertical:
            draw_graph(scr, perc_obs, perc_graph.all_observations(), perc_name,
                       freq_graph.row_num() + 2, 0, perc_fbegin, perc_fend,
                       7, 12 + available_size, perc_solid, perc_highlight,
                       default)
        else:
            draw_graph(scr, perc_obs, perc_graph.all_observations(), perc_name,
                       0, 22 + available_size, perc_fbegin, perc_fend,
                       29 + available_size, 32 + available_size * 2,
                       perc_solid, perc_highlight, default)
        mem_i = (perc_graph.row_num() + freq_graph.row_num(
                   ) + 4) if vertical else (max(perc_graph.row_num(
                   ), freq_graph.row_num()) + 2)
        draw_graph(scr, memr_obs, memr_graph.all_observations(), memr_name,
                   mem_i, 0, memr_fbegin, memr_fend,
                   7, (10 if vertical else 8) + memr_graph.track_size,
                   memr_solid, memr_highlight, default)

        draw_graph(scr, netw_i_obs, netw_graph.i_graph.all_observations(),
                   netw_name_i, mem_i + 3, 0, netw_fbegin, netw_fend,
                   7, 9 + netw_graph.track_size, netw_solid, netw_highlight,
                   default)
        draw_graph(scr, netw_o_obs, netw_graph.o_graph.all_observations(),
                   netw_name_o, mem_i + 3, 22 + netw_graph.track_size,
                   netw_fbegin, netw_fend, 29 + netw_graph.track_size,
                   28 + netw_graph.track_size * 2, netw_solid, netw_highlight,
                   default)

        draw_graph(scr, disk_i_obs, disk_graph.i_graph.all_observations(),
                   disk_name_i, mem_i + 6, 0, disk_fbegin, disk_fend,
                   7, 9 + disk_graph.track_size, disk_solid, disk_highlight,
                   default)
        draw_graph(scr, disk_o_obs, disk_graph.o_graph.all_observations(),
                   disk_name_o, mem_i + 6, 22 + disk_graph.track_size,
                   disk_fbegin, disk_fend, 29 + disk_graph.track_size,
                   28 + disk_graph.track_size * 2, disk_solid, disk_highlight,
                   default)

        scr.refresh()
        time.sleep(time_delay)


if __name__ == '__main__':
    log_file = None
    try:
        parser = arg.ArgumentParser(prog='CPU Graphs',
                                    description="""Lightweight python script
                                    for CPU frequences monitoring. [in dev]""")
        # parser.add_argument('-l', '--log', action='store', type=str,
        #                     const=None, default=0, nargs='?', help='<TO DO>')
        parser.add_argument('-v', '--vertical', action='store_true',
                            help='<TO DO>')
        parser.add_argument('-s', '--single', action='store_true',
                            help='<TO DO>')
        parser.add_argument('-d', '--default', action='store_true',
                            help='<TO DO>')
        parser.add_argument('-b', '--black', action='store_true',
                            help='<TO DO>')
        args = parser.parse_args()
        # need_log = args.log != 0
        # log_file = args.log
        need_log = False        # disabled for now
        log_file = 0

        if args.black:
            default_black = 7
            default_white = 0
            default_pallete = [1, 3, 5, 2, 6, 4]
            default_pallete.reverse()
        crc.wrapper(main, need_log=need_log, vertical=args.vertical,
                    single_pallete=args.single,
                    default_colors=args.default or args.black)
        crc.curs_set(True)
    except KeyboardInterrupt:
        if graphs:
            if log_file is None:
                log_file = time.strftime(
                    'cpu_graphs_log_%H:%M:%S_%d-%m-%Y.csv',
                    time.localtime())
            with open(log_file, 'wt') as f:
                f.write(compile_log(graphs))
            print('Log dumped.')
    except Exception as e:
        print(e)  # bad but whatever
