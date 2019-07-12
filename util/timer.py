from math import floor
from time import perf_counter

from util.print_color import blue


class Timer:

    def start(self, msg):
        self.start_time = perf_counter()
        blue(f'{msg}: ', end='')

    def finish(self):
        time = perf_counter() - self.start_time
        min = time // 60
        sec = floor(time % 60)
        ms = str(time % 1)[2:5]

        blue('%d:%02d.%s' % (min, sec, ms))
