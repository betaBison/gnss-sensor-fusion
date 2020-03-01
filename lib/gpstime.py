"""
A Python implementation of GPS related time conversions.

Copyright 2002 by Bud P. Bruegger, Sistema, Italy
mailto:bud@sistema.it
http://www.sistema.it

  8  Modifications for GPS seconds by Duncan Brown
  9
 10  PyUTCFromGpsSeconds added by Ben Johnson
 11
 12  This program is free software; you can redistribute it and/or modify it under
 13  the terms of the GNU Lesser General Public License as published by the Free
 14  Software Foundation; either version 2 of the License, or (at your option) any
 15  later version.
 16
 17  This program is distributed in the hope that it will be useful, but WITHOUT ANY
 18  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 19  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 20  details.
 21
 22  You should have received a copy of the GNU Lesser General Public License along
 23  with this program; if not, write to the Free Software Foundation, Inc., 59
 24  Temple Place, Suite 330, Boston, MA  02111-1307  USA
 25
 26  GPS Time Utility functions
 27
 28  This file contains a Python implementation of GPS related time conversions.
 29
 30  The two main functions convert between UTC and GPS time (GPS-week, time of
 31  week in seconds, GPS-day, time of day in seconds).  The other functions are
 32  convenience wrappers around these base functions.
 33
 34  A good reference for GPS time issues is:
 35  http://www.oc.nps.navy.mil/~jclynch/timsys.html
 36
 37  Note that python time types are represented in seconds since (a platform
 38  dependent Python) Epoch.  This makes implementation quite straight forward
 39  as compared to some algorigthms found in the literature and on the web.
 40
 """

import time, math

secsInWeek = 604800
secsInDay = 86400
gpsEpoch = (1980, 1, 6, 0, 0, 0)  # (year, month, day, hh, mm, ss)


def gpsFromUTC(year, month, day, hour, min, sec, leapSecs=18):
    """converts UTC to: gpsWeek, secsOfWeek, gpsDay, secsOfDay

    a good reference is:  http://www.oc.nps.navy.mil/~jclynch/timsys.html

    This is based on the following facts (see reference above):

    GPS time is basically measured in (atomic) seconds since
    January 6, 1980, 00:00:00.0  (the GPS Epoch)

    The GPS week starts on Saturday midnight (Sunday morning), and runs
    for 604800 seconds.

    Currently, GPS time is 13 seconds ahead of UTC (see above reference).
    While GPS SVs transmit this difference and the date when another leap
    second takes effect, the use of leap seconds cannot be predicted.  This
    routine is precise until the next leap second is introduced and has to be

    updated after that.
    SOW = Seconds of Week
    SOD = Seconds of Day

    Note:  Python represents time in integer seconds, fractions are lost!!!
    """
    secFract = sec % 1
    epochTuple = gpsEpoch + (-1, -1, 0)
    t0 = time.mktime(epochTuple)
    t = time.mktime((year, month, day, hour, min, sec, -1, -1, 0))
    # Note: time.mktime strictly works in localtime and to yield UTC, it should be
    #       corrected with time.timezone
    #       However, since we use the difference, this correction is unnecessary.
    # Warning:  trouble if daylight savings flag is set to -1 or 1 !!!
    t = t + leapSecs
    tdiff = t - t0
    gpsSOW = (tdiff % secsInWeek)  + secFract
    gpsWeek = int(math.floor(tdiff/secsInWeek))
    gpsDay = int(math.floor(gpsSOW/secsInDay))
    gpsSOD = (gpsSOW % secsInDay)
    return (gpsWeek, gpsSOW, gpsDay, gpsSOD)
