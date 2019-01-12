import os
import sys
from glob import glob
import pickle
import argparse
import time

import numpy as np
import h5py
import scipy.linalg as la

from collections import defaultdict

from pychfpga import NameSpace, load_yaml_config
from calibration import utils

import log

import caput.time as ctime
import time
import skyfield.api

from ch_util import tools, ephemeris, andata
from ch_util.fluxcat import FluxCatalog

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':redundantcal'))

LOG_FILE = os.environ.get('N2CAL_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'redundantcal.log'))

DEFAULT_LOGGING = {
    'formatters': {
         'std': {
             'format': "%(asctime)s %(levelname)s %(name)s: %(message)s",
             'datefmt': "%m/%d %H:%M:%S"},
          },
    'handlers': {
        'stderr': {'class': 'logging.StreamHandler', 'formatter': 'std', 'level': 'DEBUG'}
        },
    'loggers': {
        '': {'handlers': ['stderr'], 'level': 'INFO'}  # root logger

        }
    }

###################################################
# ancillary routines
###################################################

def coupled_indices(cutoff=0, cross_pol=True, N=2048, cyl_size=256):

    Ncyl = N / cyl_size

    row = []
    col = []

    for cc in range(Ncyl):

        acyl = cc*cyl_size
        bcyl = (cc+1)*cyl_size

        for ii in range(acyl, bcyl):

            aa = max(acyl, ii - cutoff)
            bb = min(bcyl, ii + cutoff + 1)

            for jj in range(aa, bb):

                row.append(ii)
                col.append(jj)


    if cross_pol:

        for cc in range(Ncyl):

            acyl = cc*cyl_size
            bcyl = (cc+1)*cyl_size

            offset = cyl_size - 2*cyl_size*(cc %  2)
            across = offset + acyl
            bcross = offset + bcyl

            for ii in range(acyl, bcyl):

                aa = max(across, offset + ii - cutoff)
                bb = min(bcross, offset + ii + cutoff + 1)

                for jj in range(aa, bb):

                    row.append(ii)
                    col.append(jj)

    return (np.array(row), np.array(col))


def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi

def sun_coord(unix_time, deg=True):

    date = ephemeris.ensure_unix(np.atleast_1d(unix_time))
    skyfield_time = ephemeris.unix_to_skyfield_time(date)
    ntime = date.size

    coord = np.zeros((ntime, 4), dtype=np.float32)

    planets = skyfield.api.load('de421.bsp')
    sun = planets['sun']

    observer = ephemeris._get_chime().skyfield_obs()

    apparent = observer.at(skyfield_time).observe(sun).apparent()
    radec = apparent.radec(epoch=skyfield_time)

    coord[:, 0] = radec[0].radians
    coord[:, 1] = radec[1].radians

    altaz = apparent.altaz()
    coord[:, 2] = altaz[0].radians
    coord[:, 3] = altaz[1].radians

    # Correct RA from equinox to CIRS coords using
    # the equation of the origins
    era = np.radians(ctime.unix_to_era(date))
    gast = 2 * np.pi * skyfield_time.gast / 24.0
    coord[:, 0] = coord[:, 0] + (era - gast)

    # Convert to hour angle
    coord[:, 0] = _correct_phase_wrap(coord[:, 0] - np.radians(ephemeris.lsa(date)))

    if deg:
        coord = np.degrees(coord)

    return coord


def generate_mapping(baseline):

    nbaseline = baseline.shape[0]

    db = np.nanmin(np.where(baseline == 0.0, np.nan, np.abs(baseline)), axis=0)

    bint = baseline / db[np.newaxis, :]

    is_chime = np.flatnonzero(np.any(np.isfinite(bint), axis=1))

    bid = [tuple(np.round(bint[ii, :]).astype(np.int)) for ii in is_chime]
    ubid = sorted(list(set(bid)))[::-1]

    bmap = np.full(nbaseline, np.nan, dtype=np.int)
    bmap[is_chime] = np.array([ubid.index(bb) for bb in bid])

    ubaseline = np.array([np.array(uu) * db for uu in ubid])

    return bmap, ubaseline


###################################################
# main routine
###################################################

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Setup logging
    log.setup_logging(logging_params)
    mlog = log.get_logger(__name__)

    # Set config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Set niceness
    current_niceness = os.nice(0)
    os.nice(config.niceness - current_niceness)
    mlog.info('Changing process niceness from %d to %d.  Confirm:  %d' %
                  (current_niceness, config.niceness, os.nice(0)))

    # Find acquisition files
    acq_files = sorted(glob(os.path.join(config.data_dir, config.acq, "*.h5")))
    nfiles = len(acq_files)

    # Determine time range of each file
    findex = []
    tindex = []
    for ii, filename in enumerate(acq_files):
        subdata = andata.CorrData.from_acq_h5(filename, datasets=())

        findex += [ii] * subdata.ntime
        tindex += range(subdata.ntime)

    findex = np.array(findex)
    tindex = np.array(tindex)

    # Determine transits within these files
    transits = []

    data = andata.CorrData.from_acq_h5(acq_files, datasets=())

    solar_rise = ephemeris.solar_rising(data.time[0] - 24.0 * 3600.0, end_time=data.time[-1])

    for rr in solar_rise:

        ss = ephemeris.solar_setting(rr)[0]

        solar_flag = np.flatnonzero((data.time >= rr) & (data.time <= ss))

        if solar_flag.size > 0:

            solar_flag = solar_flag[::config.downsample]

            tval = data.time[solar_flag]

            this_findex = findex[solar_flag]
            this_tindex = tindex[solar_flag]

            file_list, tindices = [], []

            for ii in range(nfiles):

                this_file = np.flatnonzero(this_findex == ii)

                if this_file.size > 0:

                    file_list.append(acq_files[ii])
                    tindices.append(this_tindex[this_file])

            date = ephemeris.unix_to_datetime(rr).strftime('%Y%m%dT%H%M%SZ')
            transits.append((date, tval, file_list, tindices))

    # Create file prefix and suffix
    prefix = []

    prefix.append("redundant_calibration")

    if config.output_prefix is not None:
        prefix.append(config.output_prefix)

    prefix = '_'.join(prefix)


    suffix = []

    if config.include_auto:
        suffix.append("wauto")
    else:
        suffix.append("noauto")

    if config.include_intracyl:
        suffix.append("wintra")
    else:
        suffix.append("nointra")

    if config.fix_degen:
        suffix.append("fixed_degen")
    else:
        suffix.append("degen")


    suffix = '_'.join(suffix)

    # Loop over solar transits
    for date, timestamps, files, time_indices in transits:

        nfiles = len(files)

        mlog.info("%s (%d files) " % (date, nfiles))

        output_file = os.path.join(config.output_dir, "%s_SUN_%s_%s.h5"  % (prefix, date, suffix))

        mlog.info("Saving to:  %s" % output_file)

        # Get info about this set of files
        data = andata.CorrData.from_acq_h5(files, datasets=['flags/inputs'], apply_gain=False, renormalize=False)

        coord = sun_coord(timestamps, deg=True)

        fstart = config.freq_start if config.freq_start is not None else 0
        fstop = config.freq_stop if config.freq_stop is not None else data.freq.size
        freq_index = range(fstart, fstop)

        freq =  data.freq[freq_index]

        ntime = timestamps.size
        nfreq = freq.size

        # Determind bad inputs
        if config.bad_input_file is None or not os.path.isfile(config.bad_input_file):
            bad_input = np.flatnonzero(~np.all(data.flags['inputs'][:], axis=-1))
        else:
            with open(config.bad_input_file, 'r') as handler:
                bad_input = pickle.load(handler)

        mlog.info("%d inputs flagged as bad." % bad_input.size)

        nant = data.ninput

        # Determine polarization product maps
        dbinputs = tools.get_correlator_inputs(ephemeris.unix_to_datetime(timestamps[0]), correlator='chime')

        dbinputs = tools.reorder_correlator_inputs(data.input, dbinputs)

        feedpos = tools.get_feed_positions(dbinputs)

        prod = defaultdict(list)
        dist = defaultdict(list)

        for pp, this_prod in enumerate(data.prod):

            aa, bb = this_prod
            inp_aa = dbinputs[aa]
            inp_bb = dbinputs[bb]

            if (aa in bad_input) or (bb in bad_input):
                continue

            if not tools.is_chime(inp_aa) or not tools.is_chime(inp_bb):
                continue

            if not config.include_intracyl and (inp_aa.cyl == inp_bb.cyl):
                continue

            if not config.include_auto and (aa == bb):
                continue

            this_dist = list(feedpos[aa, :] - feedpos[bb, :])

            if tools.is_array_x(inp_aa) and tools.is_array_x(inp_bb):
                key = 'XX'

            elif tools.is_array_y(inp_aa) and tools.is_array_y(inp_bb):
                key = 'YY'

            elif not config.include_crosspol:
                continue

            elif tools.is_array_x(inp_aa) and tools.is_array_y(inp_bb):
                key = 'XY'

            elif tools.is_array_y(inp_aa) and tools.is_array_x(inp_bb):
                key = 'YX'

            else:
                raise RuntimeError("CHIME feeds not polarized.")

            prod[key].append(pp)
            dist[key].append(this_dist)

        polstr = sorted(prod.keys())
        polcnt = 0
        pol_sky_id = []
        bmap = {}
        for key in polstr:
            prod[key] = np.array(prod[key])
            dist[key] = np.array(dist[key])

            p_bmap, p_ubaseline = generate_mapping(dist[key])
            nubase = p_ubaseline.shape[0]

            bmap[key] = p_bmap + polcnt

            if polcnt > 0:

                ubaseline = np.concatenate((ubaseline, p_ubaseline), axis=0)
                pol_sky_id += [key] * nubase

            else:

                ubaseline = p_ubaseline.copy()
                pol_sky_id = [key] * nubase

            polcnt += nubase
            mlog.info("%d unique baselines" % polcnt)

        nsky = bmap.size

        # Create arrays to hold the results
        ores = {}
        ores['freq'] = freq
        ores['input'] = data.input
        ores['time'] = timestamps
        ores['coord'] = coord
        ores['pol'] = np.array(pol_sky_id)
        ores['baseline'] = ubaseline

        # Create array to hold gain results
        ores['gain'] = np.zeros((nfreq, nant, ntime), dtype=np.complex)
        ores['sky'] = np.zeros((nfreq, nsky, ntime), dtype=np.complex)
        ores['err'] = np.zeros((nfreq, nant + nsky, ntime, 2), dtype=np.float)

        # Loop over polarisations
        for key in polstr:

            reverse_map = bmap[key]
            p_prod = prod[key]

            isort = np.argsort(reverse_map)

            p_prod = p_prod[isort]

            p_ant1 = data.prod['input_a'][p_prod]
            p_ant2 = data.prod['input_b'][p_prod]
            p_vismap = reverse_map[isort]

            # Find the redundant groups
            tmp = np.where(np.diff(p_vismap) != 0)[0]
            edges = np.zeros(2+tmp.size, dtype='int')
            edges[0] = 0
            edges[1:-1] = tmp+1
            edges[-1] = p_vismap.size

            kept_base = np.unique(p_vismap)

            # Determine the unique antennas
            kept_ants = np.unique(np.concatenate([p_ant1, p_ant2]))
            antmap = np.zeros(kept_ants.max()+1, dtype='int')-1

            p_nant = kept_ants.size
            for i in range(p_nant):
                antmap[kept_ants[i]]=i

            p_ant1_use = antmap[p_ant1].copy()
            p_ant2_use = antmap[p_ant2].copy()

            # Create matrix
            p_nvis = p_prod.size
            nred = edges.size - 1

            npar = p_nant + nred

            A = np.zeros((p_nvis, npar), dtype=np.float32)
            B = np.zeros((p_nvis, npar), dtype=np.float32)

            for kk in range(p_nant):

                flag_ant1 = p_ant1_use == kk
                if np.any(flag_ant1):
                    A[flag_ant1, kk] = 1.0
                    B[flag_ant1, kk] = 1.0

                flag_ant2 = p_ant2_use == kk
                if np.any(flag_ant2):
                    A[flag_ant2, kk] = 1.0
                    B[flag_ant2, kk] = -1.0

            for ee in range(nred):

                A[edges[ee]:edges[ee+1], p_nant + ee] = 1.0

                B[edges[ee]:edges[ee+1], p_nant + ee] = 1.0

            # Add equations to break degeneracy
            if config.fix_degen:
                A = np.concatenate((A, np.zeros((1, npar), dtype=np.float32)))
                A[-1, 0:p_nant] = 1.0

                B = np.concatenate((B, np.zeros((3, npar), dtype=np.float32)))
                B[-3, 0:p_nant] = 1.0
                B[-2, 0:p_nant] = feedpos[kept_ants, 0]
                B[-1, 0:p_nant] = feedpos[kept_ants, 1]

            # Loop over frequencies
            for ff, find in enumerate(freq_index):

                mlog.info("Freq %d of %d.  %0.2f MHz." % (ff+1, nfreq, freq[ff]))

                cnt = 0

                # Loop over files
                for ii, (filename, tind) in enumerate(zip(files, time_indices)):

                    ntind = len(tind)
                    mlog.info("Processing file %s (%d time samples)" % (filename, ntind))

                    # Compute noise weight
                    with h5py.File(filename, 'r') as hf:
                        wnoise = np.median(hf['flags/vis_weight'][find, :, :], axis=-1)

                    # Loop over times
                    for tt in tind:

                        t0 = time.time()

                        mlog.info("Time %d of %d.  %d index of current file." % (cnt+1, ntime, tt))

                        # Load visibilities
                        with h5py.File(filename, 'r') as hf:

                            snap = hf['vis'][find, :, tt]
                            wsnap = wnoise * (hf['flags/vis_weight'][find, :, tt] > 0.0).astype(np.float32)

                        # Extract relevant products for this polarization
                        snap = snap[p_prod]
                        wsnap = wsnap[p_prod]

                        # Turn into amplitude and phase
                        amp = np.log(np.abs(snap))
                        phi = np.angle(snap)

                        # Deal with phase wrapping
                        for aa, bb in zip(edges[:-1], edges[1:]):
                            dphi = phi[aa:bb] - np.sort(phi[aa:bb])[int((bb - aa) / 2)]
                            phi[aa:bb] += (2.0*np.pi * (dphi < -np.pi) - 2.0*np.pi * (dphi > np.pi))

                        # Add elements to fix degeneracy
                        if config.fix_degen:
                            amp = np.concatenate((amp, np.zeros(1)))
                            phi = np.concatenate((phi, np.zeros(3)))

                        # Determine noise matrix
                        inv_diagC = wsnap * np.abs(snap)**2 * 2.0

                        if config.fix_degen:
                            inv_diagC = np.concatenate((inv_diagC, np.ones(1)))

                        # Amplitude estimate and covariance
                        amp_param_cov = np.linalg.inv(np.dot(A.T, inv_diagC[:, np.newaxis] * A))
                        amp_param = np.dot(amp_param_cov, np.dot(A.T, inv_diagC * amp))

                        # Phase estimate and covariance
                        if config.fix_degen:
                            inv_diagC = np.concatenate((inv_diagC, np.ones(2)))

                        phi_param_cov = np.linalg.inv(np.dot(B.T, inv_diagC[:, np.newaxis] * B))
                        phi_param = np.dot(phi_param_cov, np.dot(B.T, inv_diagC * phi))

                        # Save to large array
                        ores['gain'][ff, kept_ants, cnt] = np.exp(amp_param[0:p_nant] + 1.0J * phi_param[0:p_nant])

                        ores['sky'][ff, kept_base, cnt] = np.exp(amp_param[p_nant:] + 1.0J * phi_param[p_nant:])

                        ores['err'][ff, kept_ants, cnt, 0] = np.diag(amp_param_cov[0:p_nant, 0:p_nant])
                        ores['err'][ff, nant + kept_base, cnt, 0] = np.diag(amp_param_cov[p_nant:, p_nant:])

                        ores['err'][ff, kept_ants, cnt, 1] = np.diag(phi_param_cov[0:p_nant, 0:p_nant])
                        ores['err'][ff, nant + kept_base, cnt, 1] = np.diag(phi_param_cov[p_nant:, p_nant:])

                        # Increment time counter
                        cnt += 1

                        # Print time elapsed
                        mlog.info("Took %0.1f seconds." % (time.time() - t0,))


        # Save to pickle file
        with h5py.File(output_file, 'w') as handler:

            handler.attrs['date'] = date

            for key, val in ores.iteritems():
                handler.create_dataset(key, data=val)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',   help='Name of configuration file.',
                                      type=str, default=None)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    try:
        os.makedirs(os.path.dirname(args.log))
    except OSError:
        if not os.path.isdir(os.path.dirname(args.log)):
            raise

    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                             'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(config_file=args.config, logging_params=logging_params)
