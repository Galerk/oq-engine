# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2020 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
import io
import copy
import psutil
import pprint
import logging
import operator
from datetime import datetime
import numpy
try:
    from PIL import Image
except ImportError:
    Image = None
from openquake.baselib import parallel, hdf5
from openquake.baselib.python3compat import encode
from openquake.baselib.general import (
    AccumDict, DictArray, block_splitter, groupby, humansize,
    get_nbytes_msg)
from openquake.hazardlib.contexts import ContextMaker, get_effect
from openquake.hazardlib.calc.hazard_curve import classical as hazclassical
from openquake.hazardlib.probability_map import ProbabilityMap
from openquake.commonlib import calc, util, logs, readinput
from openquake.calculators import getters
from openquake.calculators import base

U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32
F64 = numpy.float64
TWO32 = 2 ** 32
get_weight = operator.attrgetter('weight')
grp_extreme_dt = numpy.dtype([('et_id', U16), ('grp_trt', hdf5.vstr),
                             ('extreme_poe', F32)])


def get_source_id(src):  # used in submit_tasks
    return src.source_id.split(':')[0]


def get_extreme_poe(array, imtls):
    """
    :param array: array of shape (L, G) with L=num_levels, G=num_gsims
    :param imtls: DictArray imt -> levels
    :returns:
        the maximum PoE corresponding to the maximum level for IMTs and GSIMs
    """
    return max(array[imtls(imt).stop - 1].max() for imt in imtls)


#  ########################### task functions ############################ #

def classical(srcs, rlzs_by_gsim, params, monitor):
    """
    Read the SourceFilter and call the classical calculator in hazardlib
    """
    srcfilter = monitor.read('srcfilter')
    return hazclassical(srcs, srcfilter, rlzs_by_gsim, params, monitor)


def start_classical(sources, rlzs_by_gsim, params, monitor):
    """
    Compute the PoEs from filtered sources.
    """
    blocks = list(block_splitter(sources, params['max_weight']/2, get_weight))
    for block in blocks[:-1]:
        yield classical, block, rlzs_by_gsim, params
    yield classical(blocks[-1], rlzs_by_gsim, params, monitor)


def store_ctxs(dstore, rupdata, grp_id):
    """
    Store contexts with the same magnitude in the datastore
    """
    nr = len(rupdata['mag'])
    rupdata['nsites'] = numpy.array([len(s) for s in rupdata['sids_']])
    rupdata['grp_id'] = numpy.repeat(grp_id, nr)
    nans = numpy.repeat(numpy.nan, nr)
    for par in dstore['rup']:
        n = 'rup/' + par
        if par.endswith('_'):
            if par in rupdata:
                dstore.hdf5.save_vlen(n, rupdata[par])
            else:  # add nr empty rows
                dstore[n].resize((len(dstore[n]) + nr,))
        else:
            hdf5.extend(dstore[n], rupdata.get(par, nans))


@base.calculators.add('classical', 'preclassical', 'ucerf_classical')
class ClassicalCalculator(base.HazardCalculator):
    """
    Classical PSHA calculator
    """
    core_task = start_classical
    accept_precalc = ['classical']

    def agg_dicts(self, acc, dic):
        """
        Aggregate dictionaries of hazard curves by updating the accumulator.

        :param acc: accumulator dictionary
        :param dic: dict with keys pmap, calc_times, rup_data
        """
        # NB: dic should be a dictionary, but when the calculation dies
        # for an OOM it can become None, thus giving a very confusing error
        if dic is None:
            raise MemoryError('You ran out of memory!')
        pmap = dic['pmap']
        extra = dic['extra']
        ctimes = dic['calc_times']  # srcid -> eff_rups, eff_sites, dt
        self.calc_times += ctimes
        srcids = set()
        eff_rups = 0
        eff_sites = 0
        for srcid, rec in ctimes.items():
            srcids.add(srcid)
            eff_rups += rec[0]
            if rec[0]:
                eff_sites += rec[1] / rec[0]
        self.by_task[extra['task_no']] = (
            eff_rups, eff_sites, sorted(srcids))
        acc.eff_ruptures[extra.pop('trt')] += eff_rups
        if not pmap:
            return acc
        grp_id = extra['grp_id']
        if self.oqparam.disagg_by_src:
            # store the poes for the given source
            pmap.grp_id = grp_id
            acc[extra['source_id'].split(':')[0]] = pmap
        self.maxradius = max(self.maxradius, extra.pop('maxradius'))
        with self.monitor('aggregate curves'):
            self.totrups += extra['totrups']
            if pmap and grp_id in acc:
                acc[grp_id] |= pmap
            else:
                acc[grp_id] = copy.copy(pmap)
            # store rup_data if there are few sites
            if self.few_sites:
                store_ctxs(self.datastore, dic['rup_data'], grp_id)

        return acc

    def acc0(self):
        """
        Initial accumulator, a dict et_id -> ProbabilityMap(L, G)
        """
        zd = AccumDict()
        params = {'grp_id', 'occurrence_rate', 'clon_', 'clat_', 'rrup_',
                  'nsites', 'probs_occur_', 'sids_', 'src_id'}
        gsims_by_trt = self.full_lt.get_gsims_by_trt()
        for trt, gsims in gsims_by_trt.items():
            cm = ContextMaker(trt, gsims)
            params.update(cm.REQUIRES_RUPTURE_PARAMETERS)
            for dparam in cm.REQUIRES_DISTANCES:
                params.add(dparam + '_')
        zd.eff_ruptures = AccumDict(accum=0)  # trt -> eff_ruptures
        mags = set()
        for trt, dset in self.datastore['source_mags'].items():
            mags.update(dset[:])
        mags = sorted(mags)
        if self.few_sites:
            for param in params:
                if param == 'sids_':
                    dt = hdf5.vuint16
                elif param == 'probs_occur_':
                    dt = hdf5.vfloat64
                elif param.endswith('_'):
                    dt = hdf5.vfloat32
                elif param == 'src_id':
                    dt = U32
                elif param in {'nsites', 'grp_id'}:
                    dt = U16
                else:
                    dt = F32
                self.datastore.create_dset('rup/' + param, dt, (None,),
                                           compression='gzip')
            dset = self.datastore.getitem('rup')
            dset.attrs['__pdcolumns__'] = ' '.join(params)
        self.by_task = {}  # task_no => src_ids
        self.totrups = 0  # total number of ruptures before collapsing
        self.maxradius = 0
        self.Ns = len(self.csm.source_info)
        if self.oqparam.disagg_by_src:
            sources = self.get_source_ids()
            self.datastore.create_dset(
                'disagg_by_src', F32,
                (self.N, self.R, self.M, self.L1, self.Ns))
            self.datastore.set_shape_attrs(
                'disagg_by_src', site_id=self.N, rlz_id=self.R,
                imt=list(self.oqparam.imtls), lvl=self.L1, src_id=sources)
        return zd

    def get_source_ids(self):
        """
        :returns: the unique source IDs contained in the composite model
        """
        oq = self.oqparam
        self.M = len(oq.imtls)
        self.L1 = len(oq.imtls.array) // self.M
        sources = encode([src_id for src_id in self.csm.source_info])
        size, msg = get_nbytes_msg(
            dict(N=self.N, R=self.R, M=self.M, L1=self.L1, Ns=self.Ns))
        ps = 'pointSource' in self.full_lt.source_model_lt.source_types
        if size > TWO32 and not ps:
            raise RuntimeError('The matrix disagg_by_src is too large: %s'
                               % msg)
        elif size > TWO32:
            msg = ('The source model contains point sources: you cannot set '
                   'disagg_by_src=true unless you convert them to multipoint '
                   'sources with the command oq upgrade_nrml --multipoint %s'
                   ) % oq.base_path
            raise RuntimeError(msg)
        return sources

    def init(self):
        super().init()
        if self.oqparam.hazard_calculation_id:
            full_lt = self.datastore.parent['full_lt']
            et_ids = self.datastore.parent['et_ids'][:]
        else:
            full_lt = self.csm.full_lt
            et_ids = self.csm.get_et_ids()
        rlzs_by_gsim_list = full_lt.get_rlzs_by_gsim_list(et_ids)
        rlzs_by_g = []
        for rlzs_by_gsim in rlzs_by_gsim_list:
            for rlzs in rlzs_by_gsim.values():
                rlzs_by_g.append(rlzs)
        self.datastore.hdf5.save_vlen(
            'rlzs_by_g', [U32(rlzs) for rlzs in rlzs_by_g])
        poes_shape = (self.N, len(self.oqparam.imtls.array), len(rlzs_by_g))
        size = numpy.prod(poes_shape) * 8
        logging.info('Requiring %s for ProbabilityMap of shape %s',
                     humansize(size), poes_shape)
        avail = psutil.virtual_memory().available
        if avail < 1.5 * size:
            raise MemoryError(
                'You have only %s of free RAM' % humansize(avail))
        self.datastore.create_dset('_poes', F64, poes_shape)
        if not self.oqparam.hazard_calculation_id:
            self.datastore.swmr_on()

    def execute(self):
        """
        Run in parallel `core_task(sources, sitecol, monitor)`, by
        parallelizing on the sources according to their weight and
        tectonic region type.
        """
        oq = self.oqparam
        if oq.hazard_calculation_id and not oq.compare_with_classical:
            with util.read(self.oqparam.hazard_calculation_id) as parent:
                self.full_lt = parent['full_lt']
            self.calc_stats()  # post-processing
            return {}

        assert oq.max_sites_per_tile > oq.max_sites_disagg, (
            oq.max_sites_per_tile, oq.max_sites_disagg)
        psd = self.set_psd()

        # exit early if we want to perform only a preclassical
        if oq.calculation_mode == 'preclassical':
            recs = [tuple(row) for row in self.csm.source_info.values()]
            self.datastore['source_info'] = numpy.array(
                recs, readinput.source_info_dt)
            self.datastore['full_lt'] = self.csm.full_lt
            self.datastore.swmr_on()  # fixes HDF5 error in build_hazard
            return

        smap = parallel.Starmap(classical, h5=self.datastore.hdf5)
        smap.monitor.save('srcfilter', self.src_filter())
        self.submit_tasks(smap)
        acc0 = self.acc0()  # create the rup/ datasets BEFORE swmr_on()
        self.datastore.swmr_on()
        smap.h5 = self.datastore.hdf5
        self.calc_times = AccumDict(accum=numpy.zeros(3, F32))
        try:
            acc = smap.reduce(self.agg_dicts, acc0)
            self.store_rlz_info(acc.eff_ruptures)
        finally:
            source_ids = self.store_source_info(self.calc_times)
            if self.by_task:
                logging.info('Storing by_task information')
                num_tasks = max(self.by_task) + 1,
                er = self.datastore.create_dset('by_task/eff_ruptures',
                                                U32, num_tasks)
                es = self.datastore.create_dset('by_task/eff_sites',
                                                U32, num_tasks)
                si = self.datastore.create_dset('by_task/srcids',
                                                hdf5.vstr, num_tasks,
                                                fillvalue=None)
                for task_no, rec in self.by_task.items():
                    effrups, effsites, srcids = rec
                    er[task_no] = effrups
                    es[task_no] = effsites
                    si[task_no] = ' '.join(source_ids[s] for s in srcids)
                self.by_task.clear()
        if self.calc_times:  # can be empty in case of errors
            self.numrups = sum(arr[0] for arr in self.calc_times.values())
            numsites = sum(arr[1] for arr in self.calc_times.values())
            logging.info('Effective number of ruptures: {:_d}/{:_d}'.format(
                int(self.numrups), self.totrups))
            logging.info('Effective number of sites per rupture: %d',
                         numsites / self.numrups)
        if psd:
            psdist = max(max(psd.ddic[trt].values()) for trt in psd.ddic)
            if psdist and self.maxradius >= psdist / 2:
                logging.warning('The pointsource_distance of %d km is too '
                                'small compared to a maxradius of %d km',
                                psdist, self.maxradius)
        self.calc_times.clear()  # save a bit of memory
        return acc

    def set_psd(self):
        """
        Set the pointsource_distance
        """
        oq = self.oqparam
        mags = self.datastore['source_mags']  # by TRT
        if len(mags) == 0:  # everything was discarded
            raise RuntimeError('All sources were discarded!?')
        gsims_by_trt = self.full_lt.get_gsims_by_trt()
        mags_by_trt = {}
        for trt in mags:
            mags_by_trt[trt] = mags[trt][()]
        psd = oq.pointsource_distance
        if psd is not None:
            psd.interp(mags_by_trt)
            for trt, dic in psd.ddic.items():
                # the sum is zero for {'default': [(1, 0), (10, 0)]}
                if sum(dic.values()):
                    it = list(dic.items())
                    md = '%s->%d ... %s->%d' % (it[0] + it[-1])
                    logging.info('ps_dist %s: %s', trt, md)
        imts_with_period = [imt for imt in oq.imtls
                            if imt == 'PGA' or imt.startswith('SA')]
        imts_ok = len(imts_with_period) == len(oq.imtls)
        if (imts_ok and psd and psd.suggested()) or (
                imts_ok and oq.minimum_intensity):
            aw = get_effect(mags_by_trt, self.sitecol.one(), gsims_by_trt, oq)
            if psd:
                dic = {trt: [(float(mag), int(dst))
                             for mag, dst in psd.ddic[trt].items()]
                       for trt in psd.ddic if trt != 'default'}
                logging.info('pointsource_distance=\n%s', pprint.pformat(dic))
            if len(vars(aw)) > 1:  # more than _extra
                self.datastore['effect_by_mag_dst'] = aw
        hint = 1 if self.N <= oq.max_sites_disagg else numpy.ceil(
            self.N / oq.max_sites_per_tile)
        self.params = dict(
            truncation_level=oq.truncation_level,
            imtls=oq.imtls, reqv=oq.get_reqv(),
            pointsource_distance=oq.pointsource_distance,
            point_rupture_bins=oq.point_rupture_bins,
            shift_hypo=oq.shift_hypo,
            min_weight=oq.min_weight,
            collapse_level=oq.collapse_level, hint=hint,
            max_sites_disagg=oq.max_sites_disagg,
            split_sources=oq.split_sources, af=self.af)
        return psd

    def submit_tasks(self, smap):
        """
        Submit tasks to the passed Starmap
        """
        oq = self.oqparam
        src_groups = self.csm.src_groups
        tot_weight = 0
        et_ids = self.datastore['et_ids'][:]
        rlzs_by_gsim_list = self.full_lt.get_rlzs_by_gsim_list(et_ids)
        for rlzs_by_gsim, sg in zip(rlzs_by_gsim_list, src_groups):
            for src in sg:
                src.ngsims = len(rlzs_by_gsim)
                tot_weight += src.weight
                if src.code == b'C' and src.num_ruptures > 20_000:
                    msg = ('{} is suspiciously large, containing {:_d} '
                           'ruptures with complex_fault_mesh_spacing={} km')
                    spc = oq.complex_fault_mesh_spacing
                    logging.info(msg.format(src, src.num_ruptures, spc))
        assert tot_weight
        C = oq.concurrent_tasks or 1
        if oq.disagg_by_src or oq.is_ucerf():
            f1, f2 = classical, classical
            max_weight = max(tot_weight / C, oq.min_weight) / 5
        else:
            f1, f2 = classical, start_classical
            max_weight = max(tot_weight / C, oq.min_weight)
        self.params['max_weight'] = max_weight
        logging.info('tot_weight={:_d}, max_weight={:_d}'.format(
            int(tot_weight), int(max_weight)))
        for rlzs_by_gsim, sg in zip(rlzs_by_gsim_list, src_groups):
            nb = 0
            if sg.atomic:
                # do not split atomic groups
                nb += 1
                smap.submit((sg, rlzs_by_gsim, self.params), f1)
            else:  # regroup the sources in blocks
                blks = (groupby(sg, get_source_id).values()
                        if oq.disagg_by_src
                        else block_splitter(sg, 2 * max_weight,
                                            get_weight, sort=True))
                blocks = list(blks)
                nb += len(blocks)
                for block in blocks:
                    logging.debug('Sending %d source(s) with weight %d',
                                  len(block), sum(src.weight for src in block))
                    smap.submit((block, rlzs_by_gsim, self.params), f2)

            w = sum(src.weight for src in sg)
            it = sorted(oq.maximum_distance.ddic[sg.trt].items())
            md = '%s->%d ... %s->%d' % (it[0] + it[-1])
            logging.info('max_dist={}, gsims={}, weight={:_d}, blocks={}'.
                         format(md, len(rlzs_by_gsim), int(w), nb))
        return rlzs_by_gsim_list

    def save_hazard(self, acc, pmap_by_kind):
        """
        Works by side effect by saving hcurves and hmaps on the datastore

        :param acc: ignored
        :param pmap_by_kind: a dictionary of ProbabilityMaps

        kind can be ('hcurves', 'mean'), ('hmaps', 'mean'),  ...
        """
        with self.monitor('saving statistics'):
            for kind in pmap_by_kind:  # i.e. kind == 'hcurves-stats'
                pmaps = pmap_by_kind[kind]
                if kind in ('hmaps-rlzs', 'hmaps-stats'):
                    # pmaps is a list of R pmaps
                    dset = self.datastore.getitem(kind)
                    for r, pmap in enumerate(pmaps):
                        for s in pmap:
                            dset[s, r] = pmap[s].array  # shape (M, P)
                elif kind in ('hcurves-rlzs', 'hcurves-stats'):
                    dset = self.datastore.getitem(kind)
                    for r, pmap in enumerate(pmaps):
                        for s in pmap:
                            dset[s, r] = pmap[s].array.reshape(self.M, self.L1)
            self.datastore.flush()

    def post_execute(self, pmap_by_key):
        """
        Collect the hazard curves by realization and export them.

        :param pmap_by_key:
            a dictionary key -> hazard curves
        """
        nr = {name: len(dset['mag']) for name, dset in self.datastore.items()
              if name.startswith('rup_')}
        if nr:  # few sites, log the number of ruptures per magnitude
            logging.info('%s', nr)
        oq = self.oqparam
        et_ids = self.datastore['et_ids'][:]
        rlzs_by_gsim_list = self.full_lt.get_rlzs_by_gsim_list(et_ids)
        slice_by_g = getters.get_slice_by_g(rlzs_by_gsim_list)
        data = []
        weights = [rlz.weight for rlz in self.realizations]
        pgetter = getters.PmapGetter(
            self.datastore, weights, self.sitecol.sids, oq.imtls)
        logging.info('Saving _poes')
        enum = enumerate(self.datastore['source_info']['source_id'])
        srcid = {source_id: i for i, source_id in enum}
        with self.monitor('saving probability maps'):
            for key, pmap in pmap_by_key.items():
                if isinstance(key, str):  # disagg_by_src
                    rlzs_by_gsim = rlzs_by_gsim_list[pmap.grp_id]
                    self.datastore['disagg_by_src'][..., srcid[key]] = (
                        pgetter.get_hcurves(pmap, rlzs_by_gsim))
                elif pmap:  # pmap can be missing if the group is filtered away
                    # key is the group ID
                    trt = self.full_lt.trt_by_et[et_ids[key][0]]
                    # avoid saving PoEs == 1
                    base.fix_ones(pmap)
                    sids = sorted(pmap)
                    arr = numpy.array([pmap[sid].array for sid in sids])
                    self.datastore['_poes'][sids, :, slice_by_g[key]] = arr
                    extreme = max(
                        get_extreme_poe(pmap[sid].array, oq.imtls)
                        for sid in pmap)
                    data.append((key, trt, extreme))
        if oq.hazard_calculation_id is None and '_poes' in self.datastore:
            self.datastore['disagg_by_grp'] = numpy.array(
                sorted(data), grp_extreme_dt)
            self.datastore.swmr_on()  # needed
            self.calc_stats()

    def calc_stats(self):
        oq = self.oqparam
        hstats = oq.hazard_stats()
        # initialize datasets
        imls = oq.imtls.array
        N = len(self.sitecol.complete)
        P = len(oq.poes)
        M = self.M = len(oq.imtls)
        imts = list(oq.imtls)
        if oq.soil_intensities is not None:
            L = M * len(oq.soil_intensities)
        else:
            L = len(imls)
        L1 = self.L1 = L // M
        R = len(self.realizations)
        S = len(hstats)
        if R > 1 and oq.individual_curves or not hstats:
            self.datastore.create_dset('hcurves-rlzs', F32, (N, R, M, L1))
            self.datastore.set_shape_attrs(
                'hcurves-rlzs', site_id=N, rlz_id=R, imt=imts, lvl=L1)
            if oq.poes:
                self.datastore.create_dset('hmaps-rlzs', F32, (N, R, M, P))
                self.datastore.set_shape_attrs(
                    'hmaps-rlzs', site_id=N, rlz_id=R,
                    imt=list(oq.imtls), poe=oq.poes)
        if hstats:
            self.datastore.create_dset('hcurves-stats', F32, (N, S, M, L1))
            self.datastore.set_shape_attrs(
                'hcurves-stats', site_id=N, stat=list(hstats),
                imt=imts, lvl=numpy.arange(L1))
            if oq.poes:
                self.datastore.create_dset('hmaps-stats', F32, (N, S, M, P))
                self.datastore.set_shape_attrs(
                    'hmaps-stats', site_id=N, stat=list(hstats),
                    imt=list(oq.imtls), poe=oq.poes)
        ct = oq.concurrent_tasks or 1
        logging.info('Building hazard statistics')
        self.weights = [rlz.weight for rlz in self.realizations]
        dstore = (self.datastore.parent if oq.hazard_calculation_id
                  else self.datastore)
        allargs = [  # this list is very fast to generate
            (getters.PmapGetter(
                dstore, self.weights, t.sids, oq.imtls, oq.poes),
             N, hstats, oq.individual_curves, oq.max_sites_disagg,
             self.amplifier)
            for t in self.sitecol.split_in_tiles(ct)]
        if self.few_sites:
            dist = 'no'
        else:
            dist = None  # parallelize as usual
        if oq.hazard_calculation_id is None:  # essential before Starmap
            self.datastore.swmr_on()
        parallel.Starmap(
            build_hazard, allargs, distribute=dist, h5=self.datastore.hdf5
        ).reduce(self.save_hazard)
        if 'hmaps-stats' in self.datastore:
            hmaps = self.datastore.sel('hmaps-stats', stat='mean')  # NSMP
            maxhaz = hmaps.max(axis=(0, 1, 3))
            mh = dict(zip(self.oqparam.imtls, maxhaz))
            logging.info('The maximum hazard map values are %s', mh)
            if Image is None or not self.from_engine:  # missing PIL
                return
            M, P = hmaps.shape[2:]
            logging.info('Saving %dx%d mean hazard maps', M, P)
            inv_time = oq.investigation_time
            allargs = []
            for m, imt in enumerate(self.oqparam.imtls):
                for p, poe in enumerate(self.oqparam.poes):
                    dic = dict(m=m, p=p, imt=imt, poe=poe, inv_time=inv_time,
                               calc_id=self.datastore.calc_id,
                               array=hmaps[:, 0, m, p])
                    allargs.append((dic, self.sitecol.lons, self.sitecol.lats))
            smap = parallel.Starmap(make_hmap_png, allargs)
            for dic in smap:
                self.datastore['png/hmap_%(m)d_%(p)d' % dic] = dic['img']


def make_hmap_png(hmap, lons, lats):
    """
    :param hmap:
        a dictionary with keys calc_id, m, p, imt, poe, inv_time, array
    :param lons: an array of longitudes
    :param lats: an array of latitudes
    :returns: an Image object containing the hazard map
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_title('hmap for IMT=%(imt)s, poe=%(poe)s\ncalculation %(calc_id)d,'
                 'inv_time=%(inv_time)dy' % hmap)
    ax.set_ylabel('Longitude')
    coll = ax.scatter(lons, lats, c=hmap['array'], cmap='jet')
    plt.colorbar(coll)
    bio = io.BytesIO()
    plt.savefig(bio, format='png')
    return dict(img=Image.open(bio), m=hmap['m'], p=hmap['p'])


def build_hazard(pgetter, N, hstats, individual_curves,
                 max_sites_disagg, amplifier, monitor):
    """
    :param pgetter: an :class:`openquake.commonlib.getters.PmapGetter`
    :param N: the total number of sites
    :param hstats: a list of pairs (statname, statfunc)
    :param individual_curves: if True, also build the individual curves
    :param max_sites_disagg: if there are less sites than this, store rup info
    :param amplifier: instance of Amplifier or None
    :param monitor: instance of Monitor
    :returns: a dictionary kind -> ProbabilityMap

    The "kind" is a string of the form 'rlz-XXX' or 'mean' of 'quantile-XXX'
    used to specify the kind of output.
    """
    with monitor('read PoEs'):
        pgetter.init()
        if amplifier:
            with hdf5.File(pgetter.filename, 'r') as f:
                ampcode = f['sitecol'].ampcode
            imtls = DictArray({imt: amplifier.amplevels
                               for imt in pgetter.imtls})
        else:
            imtls = pgetter.imtls
    poes, weights = pgetter.poes, pgetter.weights
    M = len(imtls)
    P = len(poes)
    L = len(imtls.array)
    R = len(weights)
    S = len(hstats)
    pmap_by_kind = {}
    if R > 1 and individual_curves or not hstats:
        pmap_by_kind['hcurves-rlzs'] = [ProbabilityMap(L) for r in range(R)]
        if poes:
            pmap_by_kind['hmaps-rlzs'] = [
                ProbabilityMap(M, P) for r in range(R)]
    if hstats:
        pmap_by_kind['hcurves-stats'] = [ProbabilityMap(L) for r in range(S)]
        if poes:
            pmap_by_kind['hmaps-stats'] = [
                ProbabilityMap(M, P) for r in range(S)]
    combine_mon = monitor('combine pmaps', measuremem=False)
    compute_mon = monitor('compute stats', measuremem=False)
    for sid in pgetter.sids:
        with combine_mon:
            pcurves = pgetter.get_pcurves(sid)
            if amplifier:
                pcurves = amplifier.amplify(ampcode[sid], pcurves)
                # NB: the pcurves have soil levels != IMT levels
        if sum(pc.array.sum() for pc in pcurves) == 0:  # no data
            continue
        with compute_mon:
            if hstats:
                arr = numpy.array([pc.array for pc in pcurves])
                for s, (statname, stat) in enumerate(hstats.items()):
                    pc = getters.build_stat_curve(arr, imtls, stat, weights)
                    pmap_by_kind['hcurves-stats'][s][sid] = pc
                    if poes:
                        hmap = calc.make_hmap(pc, imtls, poes, sid)
                        pmap_by_kind['hmaps-stats'][s].update(hmap)
            if R > 1 and individual_curves or not hstats:
                for pmap, pc in zip(pmap_by_kind['hcurves-rlzs'], pcurves):
                    pmap[sid] = pc
                if poes:
                    for r, pc in enumerate(pcurves):
                        hmap = calc.make_hmap(pc, imtls, poes, sid)
                        pmap_by_kind['hmaps-rlzs'][r].update(hmap)
    return pmap_by_kind
