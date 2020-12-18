# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2019-2020, GEM Foundation
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
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy

from openquake.baselib import general, datastore, parallel
from openquake.hazardlib.stats import set_rlzs_stats
from openquake.risklib import scientific
from openquake.calculators import base, views

F32 = numpy.float32
U32 = numpy.uint32


def get_loss_builder(dstore, return_periods=None, loss_dt=None):
    """
    :param dstore: datastore for an event based risk calculation
    :returns: a LossCurvesMapsBuilder instance
    """
    oq = dstore['oqparam']
    weights = dstore['weights'][()]
    eff_time = oq.investigation_time * oq.ses_per_logic_tree_path
    num_events = numpy.bincount(dstore['events']['rlz_id'])
    periods = return_periods or oq.return_periods or scientific.return_periods(
        eff_time, num_events.max())
    return scientific.LossCurvesMapsBuilder(
        oq.conditional_loss_poes, numpy.array(periods),
        loss_dt or oq.loss_dt(), weights, dict(enumerate(num_events)),
        eff_time, oq.risk_investigation_time)


def get_src_loss_table(dstore, L):
    """
    :returns:
        (source_ids, array of losses of shape (Ns, L))
    """
    alt = dstore.read_df('agg_loss_table', 'agg_id').loc[0]
    evs = dstore['events'][()]
    rlz_ids = evs['rlz_id'][alt.event_id]
    rup_ids = evs['rup_id'][alt.event_id]
    source_id = dstore['ruptures']['source_id'][rup_ids]
    w = dstore['weights'][:]
    acc = general.AccumDict(accum=numpy.zeros(L, F32))
    for source_id, rlz_id, loss in zip(source_id, rlz_ids, alt['loss']):
        acc[source_id] += loss * w[rlz_id]
    return zip(*sorted(acc.items()))


def post_risk(builder, krl_losses, monitor):
    """
    :returns: dictionary krl -> loss curve
    """
    res = {}
    for krl, losses in krl_losses:
        res[krl] = builder.build_curves(losses, krl[1])
    return res


@base.calculators.add('post_risk')
class PostRiskCalculator(base.RiskCalculator):
    """
    Compute losses and loss curves starting from an event loss table.
    """
    def pre_execute(self):
        oq = self.oqparam
        if oq.hazard_calculation_id and not self.datastore.parent:
            self.datastore.parent = datastore.read(oq.hazard_calculation_id)
        self.L = len(oq.loss_names)
        self.tagcol = self.datastore['assetcol/tagcol']

    def execute(self):
        oq = self.oqparam
        if oq.return_periods != [0]:
            # setting return_periods = 0 disable loss curves
            eff_time = oq.investigation_time * oq.ses_per_logic_tree_path
            if eff_time < 2:
                logging.warning(
                    'eff_time=%s is too small to compute loss curves',
                    eff_time)
                return
        '''
        if 'source_info' in self.datastore:  # missing for gmf_ebrisk
            logging.info('Building src_loss_table')
            source_ids, losses = get_src_loss_table(self.datastore, self.L)
            self.datastore['src_loss_table'] = losses
            self.datastore.set_shape_attrs('src_loss_table',
                                           source=source_ids,
                                           loss_type=oq.loss_names)
        '''
        builder = get_loss_builder(self.datastore)
        try:
            K = len(self.datastore['assetcol/aggkeys']) + 1
        except KeyError:  # no aggregations
            K = 1
        P = len(builder.return_periods)
        # do everything in process since it is really fast
        rlz_id = self.datastore['events']['rlz_id']
        alt_df = self.datastore.read_df('agg_loss_table', 'agg_id')
        alt_df['rlz_id'] = rlz_id[alt_df.event_id.to_numpy()]

        with self.monitor('agg_losses and agg_curves', measuremem=True):
            smap = parallel.Starmap(post_risk, h5=self.datastore.hdf5)
            num_curves = K * self.R * self.L
            blocksize = numpy.ceil(num_curves / (oq.concurrent_tasks or 1))
            krl_losses = []
            agg_losses = numpy.zeros((K, self.R, self.L), F32)
            agg_curves = numpy.zeros((K, self.R, self.L, P), F32)
            gb = alt_df.groupby([alt_df.index, alt_df.rlz_id])
            logging.info('Computing at max %d/%d curves per task',
                         blocksize, num_curves)
            for (k, r), df in gb:
                for l, lname in enumerate(oq.loss_names):
                    krl = k, r, l
                    agg_losses[krl] = df[lname].sum() * oq.ses_ratio
                    krl_losses.append((krl, df[lname]))
                    if len(krl_losses) >= blocksize:
                        smap.submit((builder, krl_losses))
                        krl_losses[:] = []
            if krl_losses:
                smap.submit((builder, krl_losses))
            for krl, arr in smap.reduce().items():
                agg_curves[krl] = arr
            self.datastore['agg_losses-rlzs'] = agg_losses
            self.datastore['agg_curves-rlzs'] = agg_curves

        units = self.datastore['cost_calculator'].get_units(oq.loss_names)
        set_rlzs_stats(self.datastore, 'agg_curves',
                       agg_ids=K, loss_types=oq.loss_names,
                       return_periods=builder.return_periods,
                       units=units)
        set_rlzs_stats(self.datastore, 'agg_losses',
                       agg_ids=K, loss_types=oq.loss_names, units=units)
        return 1

    def post_execute(self, dummy):
        """
        Sanity check on tot_losses
        """
        logging.info('Mean portfolio loss\n' +
                     views.view('portfolio_loss', self.datastore))
        logging.info('Sanity check on agg_losses')
        # TODO: add it
