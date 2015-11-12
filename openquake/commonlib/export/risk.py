#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2014, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import csv
import operator
import itertools
import collections

import numpy

from openquake.baselib.general import AccumDict
from openquake.commonlib.export import export
from openquake.commonlib import writers, risk_writers, riskmodels
from openquake.commonlib.writers import scientificformat
from openquake.commonlib.oqvalidation import OqParam
from openquake.commonlib.export import export_csv
from openquake.commonlib.risk_writers import (
    DmgState, DmgDistPerTaxonomy, DmgDistPerAsset, DmgDistTotal,
    ExposureData, Site)

Output = collections.namedtuple('Output', 'ltype path array')


# ########################## utility functions ############################## #

def compose_arrays(a1, a2):
    """
    Compose composite arrays by generating an extended datatype containing
    all the fields. The two arrays must have the same length.
    """
    assert len(a1) == len(a2),  (len(a1), len(a2))
    if a1.dtype.names is None and len(a1.shape) == 1:
        # the first array is not composite, but it is one-dimensional
        a1 = numpy.array(a1, numpy.dtype([('tag', a1.dtype)]))

    fields1 = [(f, a1.dtype.fields[f][0]) for f in a1.dtype.names]
    if a2.dtype.names is None:  # the second array is not composite
        assert len(a2.shape) == 2, a2.shape
        width = a2.shape[1]
        fields2 = [('value%d' % i, a2.dtype) for i in range(width)]
        composite = numpy.zeros(a1.shape, numpy.dtype(fields1 + fields2))
        for f1 in dict(fields1):
            composite[f1] = a1[f1]
        for i in range(width):
            composite['value%d' % i] = a2[:, i]
        return composite

    fields2 = [(f, a2.dtype.fields[f][0]) for f in a2.dtype.names]
    composite = numpy.zeros(a1.shape, numpy.dtype(fields1 + fields2))
    for f1 in dict(fields1):
        composite[f1] = a1[f1]
    for f2 in dict(fields2):
        composite[f2] = a2[f2]
    return composite

asset_dt = numpy.dtype(
    [('asset_ref', bytes, 20), ('lon', float), ('lat', float)])


def get_assets(dstore):
    """
    :param dstore: a datastore with a key `specific_assets`
    :returns: an ordered array of records (asset_ref, lon, lat)
    """
    assets = sorted(sum(map(list, dstore['assets_by_site']), []),
                    key=operator.attrgetter('id'))
    asset_data = numpy.array(
        [(asset.id, asset.location[0], asset.location[1])
         for asset in assets], asset_dt)
    return asset_data


def get_assets_sites(dstore):
    """
    :param dstore: a datastore with a key `specific_assets`
    :returns: an ordered array of records (asset_ref, lon, lat)
    """
    assetcol = dstore['assetcol']
    sitemesh = dstore['sitemesh']
    rows = []
    for asset in assetcol:
        loc = sitemesh[asset['site_id']]
        rows.append((asset['asset_ref'], loc[0], loc[1]))
    return numpy.array(rows, asset_dt)


def extract_outputs(dkey, dstore, loss_type=None, ext=''):
    """
    An utility to extract outputs ordered by loss types from a datastore
    containing nested structures as follows:


    >> dstore = {
    ..     'risk_output':
    ..        {'structural':
    ..            {'b1': array N x 2,
    ..             'b2': array N x 2,
    ..            }}}
    >> outputs = extract_outputs('risk_output', dstore)
    >> [o.path for o in outputs]
    ['risk_output-structural-b1', 'risk_output-structural-b2']
    >> [o.ltype for o in outputs]
    ['structural', 'structural']
    """
    group = dstore[dkey]
    dashkey = dkey.replace('-rlzs', '').replace('-stats', '')
    if ext and not ext.startswith('.'):
        ext = '.' + ext
    outputs = []
    for ltype in sorted(group):
        subgroup = group[ltype]
        for key in subgroup:
            for i in 0, 1:
                ins = '_ins' if i else ''
                path = dstore.export_path(
                    '%s-%s-%s%s%s' % (dashkey, ltype, key, ins, ext))
                if loss_type:
                    data = subgroup[key][loss_type][:, i]
                else:
                    data = subgroup[key][:, i]
                outputs.append(Output(ltype, path, data))
    return outputs

# ############################### exporters ############################## #


def compactify(array):
    """
    Compactify a composite array of type (name, N1, N2, N3, 2) into a
    composite array of type (name, N1, N2, (N3, 2)). Works with any number
    of Ns.
    """
    if array.shape[-1] != 2:
        raise ValueError('You can only compactify an array which last '
                         'dimension is 2, got shape %s' % array.shape)
    dtype = array.dtype
    pairs = []
    for name in dtype.names:
        dt = dtype.fields[name][0]
        if dt.subdtype is None:
            pairs.append((name, (dt, 2)))
        else:  # this is the case for rcurves
            sdt, shp = dt.subdtype
            pairs.append((name, (sdt, shp + (2,))))
    zeros = numpy.zeros(array.shape[:-1], numpy.dtype(pairs))
    for idx, _ in numpy.ndenumerate(zeros):
        zeros[idx] = array[idx]
    return zeros


# this is used by classical_risk and event_based_risk
@export.add(('avg_losses-rlzs', 'csv'))
def export_avg_losses(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    avg_losses = compactify(dstore[ekey[0]].value)
    rlzs = dstore['rlzs_assoc'].realizations
    assets = get_assets(dstore)
    fnames = []
    for rlz in rlzs:
        losses = avg_losses[:, rlz.ordinal]
        dest = dstore.export_path('avg_losses-rlz%03d.csv' % rlz.ordinal)
        data = compose_arrays(assets, losses)
        writers.write_csv(dest, data, fmt='%10.6E')
        fnames.append(dest)
    return fnames


@export.add(('avg_losses-stats', 'csv'))
def export_avg_losses_stats(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    oq = OqParam.from_(dstore.attrs)
    avg_losses = compactify(dstore[ekey[0]].value)
    quantiles = ['mean'] + ['quantile-%s' % q for q in oq.quantile_loss_curves]
    assets = get_assets(dstore)
    fnames = []
    for i, quantile in enumerate(quantiles):
        losses = avg_losses[:, i]
        dest = dstore.export_path('avg_losses-%s.csv' % quantile)
        data = compose_arrays(assets, losses)
        writers.write_csv(dest, data, fmt='%10.6E')
        fnames.append(dest)
    return fnames


# this is used by classical_risk and event_based_risk
@export.add(('agg_losses-rlzs', 'csv'))
def export_agg_losses(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    agg_losses = compactify(dstore[ekey[0]].value)
    rlzs = dstore['rlzs_assoc'].realizations
    tags = dstore['tags'].value
    fnames = []
    for rlz in rlzs:
        losses = agg_losses[:, rlz.ordinal]
        dest = dstore.export_path('agg_losses-rlz%03d.csv' % rlz.ordinal)
        data = compose_arrays(tags, losses)
        writers.write_csv(dest, data, fmt='%10.6E')
        fnames.append(dest)
    return fnames


# this is used by event_based_risk
@export.add(('agg_losses', 'csv'))
def export_agg_losses_ebr(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    agg_losses = dstore[ekey[0]]
    rlzs = dstore['rlzs_assoc'].realizations
    tags = dstore['tags'].value
    E = len(tags)
    loss_type_dt = dstore['riskmodel'].loss_type_dt
    fnames = []
    for rlz in rlzs:
        array = numpy.zeros((E, 2), loss_type_dt)
        for l, ltype in enumerate(loss_type_dt.names):
            array[ltype] = agg_losses[:, l, rlz.ordinal, :]
        dest = dstore.export_path('agg_losses-rlz%03d.csv' % rlz.ordinal)
        data = compose_arrays(tags, compactify(array))
        writers.write_csv(dest, data, fmt='%10.6E')
        fnames.append(dest)
    return fnames


# alternative export format for the average losses, used by the platform
@export.add(('avglosses-rlzs', 'csv'))
def export_avglosses_csv(ekey, dstore):
    """
    :param ekey: export key, i.e. a pair (datastore key, fmt)
    :param dstore: datastore object
    """
    outs = extract_outputs(ekey[0], dstore, ext=ekey[1])
    sitemesh = dstore['sitemesh']
    assetcol = dstore['assetcol']
    header = ['lon', 'lat', 'asset_ref', 'asset_value', 'average_loss',
              'stddev_loss', 'loss_type']
    for out in outs:
        rows = []
        for asset, loss in zip(assetcol, out.array):
            loc = sitemesh[asset['site_id']]
            row = [loc['lon'], loc['lat'], asset['asset_ref'],
                   asset[out.ltype], loss, numpy.nan, out.ltype]
            rows.append(row)
        writers.write_csv(out.path, [header] + rows)
    return [out.path for out in outs]


@export.add(('rcurves-rlzs', 'csv'))
def export_ebr_curves(ekey, dstore):
    rlzs = dstore['rlzs_assoc'].realizations
    assets = get_assets_sites(dstore)
    curves = compactify(dstore[ekey[0]].value)
    paths = []
    name = ekey[0].split('-')[0]  # rcurves, rmaps
    for rlz in rlzs:
        array = compose_arrays(assets, curves[:, rlz.ordinal])
        path = dstore.export_path('%s-%s.csv' % (name, rlz.uid))
        writers.write_csv(path, array, fmt='%9.7E')
        paths.append(path)
    return paths


@export.add(('dmg_by_asset', 'xml'))
def export_damage(ekey, dstore):
    riskmodel = dstore['riskmodel']
    rlzs = dstore['rlzs_assoc'].realizations
    dmg_by_asset = dstore['dmg_by_asset']  # shape (N, L, R)
    assetcol = dstore['assetcol']
    sitemesh = dstore['sitemesh']
    dmg_states = [DmgState(s, i)
                  for i, s in enumerate(riskmodel.damage_states)]
    D = len(dmg_states)
    N, R = dmg_by_asset.shape
    L = len(riskmodel.loss_types)
    fnames = []

    for l, r in itertools.product(range(L), range(R)):
        lt = riskmodel.loss_types[l]
        rlz = rlzs[r]
        suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (rlz.uid, lt)

        dd_asset = []
        for n in range(N):
            aref = assetcol[n]['asset_ref']
            dist = dmg_by_asset[n, r][lt]
            point = sitemesh[assetcol[n]['site_id']]
            site = Site(point['lon'], point['lat'])
            for ds in range(D):
                dd_asset.append(
                    DmgDistPerAsset(
                        ExposureData(aref, site), dmg_states[ds],
                        dist['mean'][ds], dist['stddev'][ds]))

        f1 = export_dmg_xml(('dmg_dist_per_asset', 'xml'), dstore,
                            dmg_states, dd_asset, suffix)
        max_damage = dmg_states[-1]
        # the collapse map is extracted from the damage distribution per asset
        # (dda) by taking the value corresponding to the maximum damage
        collapse_map = [dda for dda in dd_asset if dda.dmg_state == max_damage]
        f2 = export_dmg_xml(('collapse_map', 'xml'), dstore,
                            dmg_states, collapse_map, suffix)
        fnames.extend(sum((f1 + f2).values(), []))
    return sorted(fnames)


@export.add(('dmg_by_taxon', 'xml'))
def export_damage_taxon(ekey, dstore):
    riskmodel = dstore['riskmodel']
    rlzs = dstore['rlzs_assoc'].realizations
    dmg_by_taxon = dstore['dmg_by_taxon']  # shape (T, L, R)
    taxonomies = dstore['taxonomies']
    dmg_states = [DmgState(s, i)
                  for i, s in enumerate(riskmodel.damage_states)]
    D = len(dmg_states)
    T, R = dmg_by_taxon.shape
    L = len(riskmodel.loss_types)
    fnames = []

    for l, r in itertools.product(range(L), range(R)):
        lt = riskmodel.loss_types[l]
        rlz = rlzs[r]
        suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (rlz.uid, lt)

        dd_taxo = []
        for t in range(T):
            dist = dmg_by_taxon[t, r][lt]
            for ds in range(D):
                dd_taxo.append(
                    DmgDistPerTaxonomy(
                        taxonomies[t], dmg_states[ds],
                        dist['mean'][ds], dist['stddev'][ds]))

        f = export_dmg_xml(('dmg_dist_per_taxonomy', 'xml'),
                           dstore, dmg_states, dd_taxo, suffix)
        fnames.extend(sum(f.values(), []))
    return sorted(fnames)


@export.add(('dmg_total', 'xml'))
def export_damage_total(ekey, dstore):
    riskmodel = dstore['riskmodel']
    rlzs = dstore['rlzs_assoc'].realizations
    dmg_total = dstore['dmg_total']
    R, = dmg_total.shape
    L = len(riskmodel.loss_types)
    dmg_states = [DmgState(s, i)
                  for i, s in enumerate(riskmodel.damage_states)]
    D = len(dmg_states)
    fnames = []
    for l, r in itertools.product(range(L), range(R)):
        lt = riskmodel.loss_types[l]
        rlz = rlzs[r]
        suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (rlz.uid, lt)

        dd_total = []
        for ds in range(D):
            dist = dmg_total[r][lt]
            dd_total.append(DmgDistTotal(
                dmg_states[ds], dist['mean'][ds], dist['stddev'][ds]))

        f = export_dmg_xml(('dmg_dist_total', 'xml'), dstore,
                           dmg_states, dd_total, suffix)
        fnames.extend(sum(f.values(), []))
    return sorted(fnames)


@export.add(('csq_by_asset', 'csv'), ('csq_by_taxon', 'csv'))
def export_csq_csv(ekey, dstore):
    rlzs = dstore['rlzs_assoc'].realizations
    R = len(rlzs)
    value = dstore[ekey[0]].value  # matrix N x R or T x R
    fnames = []
    for rlz, values in zip(rlzs, value.T):
        suffix = '.csv' if R == 1 else '-gsimltp_%s.csv' % rlz.uid
        fname = dstore.export_path(ekey[0] + suffix)
        writers.write_csv(fname, values)
        fnames.append(fname)
    return fnames


@export.add(('csq_total', 'csv'))
def export_csq_total_csv(ekey, dstore):
    rlzs = dstore['rlzs_assoc'].realizations
    R = len(rlzs)
    value = dstore[ekey[0]].value
    fnames = []
    for rlz, values in zip(rlzs, value):
        suffix = '.csv' if R == 1 else '-gsimltp_%s.csv' % rlz.uid
        fname = dstore.export_path(ekey[0] + suffix)
        writers.write_csv(fname, numpy.array([values], value.dtype))
        fnames.append(fname)
    return fnames


export.add(('dmg_by_asset', 'csv'), ('dmg_by_taxon', 'csv'),
           ('dmg_total', 'csv'))(export_csv)


def export_dmg_xml(key, dstore, damage_states, dmg_data, suffix):
    """
    Export damage outputs in XML format.

    :param key:
        dmg_dist_per_asset|dmg_dist_per_taxonomy|dmg_dist_total|collapse_map
    :param dstore:
        the datastore
    :param damage_states:
        the list of damage states
    :param dmg_data:
        a list [(loss_type, unit, asset_ref, mean, stddev), ...]
    :param suffix:
        a suffix specifying the GSIM realization
    """
    dest = dstore.export_path('%s%s.%s' % (key[0], suffix, key[1]))
    risk_writers.DamageWriter(damage_states).to_nrml(key[0], dmg_data, dest)
    return AccumDict({key: [dest]})


@export.add(('damages_by_rlz', 'csv'))
def export_classical_damage_csv(ekey, dstore):
    damages_by_rlz = dstore['damages_by_rlz']
    rlzs = dstore['rlzs_assoc'].realizations
    damage_states = dstore['riskmodel'].damage_states
    dmg_states = [DmgState(s, i) for i, s in enumerate(damage_states)]
    fnames = []
    for rlz in rlzs:
        damages = damages_by_rlz[rlz.ordinal]
        fname = 'damage_%d.csv' % rlz.ordinal
        fnames.append(
            _export_classical_damage_csv(dstore, fname, dmg_states, damages))
    return fnames


def _export_classical_damage_csv(dstore, fname, damage_states,
                                 fractions_by_asset):
    """
    Export damage fractions in CSV.

    :param dstore: the datastore
    :param fname: the name of the exported file
    :param damage_states: the damage states
    :fractions_by_asset: a dictionary with the fractions by asset
    """
    dest = dstore.export_path(fname)
    with open(dest, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='|', lineterminator='\n')
        writer.writerow(['asset_ref'] + [ds.dmg_state for ds in damage_states])
        for asset_ref in sorted(fractions_by_asset):
            data = fractions_by_asset[asset_ref]
            writer.writerow([asset_ref] + list(map(scientificformat, data)))
    return dest


# exports for scenario_risk

AggLoss = collections.namedtuple(
    'AggLoss', 'loss_type unit mean stddev')

PerAssetLoss = collections.namedtuple(
    'PerAssetLoss', 'loss_type unit asset_ref mean stddev')


@export.add(('avglosses-rlzs', 'csv'))
def export_avglosses(ekey, dstore):
    unit_by_lt = {riskmodels.cost_type_to_loss_type(ct['name']): ct['unit']
                  for ct in dstore['cost_types']}
    unit_by_lt['fatalities'] = 'people'
    rlzs = dstore['rlzs_assoc'].realizations
    avglosses = dstore[ekey[0]]
    riskmodel = dstore['riskmodel']
    assets = dstore['assetcol']['asset_ref']
    N, R = avglosses.shape
    L = len(riskmodel.loss_types)
    fnames = []
    for l, lt in enumerate(riskmodel.loss_types):
        alosses = avglosses[lt]
        for r in range(R):
            rlz = rlzs[r]
            lt = riskmodel.loss_types[l]
            unit = unit_by_lt[lt]
            suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (
                rlz.uid, lt)
            losses = [PerAssetLoss(lt, unit, ass, stat['mean'], stat['stddev'])
                      for ass, stat in zip(assets, alosses[:, r])]
            out = export_loss_csv(('avg', 'csv'), dstore, losses, suffix)
            fnames.append(out)
    return sorted(fnames)

LossMap = collections.namedtuple('LossMap', 'location asset_ref value std_dev')


# emulate a Django point
class Location(object):
    def __init__(self, xy):
        self.x, self.y = xy
        self.wkt = 'POINT(%s %s)' % tuple(xy)


@export.add(('loss_maps-rlzs', 'xml'), ('loss_maps-rlzs', 'geojson'))
def export_loss_maps_xml_geojson(ekey, dstore):
    oq = OqParam.from_(dstore.attrs)
    unit_by_lt = {riskmodels.cost_type_to_loss_type(ct['name']): ct['unit']
                  for ct in dstore['cost_types']}
    unit_by_lt['fatalities'] = 'people'
    rlzs = dstore['rlzs_assoc'].realizations
    loss_maps = dstore[ekey[0]]
    riskmodel = dstore['riskmodel']
    assetcol = dstore['assetcol']
    R = len(rlzs)
    sitemesh = dstore['sitemesh']
    L = len(riskmodel.loss_types)
    fnames = []
    export_type = ekey[1]
    writercls = (risk_writers.LossMapGeoJSONWriter
                 if export_type == 'geojson' else
                 risk_writers.LossMapXMLWriter)
    for lt in riskmodel.loss_types:
        alosses = loss_maps[lt]
        for r, lmaps in enumerate(alosses.values()):
            for p, poe in enumerate(oq.conditional_loss_poes):
                for ins in range(oq.insured_losses + 1):
                    rlz = rlzs[r]
                    unit = unit_by_lt[lt]
                    suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (
                        rlz.uid, lt)
                    name = '%s%s-poe-%s%s.%s' % (
                        ekey[0], suffix, poe, '_ins' if ins else '', ekey[1])
                    fname = dstore.export_path(name)
                    data = []
                    for ass, stat in zip(assetcol, lmaps[:, p]):
                        loc = Location(sitemesh[ass['site_id']])
                        lm = LossMap(loc, ass['asset_ref'], stat[ins], None)
                        data.append(lm)
                    writer = writercls(
                        fname, oq.investigation_time, poe=poe, loss_type=lt,
                        gsim_tree_path=rlz.gsim_rlz.uid, unit=unit)
                    writer.serialize(data)
                    fnames.append(fname)
    return sorted(fnames)


@export.add(('loss_maps-stats', 'xml'), ('loss_stats-rlzs', 'geojson'))
def export_loss_maps_stats_xml_geojson(ekey, dstore):
    oq = OqParam.from_(dstore.attrs)
    unit_by_lt = {riskmodels.cost_type_to_loss_type(ct['name']): ct['unit']
                  for ct in dstore['cost_types']}
    unit_by_lt['fatalities'] = 'people'
    rlzs = dstore['rlzs_assoc'].realizations
    loss_maps = dstore[ekey[0]]
    riskmodel = dstore['riskmodel']
    assetcol = dstore['assetcol']
    R = len(rlzs)
    sitemesh = dstore['sitemesh']
    L = len(riskmodel.loss_types)
    fnames = []
    export_type = ekey[1]
    writercls = (risk_writers.LossMapGeoJSONWriter
                 if export_type == 'geojson' else
                 risk_writers.LossMapXMLWriter)
    for lt in riskmodel.loss_types:
        alosses = loss_maps[lt]
        for r, lmaps in enumerate(alosses.values()):
            for p, poe in enumerate(oq.conditional_loss_poes):
                for ins in range(oq.insured_losses + 1):
                    rlz = rlzs[r]
                    unit = unit_by_lt[lt]
                    suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (
                        rlz.uid, lt)
                    name = '%s%s-poe-%s%s.%s' % (
                        ekey[0], suffix, poe, '_ins' if ins else '', ekey[1])
                    fname = dstore.export_path(name)
                    data = []
                    for ass, stat in zip(assetcol, lmaps[:, p]):
                        loc = Location(sitemesh[ass['site_id']])
                        lm = LossMap(loc, ass['asset_ref'], stat[ins], None)
                        data.append(lm)
                    writer = writercls(
                        fname, oq.investigation_time, poe=poe, loss_type=lt,
                        gsim_tree_path=rlz.gsim_rlz.uid, unit=unit)
                    writer.serialize(data)
                    fnames.append(fname)
    return sorted(fnames)


# this is used by scenario_risk
@export.add(('agglosses-rlzs', 'csv'))
def export_agglosses(ekey, dstore):
    unit_by_lt = {riskmodels.cost_type_to_loss_type(ct['name']): ct['unit']
                  for ct in dstore['cost_types']}
    unit_by_lt['fatalities'] = 'people'
    rlzs = dstore['rlzs_assoc'].realizations
    agglosses = dstore[ekey[0]]
    riskmodel = dstore['riskmodel']
    L = len(riskmodel.loss_types)
    R, = agglosses.shape
    fnames = []
    for lt in riskmodel.loss_types:
        for r in range(R):
            rlz = rlzs[r]
            unit = unit_by_lt[lt]
            suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (
                rlz.uid, lt)
            loss = agglosses[r][lt]
            losses = [AggLoss(lt, unit, loss['mean'], loss['stddev'])]
            out = export_loss_csv(('agg', 'csv'), dstore, losses, suffix)
            fnames.append(out)
    return sorted(fnames)


def export_loss_csv(key, dstore, data, suffix):
    """
    Export (aggregate) losses in CSV.

    :param key: per_asset_loss|asset-ins
    :param dstore: the datastore
    :param data: a list [(loss_type, unit, asset_ref, mean, stddev), ...]
    :param suffix: a suffix specifying the GSIM realization
    """
    dest = dstore.export_path('%s%s.%s' % (key[0], suffix, key[1]))
    if key[0] in ('agg', 'ins'):  # aggregate
        header = ['LossType', 'Unit', 'Mean', 'Standard Deviation']
    else:
        header = ['LossType', 'Unit', 'Asset', 'Mean', 'Standard Deviation']
        data.sort(key=operator.itemgetter(2))  # order by asset_ref
    writers.write_csv(dest, [header] + data, fmt='%11.7E')
    return dest

AggCurve = collections.namedtuple(
    'AggCurve', ['losses', 'poes', 'average_loss', 'stddev_loss'])


def _gen_writers(dstore, writercls, root):
    # build XMLWriter instances
    oq = OqParam.from_(dstore.attrs)
    rlzs = dstore['rlzs_assoc'].realizations
    cost_types = dstore['cost_types']
    L, R = len(cost_types), len(rlzs)
    for l, ct in enumerate(cost_types):
        loss_type = riskmodels.cost_type_to_loss_type(ct['name'])
        for ins in range(oq.insured_losses + 1):
            if root.endswith('-rlzs'):
                for rlz in rlzs:
                    suffix = '' if L == 1 and R == 1 else '-gsimltp_%s_%s' % (
                        rlz.uid, loss_type)
                    dest = dstore.export_path('%s%s%s.xml' % (
                        root[:-5], suffix, '_ins' if ins else ''))
                    yield writercls(
                        dest, oq.investigation_time, loss_type,
                        source_model_tree_path='_'.join(rlz.sm_lt_path),
                        gsim_tree_path='_'.join(rlz.gsim_lt_path),
                        unit=ct['unit']), (l, rlz.ordinal, ins)
            elif root.endswith('-stats'):  # stats
                pairs = [('mean', None)] + [
                    ('quantile-%s' % q, q) for q in oq.quantile_loss_curves]
                for ordinal, (statname, statvalue) in enumerate(pairs):
                    dest = dstore.export_path('%s-%s-%s%s.xml' % (
                        root[:-6], statname, loss_type, '_ins' if ins else ''))
                    yield writercls(
                        dest, oq.investigation_time, loss_type,
                        statistics='mean' if ordinal == 0 else 'quantile',
                        quantile_value=statvalue, unit=ct['unit']
                    ), (loss_type, statname, ins)


# this is used by event_based_risk
@export.add(('agg_curve-rlzs', 'xml'))
def export_agg_curve(ekey, dstore):
    agg_curve = dstore[ekey[0]]
    fnames = []
    for writer, (l, r, ins) in _gen_writers(
            dstore, risk_writers.AggregateLossCurveXMLWriter, ekey[0]):
        rec = agg_curve[l, r, ins]
        curve = AggCurve(rec['losses'], rec['poes'], rec['avg'], None)
        writer.serialize(curve)
        fnames.append(writer._dest)
    return sorted(fnames)


# this is used by event_based_risk
@export.add(('agg_curve-stats', 'xml'))
def export_agg_curve_stats(ekey, dstore):
    agg_curve = dstore[ekey[0]]
    fnames = []
    for writer, (loss_type, statname, ins) in _gen_writers(
            dstore, risk_writers.AggregateLossCurveXMLWriter, ekey[0]):
        rec = agg_curve[loss_type][statname][ins]
        curve = AggCurve(rec['losses'], rec['poes'], rec['avg'], None)
        writer.serialize(curve)
        fnames.append(writer._dest)
    return sorted(fnames)
