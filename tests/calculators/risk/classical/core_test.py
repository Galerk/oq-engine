# Copyright (c) 2010-2012, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import mock
from tests.utils import helpers
from tests.calculators.risk import general_test

from openquake.db import models
from openquake.calculators.risk.classical import core as classical


class ClassicalRiskCalculatorTestCase(general_test.BaseRiskCalculatorTestCase):
    """
    Integration test for the classical risk calculator
    """
    def setUp(self):
        super(ClassicalRiskCalculatorTestCase, self).setUp()

        self.calculator = classical.ClassicalRiskCalculator(self.job)

    def test_celery_task(self):
        self.calculator.pre_execute()
        self.job.is_running = True
        self.job.status = 'executing'
        self.job.save()

        patch = helpers.patch(
            'openquake.calculators.risk.general.write_loss_curve')
        mocked_writer = patch.start()

        classical.classical(*self.calculator.task_arg_gen(
            self.calculator.block_size()).next())

        patch.stop()

        # we expect 1 asset being filtered out by the region
        # constraint, so there are only two loss curves to be written
        self.assertEqual(2, mocked_writer.call_count)

    def test_complete_workflow(self):
        """
        Test the complete risk classical calculation workflow and test
        for the presence of the outputs
        """
        self.calculator.pre_execute()

        self.job.is_running = True
        self.job.status = 'executing'
        self.job.save()
        self.calculator.execute()

        self.assertEqual(4, models.Output.objects.filter(
            oq_job=self.job).count())

        self.assertEqual(1, models.LossCurve.objects.filter(
            output__oq_job=self.job).count())

        self.assertEqual(2, models.LossCurveData.objects.filter(
            loss_curve__output__oq_job=self.job).count())

        self.assertEqual(6, models.LossMapData.objects.filter(
            loss_map__output__oq_job=self.job).count())

        files = self.calculator.export(exports='xml')
        self.assertEqual(4, len(files))

    def test_hazard_id(self):
        """
        Test that the hazard output used by the calculator is a
        `openquake.db.models.HazardCurve` object
        """

        self.assertEqual(1, models.HazardCurve.objects.filter(
            pk=self.calculator.hazard_id).count())

    def test_create_outputs(self):
        """
        Test that the proper output containers are created
        """

        loss_curve_id = self.calculator.create_loss_curve_output()
        loss_maps_ids = self.calculator.create_loss_maps_outputs()

        self.assertTrue(models.LossCurve.objects.filter(
            pk=loss_curve_id).exists())

        self.assertEqual(
            sorted(self.job.risk_calculation.conditional_loss_poes),
            sorted(loss_maps_ids.keys()))

        for _, map_id in loss_maps_ids.items():
            self.assertTrue(models.LossMap.objects.filter(
                pk=map_id).exists())
